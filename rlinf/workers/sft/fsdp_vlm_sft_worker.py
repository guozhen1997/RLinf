# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.fp32_head import Fp32HeadLogProbs
from rlinf.hybrid_engines.fsdp.utils import (
    activation_cpu_offload_context,
    generate_with_kv_cache,
)
from rlinf.utils.distributed import all_reduce_dict
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
from rlinf.workers.sft.utils import vlm_extract_answer, vlm_normalize_text

_VLM_SFT_SUPPORTED_MODELS = (
    SupportedModel.QWEN2_5_VL_SFT,
    SupportedModel.QWEN3_VL_SFT,
    SupportedModel.QWEN3_VL_MOE_SFT,
    SupportedModel.QWEN3_5_SFT,
    SupportedModel.QWEN3_8_SFT,
)


class FSDPVlmSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        is_qwen3_8 = model_type in (
            SupportedModel.QWEN3_5_SFT,
            SupportedModel.QWEN3_8_SFT,
        )
        self._use_fp32_head = bool(self.cfg.actor.get("use_fp32_head", is_qwen3_8))
        self._eval_mode = str(self.cfg.actor.get("eval_mode", "generate")).lower()
        self._eval_max_new_tokens = int(
            self.cfg.actor.get("eval_max_new_tokens", 128)
        )

        fsdp_cfg = self.cfg.actor.get("fsdp_config", {})
        self._activation_cpu_offload = bool(
            fsdp_cfg.get("activation_cpu_offload", False)
        )
        self._activation_cpu_offload_pin_memory = bool(
            fsdp_cfg.get("activation_cpu_offload_pin_memory", True)
        )
        gpu_limit_gb = float(
            fsdp_cfg.get("activation_cpu_offload_gpu_limit_gb", 0.0)
        )
        self._activation_cpu_offload_gpu_limit_bytes = (
            int(gpu_limit_gb * (1024**3)) if gpu_limit_gb > 0 else 0
        )

    def _activation_offload_context(self):
        return activation_cpu_offload_context(
            enabled=self._activation_cpu_offload and torch.is_grad_enabled(),
            pin_memory=self._activation_cpu_offload_pin_memory,
            gpu_memory_threshold_bytes=self._activation_cpu_offload_gpu_limit_bytes,
        )

    def _prepare_multi_modal_inputs(
        self, raw_mm: dict[str, Any] | None
    ) -> dict[str, torch.Tensor]:
        if not raw_mm:
            return {}
        prepared: dict[str, torch.Tensor] = {}
        for k, v in raw_mm.items():
            if v is None:
                continue
            if isinstance(v, list):
                tensors = [t for t in v if t is not None]
                if not tensors:
                    continue
                prepared[k] = torch.cat(tensors, dim=0).to(device=self.device)
            else:
                prepared[k] = v.to(device=self.device)
        return prepared

    def _save_data_state(self, save_path: str):
        state = {
            "data_epoch": self._data_epoch,
            "data_iter_offset": self._data_iter_offset,
        }
        with open(os.path.join(save_path, "data_state.json"), "w") as f:
            json.dump(state, f)

    def save_checkpoint(self, save_path: str, step: int = 0):
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            self._save_data_state(save_path)

    def _load_data_state(self, load_path: str):
        path = os.path.join(load_path, "data_state.json")
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            state = json.load(f)
        self._data_epoch = int(state.get("data_epoch", 0))
        self._data_iter_offset = int(state.get("data_iter_offset", 0))

        if hasattr(self.data_loader, "sampler") and hasattr(
            self.data_loader.sampler, "set_epoch"
        ):
            self.data_loader.sampler.set_epoch(self._data_epoch)

        self.data_iter = iter(self.data_loader)
        for _ in range(self._data_iter_offset):
            try:
                next(self.data_iter)
            except StopIteration:
                self._data_epoch += 1
                if hasattr(self.data_loader, "sampler") and hasattr(
                    self.data_loader.sampler, "set_epoch"
                ):
                    self.data_loader.sampler.set_epoch(self._data_epoch)
                self.data_iter = iter(self.data_loader)

    def load_checkpoint(self, load_path: str):
        super().load_checkpoint(load_path)
        if self.data_loader is not None:
            # run the eval model not to load data_loader ckpt
            self._load_data_state(load_path)

    def build_tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.actor.model.model_path,
            trust_remote_code=self.cfg.actor.model.get("trust_remote_code", True),
        )
        # set the padding side to left for the tokenizer, QWEN 2.5 VL just use left padding
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in _VLM_SFT_SUPPORTED_MODELS:
            from torch.utils.data import DataLoader, DistributedSampler

            from rlinf.data.datasets.vlm import collate_fn, create_vlm_datasets

            # vlm sft before load dataloader should build the tokenizer
            if not hasattr(self, "tokenizer"):
                self.tokenizer = self.build_tokenizer()

            train_dataset, _ = create_vlm_datasets(
                self.cfg,
                self.tokenizer,
                data_paths=data_paths,
                eval_dataset=eval_dataset,
            )
            dataset_name = self.cfg.data.get("dataset_name", "robo2vlmsft")

            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=self.cfg.data.get("shuffle", True),
                    seed=self.cfg.data.get("seed", 42),
                    drop_last=True,
                )
            else:
                sampler = None

            batch_size = (
                self.micro_batch_size
                if not eval_dataset
                else self.cfg.actor.get("eval_batch_size", 1)
            )
            data_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=self.cfg.data.get("num_workers", 4),
                drop_last=True,
                collate_fn=collate_fn,
            )
            logging.info(
                f"Build data loader from {data_paths} with {len(train_dataset)} samples"
            )
            assert len(data_loader) != 0, (
                f"data_loader is not empty, please check the data_path {data_paths}"
            )

            data_config = {
                "dataset_name": dataset_name,
                "num_samples": len(train_dataset),
            }

            return data_loader, data_config

        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def run_eval(self):
        if self._eval_mode != "loss":
            return super().run_eval()

        assert self.eval_data_loader is not None, "eval_data_loader is not set"
        eval_data_iter = iter(self.eval_data_loader)
        with self.worker_timer():
            self.model.eval()
            total_loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for batch in eval_data_iter:
                    loss, _ = self.get_train_model_output(batch)
                    total_loss += float(loss.detach().item())
                    num_batches += 1
            metrics = {
                "eval_loss": total_loss / max(1, num_batches),
            }
            return all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)

    def get_eval_model_output(self, batch: dict[str, Any]):
        # handle the input batch
        correct = 0
        input_ids = batch["prompt"].to(self.device)
        answers = batch["answer"]
        attention_mask = batch["attention_mask"].to(self.device, dtype=torch.long)
        multi_modal_inputs = self._prepare_multi_modal_inputs(
            batch.get("multi_modal_inputs")
        )

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (eos_token_id if eos_token_id is not None else 0)
        )

        with torch.no_grad():
            # use kv cache to generate the text
            # the generate_with_kv_cache() is more efficient than the generate() in utils.py
            generate_ids = generate_with_kv_cache(
                model=self.model,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                amp_context=self.amp_context,
                input_ids=input_ids,
                attention_mask=attention_mask,
                multi_modal_inputs=multi_modal_inputs,
                max_new_tokens=self._eval_max_new_tokens,
            )

        # encode the generated text
        for i in range(len(answers)):
            new_token_ids = generate_ids[i, input_ids.shape[1] :]
            full_pred_text = self.tokenizer.decode(
                new_token_ids.tolist(), skip_special_tokens=False
            )

            pred_text = vlm_extract_answer(
                full_pred_text, self.cfg.actor.model.model_type
            )
            gold_text = answers[i]

            if vlm_normalize_text(pred_text) == vlm_normalize_text(gold_text):
                correct += 1

        # eval model return the correct number of answers
        return correct

    def _forward_chunked_fp32_head_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        multi_modal_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute exact FP32 next-token CE loss via chunked LM-head GEMM.

        Avoids materializing full ``[batch, seq_len, vocab_size]`` logits while
        routing through the root module so FSDP2 unshards ``lm_head.weight``
        and attaches pre-/post-backward hooks cleanly.
        """
        backbone = self.model.model
        lm_head = self.model.get_output_embeddings()

        shift_labels = labels[:, 1:]
        valid = shift_labels != -100
        captured_hidden: list[torch.Tensor] = []

        def _capture_backbone(_module, _args, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output.last_hidden_state
            captured_hidden.append(hidden)

        def _chunked_ce_hook(_module, _args, _output):
            hidden = captured_hidden.pop()
            if not bool(valid.any()):
                return hidden.sum() * 0.0
            shift_hidden = hidden[:, :-1, :]
            lm_weight = lm_head.weight
            selected_hidden = shift_hidden[valid].to(dtype=lm_weight.dtype)
            selected_ids = shift_labels[valid].unsqueeze(-1)
            log_probs, _ = Fp32HeadLogProbs.apply(
                selected_hidden, lm_weight, selected_ids, 1.0
            )
            return -log_probs[:, 0].mean()

        backbone_hook = backbone.register_forward_hook(_capture_backbone)
        head_hook = lm_head.register_forward_hook(_chunked_ce_hook)
        try:
            with self.amp_context, self._activation_offload_context():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=False,
                    logits_to_keep=slice(0, 0),
                    **multi_modal_inputs,
                )
        finally:
            backbone_hook.remove()
            head_hook.remove()

        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits

    def get_train_model_output(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # handle the input batch
        input_ids = batch["prompt"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device, dtype=torch.long)
        multi_modal_inputs = self._prepare_multi_modal_inputs(
            batch.get("multi_modal_inputs")
        )
        label_mask = batch["label_mask"].to(device=self.device, dtype=torch.bool)

        labels = input_ids.detach().clone().masked_fill(~attention_mask.bool(), -100)
        # label_mask masks prompt/non-assistant tokens so loss is computed only on targets
        labels = labels.masked_fill(label_mask, -100)

        can_use_fp32_head = (
            self._use_fp32_head
            and hasattr(self.model, "model")
            and hasattr(self.model, "get_output_embeddings")
            and self.model.get_output_embeddings() is not None
        )
        if can_use_fp32_head:
            loss = self._forward_chunked_fp32_head_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                multi_modal_inputs=multi_modal_inputs,
            )
        else:
            with self.amp_context, self._activation_offload_context():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                    **multi_modal_inputs,
                )
            loss = outputs.loss
        return loss, {"loss": loss.detach().item()}
