# Copyright 2025 The RLinf Authors.
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

import os
from typing import ContextManager, Dict, Union

import torch
from omegaconf import DictConfig
from packaging import version
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

from rlinf.config import torch_dtype_from_precision
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import (
    create_device_mesh,
)
from rlinf.utils.logging import get_logger

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
    )
else:
    MixedPrecisionPolicy, CPUOffloadPolicy = None, None


class FSDPModelManager:
    """
    FSDP Model Manager for RL training
    """

    def __init__(self, cfg: DictConfig, world_size: int, rank: int) -> None:
        """
        Initialize FSDP Model Manager.

        Assumes:
            - torch.distributed has been initialized outside before calling this constructor.
            - all cfg parameters are validated in `valid_fsdp_config`.

        Params:
            cfg: actor config in yaml file.
            world_size: total number of FSDP actor processes.
        """
        self._cfg = cfg
        self._logger = get_logger()
        self.torch_dtype = torch_dtype_from_precision(self._cfg.model.precision)

        self.use_fp16 = self.torch_dtype == torch.float16
        if cfg.get("tokenizer", {}).get("tokenizer_model", None) is not None:
            self.tokenizer = hf_tokenizer(cfg.tokenizer.tokenizer_model)

        self.world_size = world_size
        self.rank = rank

        self._strategy = FSDPStrategyBase.create(
            self._cfg, self.world_size, self.rank, self._logger
        )

        self._device_mesh = create_device_mesh(
            world_size, self._cfg.fsdp_config.get("fsdp_size", -1)
        )
        self._dp_group = (
            self._device_mesh["ddp"].get_group()
            if "ddp" in self._device_mesh.mesh_dim_names
            else None
        )

    def model_provider_func(self) -> torch.nn.Module:
        """
        Initialize model used by FSDP actor
        """
        cfg = self._cfg
        use_gptq = cfg.model.get("gptq_model", False)
        load_in_8bit = cfg.model.get("load_in_8bit", False)

        use_triton = cfg.get("use_triton", True)

        assert torch.cuda.is_available(), "CUDA is not available."
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")

        model_config = AutoConfig.from_pretrained(
            cfg.model.model_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        if use_gptq:
            from auto_gptq import AutoGPTQForCausalLM  # type: ignore[import-not-found]

            model_wrapper = AutoGPTQForCausalLM.from_quantized(
                cfg.model.model_path,
                device=device,
                use_triton=use_triton,
            )
            model = model_wrapper.model
        elif load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_path,
                config=model_config,
                load_in_8bit=True,
            )
        else:
            if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
                auto_model_class = AutoModelForVision2Seq
            else:
                auto_model_class = AutoModelForCausalLM

            model = auto_model_class.from_pretrained(
                cfg.model.model_path,
                torch_dtype=torch.float32 if self.use_fp16 else self.torch_dtype,
                config=model_config,
                trust_remote_code=True,
            )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if cfg.fsdp_config.use_liger_kernel:
            self._optimize_with_liger_kernel(model)

        return model

    def _optimize_with_liger_kernel(self, model: torch.nn.Module) -> None:
        """
        Replace model modules with liger-kernel optimized modules.

        Params:
            model: the model to be optimized.
        """
        if self._cfg.model.get("gptq_model", False) or self._cfg.model.get(
            "load_in_8bit", False
        ):
            self._logger.info(
                "[FSDP] Skip using liger-kernel optimized modules for GPTQ/8bit models."
            )
            return
        try:
            from liger_kernel.transformers import (
                apply_liger_kernel_to_qwen2,
                apply_liger_kernel_to_qwen2_5_vl,
            )

            MODEL_ARCH_APPLY_FUNC = {
                "qwen2.5": (
                    apply_liger_kernel_to_qwen2,
                    {
                        "rope": True,
                        "rms_norm": True,
                        "swiglu": True,
                        "fused_linear_cross_entropy": True,
                    },
                ),
                "qwen2.5-vl": (
                    apply_liger_kernel_to_qwen2_5_vl,
                    {
                        "rope": True,
                        "rms_norm": True,
                        "swiglu": True,
                        "fused_linear_cross_entropy": True,
                    },
                ),
            }
            model_arch = self._cfg.model.get("model_arch", "").lower()
            if model_arch in MODEL_ARCH_APPLY_FUNC:
                apply_func, apply_kwargs = MODEL_ARCH_APPLY_FUNC[model_arch]
                apply_func(
                    model=model,
                    **apply_kwargs,
                )
                self._logger.info(
                    f"[FSDP] Applied liger-kernel optimizations for model_arch: {model_arch}, used kwargs: {apply_kwargs}"
                )
            else:
                self._logger.info(
                    f"[FSDP] No liger-kernel optimizations applied for model_arch: {model_arch}"
                )
                return
        except Exception as e:
            self._logger.warning(f"[FSDP] Liger kernels not applied: {e}")

    def setup_model_and_optimizer(self) -> None:
        """Setup model, lr_scheduler, optimizer and grad_scaler."""
        module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self._cfg.model.get("gradient_checkpointing", False):
            self._logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
        else:
            self._logger.info("[FSDP] Gradient checkpointing is disabled")

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        self.optimizer = self._strategy.build_optimizer(model=self.model)
        self.lr_scheduler = self._strategy.build_lr_scheduler(optimizer=self.optimizer)
        self.grad_scaler = self._strategy.build_grad_scaler()

    def get_rng_state(self) -> Dict:
        """
        Get rng state.

        Returns:
            rng_state: the current rng state.
        """
        return self._strategy.save_rng_state()

    def load_rng_state(self, rng_state: Dict) -> None:
        """
        Load rng state.

        Params:
            rng_state: the rng state to load.
        """
        self._strategy.load_rng_state(rng_state)

    def get_model_state_dict(self) -> Dict:
        """
        Get full model state dict.
        """
        state_dict = self._strategy.get_model_state_dict(self.model)
        return state_dict

    def load_checkpoint(self, load_path: str) -> None:
        """
        Load checkpoint from local path.

        Params:
            load_path: the directory to load checkpoint.
        """
        self._strategy.load_checkpoint(
            self.model, self.optimizer, self.lr_scheduler, load_path
        )

    def save_checkpoint(self, save_path: str) -> None:
        """
        Save checkpoint to local path.
        Every rank will save its own model and optim shard.

        Params:
            save_path: the directory to save checkpoint.
        """
        self._strategy.save_checkpoint(
            self.model, self.optimizer, self.lr_scheduler, save_path
        )

    def offload_param_and_grad(self, offload_grad: bool = False) -> None:
        """
        Offload FSDP parameters and gradients(options) to CPU.

        Params:
            offload_grad: whether to offload gradients.
        """
        self._strategy.offload_param_and_grad(self.model, offload_grad)

    def load_param_and_grad(self, device_id: int, load_grad: bool = False) -> None:
        """
        Load FSDP parameters and gradients(options) to the specified device.

        Params:
            device_id: the target device id to load parameters and gradients.
            load_grad: whether to load gradients.
        """
        self._strategy.onload_param_and_grad(self.model, device_id, load_grad)

    def offload_optimizer(self) -> None:
        """
        Offload optimizer states to CPU.
        """
        self._strategy.offload_optimizer(self.optimizer)

    def load_optimizer(self, device_id: int) -> None:
        """
        Load optimizer states to the specified device.

        Params:
            device_id: the target device id to load optimizer states.
        """
        self._strategy.onload_optimizer(self.optimizer, device_id)

    def optimizer_step(self) -> tuple[float, float]:
        """
        Perform optimizer step using its optimizer, lr_scheduler and grad_scaler.

        Returns:
            A tuple of (grad_norm, lr).
        """
        return self._strategy.optimizer_step(
            model=self.model,
            optimizer=self.optimizer,
            grad_scaler=self.grad_scaler,
            lr_scheduler=self.lr_scheduler,
            dp_group=self._dp_group,
        )

    def before_micro_batch(
        self, model: Union[FSDP, FSDPModule], is_last_micro_batch: bool
    ) -> ContextManager:
        """
            Setup context manager before processing a micro-batch.
            This is used to control gradient synchronization behavior.
            Depending on the specific FSDP strategy being used, if using
            FSDP, it will return model.no_sync() for non-last micro-batches to
            avoid gradient synchronization, and nullcontext() for the last
            micro-batch to ensure gradients are synchronized and updated.
            If using FSDP2, it will set requires_gradient_sync flag
            on the model accordingly.

        Args:
            model: The FSDP or FSDPModule model.
            is_last_micro_batch: A boolean indicating if this is the last micro-batch.

        Returns:
            A context manager for the micro-batch processing.
        """
        return self._strategy.before_micro_batch(
            model=model, is_last_micro_batch=is_last_micro_batch
        )
