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
from typing import Dict, Tuple

import torch
from omegaconf import DictConfig
from packaging import version
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import (
    clip_grad_by_total_norm_,
    create_device_mesh,
    fsdp_version,
    get_grad_norm,
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

    def __init__(self, cfg: DictConfig, world_size: int) -> None:
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

        assert torch.distributed.is_initialized(), (
            "torch distributed is not initialized in FSDPModelManager's constructor."
        )
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

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
                torch_dtype=self.torch_dtype,
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
        """Setup model and optimizer."""
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

    def optimizer_step(self) -> Tuple[int, float, float]:
        """
        Perform optimizer step with gradient clipping if configured.

        Returns:
            success: 1 if the step is successful, 0 if skipped due to non-finite grad norm.
            grad_norm: the gradient norm.
            lr: the current learning rate.
        """

        assert self._cfg.optim.clip_grad is not None

        if fsdp_version(self.model) == 1:
            grad_norm = self.model.clip_grad_norm_(max_norm=self._cfg.optim.clip_grad)
        else:
            grad_norm = get_grad_norm(
                self.model.parameters(),
                dp_group=self._dp_group,
                dtype=torch.float32,
            )
            if self._cfg.optim.clip_grad is not None:
                clip_grad_by_total_norm_(
                    self.model.parameters(),
                    max_grad_norm=self._cfg.optim.clip_grad,
                    total_norm=grad_norm,
                    dtype=torch.float32,
                )
            grad_norm = torch.tensor([grad_norm])

        # if grad_norm is not finite, skip the update
        success = 1
        if not torch.isfinite(grad_norm):
            self._logger.warning(
                f"[Rank {torch.distributed.get_rank()}] grad_norm is not finite: {grad_norm}"
            )
            self.optimizer.zero_grad()
            success = 0
        else:
            self.optimizer.step()

        # TODO: lr scheduler step after all rollout batches are processed
        lr = self.lr_scheduler.get_last_lr()

        self.lr_scheduler.step()

        return success, grad_norm.item(), lr

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
        self._strategy.load_checkpoint(self.model, self.optimizer, load_path)

    def save_checkpoint(self, save_path: str) -> None:
        """
        Save checkpoint to local path.
        Every rank will save its own model and optim shard.

        Params:
            save_path: the directory to save checkpoint.
        """
        self._strategy.save_checkpoint(self.model, self.optimizer, save_path, self.rank)

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
