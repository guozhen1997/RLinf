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

"""SFT worker that runs truncated BPTT through TTT fast weights."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.gr00t.gr00t_n1d7.sequence_utils import (
    segment_bounds,
    slice_batch_major,
)
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


class FSDPRoboTTTSftWorker(FSDPVlaSftWorker):
    """Sequence SFT with truncated backpropagation through TTT state.

    The base SFT worker issues a single ``backward`` per micro-batch. For a
    long trajectory that would keep every TTT inner-loop iterate on the graph.
    This worker splits the ``[B * T]`` batch into segments, backpropagates
    inside each segment, then detaches the fast weights before the next one so
    peak memory tracks segment length rather than ``T``.

    Intermediate segment backwards are wrapped in FSDP ``no_sync`` (or FSDP2
    ``set_requires_gradient_sync(False)``). Only the last segment uses the
    outer ``backward_ctx``, matching PyTorch's "sync on the last backward"
    contract. Segment losses are length-weighted so the total gradient matches
    the paper's :math:`(1/T)\\sum_t \\ell_t`.
    """

    def _tbptt_segment_length(self, num_timesteps: int) -> int:
        configured = int(self.cfg.actor.model.get("tbptt_segment_length", 0) or 0)
        if configured <= 0:
            return num_timesteps
        return min(configured, num_timesteps)

    @contextmanager
    def _tbptt_no_sync(self) -> Iterator[None]:
        """Disable gradient reduction for an intermediate TBPTT backward."""
        model = self.model
        if hasattr(model, "no_sync"):
            with model.no_sync():
                yield
            return
        if hasattr(model, "set_requires_gradient_sync"):
            model.set_requires_gradient_sync(False)
            try:
                yield
            finally:
                # Last-segment backward is issued by the base worker; restore
                # sync so that path can all-reduce when it is the last micro-batch.
                model.set_requires_gradient_sync(True)
            return
        with nullcontext():
            yield

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        num_timesteps = int(batch.get("num_timesteps", 1) or 1)
        segment_length = self._tbptt_segment_length(num_timesteps)
        bounds = segment_bounds(num_timesteps, segment_length)
        if len(bounds) == 1:
            return super().get_train_model_output(batch)

        ttt_context = None
        last_loss = None
        metrics: dict[str, Any] = {}
        for index, (start, end) in enumerate(bounds):
            window = end - start
            segment = slice_batch_major(batch, start, end, num_timesteps)
            if ttt_context is not None:
                ttt_context = ttt_context.detached()
                ttt_context.num_timesteps = window
            with self.amp_context:
                output = self.model(
                    forward_type=ForwardType.SFT,
                    data=segment,
                    ttt_context=ttt_context,
                )
            if isinstance(output, torch.Tensor):
                loss = output
            else:
                loss = output["loss"]
                ttt_context = output.get("ttt_context", ttt_context)
                for key, value in output.items():
                    if key in {"loss", "ttt_context", "action_loss", "action_mask"}:
                        continue
                    if torch.is_tensor(value) and value.numel() == 1:
                        metrics[key] = value.detach().item()
                    elif isinstance(value, (float, int)):
                        metrics[key] = value

            # Segment ``loss`` is already (1/window) sum_{t in seg} ell_t.
            # Weight by window/T so the sum of segment contributions is
            # (1/T) sum_t ell_t. The extra 1/grad_acc matches the base worker.
            scaled = loss * (window / num_timesteps) / self.gradient_accumulation
            is_last_segment = index == len(bounds) - 1
            if is_last_segment:
                last_loss = scaled * self.gradient_accumulation
            else:
                with self._tbptt_no_sync():
                    self.grad_scaler.scale(scaled).backward()

        assert last_loss is not None
        metrics["loss"] = last_loss.detach().item()
        metrics["tbptt_segments"] = len(bounds)
        return last_loss, metrics
