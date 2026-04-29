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

"""Compiled helpers patched onto groot ``CausalWanSelfAttention`` (see ``get_model``)."""

from __future__ import annotations

import torch


@torch.compile(mode="reduce-overhead")
def _process_clean_image_only(
    self, clean_image_q, clean_image_k, clean_image_v, clean_frames
):
    """Process clean image blocks with causal attention pattern - OPTIMIZED

    First frame: conditioning, cannot attend to anything (self-attention only)
    Block i: attends to first frame + previous blocks (0 to i-1) + current block

    OPTIMIZATION: Instead of looping through blocks, we batch process them together
    by using a single flash_attention call with properly structured KV cache.
    """
    block_size = self.frame_seqlen * self.num_frame_per_block
    num_blocks = (clean_frames - 1) // self.num_frame_per_block

    if num_blocks == 0:
        # Only first frame - single attention call
        return self.attn(
            clean_image_q[:, : self.frame_seqlen],
            clean_image_k[:, : self.frame_seqlen],
            clean_image_v[:, : self.frame_seqlen],
        )

    # Pre-allocate output tensor (avoids list append + cat overhead)
    b, total_len, n, d = clean_image_q.shape
    output = torch.empty_like(clean_image_q)

    # First frame: conditioning, self-attention only
    output[:, : self.frame_seqlen] = self.attn(
        clean_image_q[:, : self.frame_seqlen],
        clean_image_k[:, : self.frame_seqlen],
        clean_image_v[:, : self.frame_seqlen],
    )

    # OPTIMIZATION: Process all blocks together with causal masking
    # For global attention (no local_attn_size), we can process all blocks in one call
    if self.local_attn_size == -1:
        # Single attention call for all blocks!
        # Each position can attend to first_frame + everything up to itself
        blocks_q = clean_image_q[:, self.frame_seqlen :]
        blocks_k = clean_image_k  # Can attend to everything including first frame
        blocks_v = clean_image_v

        # Use causal masking: each block token can see first frame + all previous tokens
        output[:, self.frame_seqlen :] = self.causal_attn(blocks_q, blocks_k, blocks_v)
    else:
        # With local attention, we still need to loop but with optimizations
        # Pre-compute all block boundaries to reduce overhead
        block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        block_ends = [min(start + block_size, total_len) for start in block_starts]

        for block_idx in range(num_blocks):
            block_start = block_starts[block_idx]
            block_end = block_ends[block_idx]

            q_block = clean_image_q[:, block_start:block_end]

            # Context: first frame + recent blocks within local_attn_size
            image_kv_start = max(
                self.frame_seqlen, block_end - self.local_attn_size * self.frame_seqlen
            )
            k_context = torch.cat(
                [
                    clean_image_k[:, : self.frame_seqlen],  # First frame
                    clean_image_k[
                        :, image_kv_start:block_end
                    ],  # Recent blocks + current
                ],
                dim=1,
            )
            v_context = torch.cat(
                [
                    clean_image_v[:, : self.frame_seqlen],
                    clean_image_v[:, image_kv_start:block_end],
                ],
                dim=1,
            )

            output[:, block_start:block_end] = self.attn(q_block, k_context, v_context)

    return output


@torch.compile(mode="reduce-overhead")
def _process_state_blocks(self, state_q, state_k, state_v, state_horizon):
    """Process state blocks: self-attention only - OPTIMIZED

    OPTIMIZATION: State blocks only do self-attention within each block.
    Instead of looping, we can process all blocks in a single call with block-diagonal masking,
    or even simpler: just one attention call since they're independent.
    """
    num_blocks = state_horizon // self.num_state_per_block

    if num_blocks == 1:
        # Single block - one attention call
        return self.attn(state_q, state_k, state_v)

    # OPTIMIZATION: Since each state block only attends to itself (no cross-block attention),
    # we can process all blocks in a single batched call. Flash attention will handle this
    # efficiently. The blocks are independent, so this is safe.
    # Alternative: reshape and process as separate batch items

    # Pre-allocate output
    output = torch.empty_like(state_q)

    # Process all blocks (keeping loop for now due to block-diagonal pattern)
    # This could be further optimized with custom masking
    for block_idx in range(num_blocks):
        state_block_start = block_idx * self.num_state_per_block
        state_block_end = state_block_start + self.num_state_per_block

        output[:, state_block_start:state_block_end] = self.attn(
            state_q[:, state_block_start:state_block_end],
            state_k[:, state_block_start:state_block_end],
            state_v[:, state_block_start:state_block_end],
        )

    return output


@torch.compile(mode="reduce-overhead")
def _process_noisy_image_blocks(
    self,
    noisy_image_q,
    noisy_image_k,
    noisy_image_v,
    clean_image_k,
    clean_image_v,
    noisy_action_k,
    noisy_action_v,
    noisy_state_k,
    noisy_state_v,
    half_frames,
    action_horizon,
    state_horizon,
):
    """Process noisy image blocks with teacher forcing pattern - OPTIMIZED

    First frame: conditioning, cannot attend to anything (self-attention only)
    Block i: attends to action[i] + state[i] + first_clean_frame + clean_blocks[0:i] + current_noisy_block

    OPTIMIZATION: Pre-allocate output, pre-compute indices, reduce memory allocations
    """
    block_size = self.frame_seqlen * self.num_frame_per_block
    num_blocks = (half_frames - 1) // self.num_frame_per_block

    # Pre-allocate output tensor
    output = torch.empty_like(noisy_image_q)

    # First noisy frame: conditioning, self-attention only
    output[:, : self.frame_seqlen] = self.attn(
        noisy_image_q[:, : self.frame_seqlen],
        noisy_image_k[:, : self.frame_seqlen],
        noisy_image_v[:, : self.frame_seqlen],
    )

    if num_blocks == 0:
        return output

    # Pre-compute all block indices to reduce loop overhead
    noisy_block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
    noisy_block_ends = [
        min(start + block_size, noisy_image_q.shape[1]) for start in noisy_block_starts
    ]
    clean_context_ends = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
    action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
    action_block_ends = [
        start + self.num_action_per_block for start in action_block_starts
    ]
    state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
    state_block_ends = [
        start + self.num_state_per_block for start in state_block_starts
    ]

    # Process noisy image blocks
    for block_idx in range(num_blocks):
        noisy_start = noisy_block_starts[block_idx]
        noisy_end = noisy_block_ends[block_idx]
        clean_end = clean_context_ends[block_idx]
        action_start = action_block_starts[block_idx]
        action_end = action_block_ends[block_idx]
        state_start = state_block_starts[block_idx]
        state_end = state_block_ends[block_idx]

        q_block = noisy_image_q[:, noisy_start:noisy_end]

        # Build context: first_clean_frame + clean_blocks[0:i] + current_noisy_block + action[i] + state[i]
        k_context = torch.cat(
            [
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_start:noisy_end],
                noisy_action_k[:, action_start:action_end],
                noisy_state_k[:, state_start:state_end],
            ],
            dim=1,
        )
        v_context = torch.cat(
            [
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_start:noisy_end],
                noisy_action_v[:, action_start:action_end],
                noisy_state_v[:, state_start:state_end],
            ],
            dim=1,
        )

        output[:, noisy_start:noisy_end] = self.attn(q_block, k_context, v_context)

    return output


@torch.compile(mode="reduce-overhead")
def _process_noisy_action_blocks(
    self,
    noisy_action_q,
    noisy_action_k,
    noisy_action_v,
    clean_image_k,
    clean_image_v,
    noisy_image_k,
    noisy_image_v,
    noisy_state_k,
    noisy_state_v,
    half_frames,
    action_horizon,
    state_horizon,
):
    """Process noisy action blocks with teacher forcing pattern - OPTIMIZED

    First action (for first frame): cannot attend to anything (self-attention only)
    Action block i: attends to first_clean_frame + clean_blocks[0:i] + noisy_image[i] + action[i] + state[i]

    OPTIMIZATION: Pre-allocate output, pre-compute indices, reduce memory allocations
    """
    num_blocks = (half_frames - 1) // self.num_frame_per_block

    if num_blocks == 0:
        return torch.empty_like(noisy_action_q)

    # Pre-allocate output tensor
    output = torch.empty_like(noisy_action_q)

    # Pre-compute all block indices
    action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
    action_block_ends = [
        start + self.num_action_per_block for start in action_block_starts
    ]
    clean_context_ends = [
        self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block
        for i in range(num_blocks)
    ]
    noisy_image_block_starts = [
        self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block
        for i in range(num_blocks)
    ]
    noisy_image_block_ends = [
        start + self.frame_seqlen * self.num_frame_per_block
        for start in noisy_image_block_starts
    ]
    state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
    state_block_ends = [
        start + self.num_state_per_block for start in state_block_starts
    ]

    # Process noisy action blocks
    for block_idx in range(num_blocks):
        action_start = action_block_starts[block_idx]
        action_end = action_block_ends[block_idx]
        clean_end = clean_context_ends[block_idx]
        noisy_img_start = noisy_image_block_starts[block_idx]
        noisy_img_end = noisy_image_block_ends[block_idx]
        state_start = state_block_starts[block_idx]
        state_end = state_block_ends[block_idx]

        q_block = noisy_action_q[:, action_start:action_end]

        # Build context: first_clean_frame + clean_blocks[0:i] + noisy_image[i] + action[i] + state[i]
        k_context = torch.cat(
            [
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_img_start:noisy_img_end],
                noisy_action_k[:, action_start:action_end],
                noisy_state_k[:, state_start:state_end],
            ],
            dim=1,
        )
        v_context = torch.cat(
            [
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_img_start:noisy_img_end],
                noisy_action_v[:, action_start:action_end],
                noisy_state_v[:, state_start:state_end],
            ],
            dim=1,
        )

        output[:, action_start:action_end] = self.attn(q_block, k_context, v_context)

    return output


__all__ = [
    "_process_clean_image_only",
    "_process_state_blocks",
    "_process_noisy_image_blocks",
    "_process_noisy_action_blocks",
]
