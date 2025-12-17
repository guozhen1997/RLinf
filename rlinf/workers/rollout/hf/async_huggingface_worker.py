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

import asyncio
import gc
import torch
from tqdm import tqdm

from rlinf.data.io_struct import AsyncEmbodiedRolloutBuffer
from rlinf.scheduler import Channel
from rlinf.utils.metric_utils import compute_split_num
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    def get_split_num(self):
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    async def generate(self, data_channel: Channel):
        self.buffer_list: list[AsyncEmbodiedRolloutBuffer] = [
            AsyncEmbodiedRolloutBuffer() for _ in range(self.num_pipeline_stages)
        ]

        tasks = []
        for buffer in self.buffer_list:
            tasks.append(
                asyncio.create_task(buffer.run(data_channel, self.get_split_num()))
            )

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            extracted_obs = [None for i in range(self.num_pipeline_stages)]
            for chunk_step in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    await asyncio.sleep(0)
                    env_output = self.recv_env_output()

                    next_extracted_obs = self.hf_model.preprocess_env_obs(
                        env_output["obs"]
                    )
                    dones, rewards, real_next_extracted_obs = self.get_dones_and_rewards(
                        env_output, next_extracted_obs
                    )

                    actions, result = self.predict(next_extracted_obs)

                    await self.buffer_list[stage_id].add(
                        "truncations", env_output["truncations"].bool().cpu().contiguous()
                    )
                    await self.buffer_list[stage_id].add(
                        "terminations", env_output["terminations"].bool().cpu().contiguous()
                    )
                    await self.buffer_list[stage_id].add("dones", dones)
                    if rewards is not None:
                        await self.buffer_list[stage_id].add("rewards", rewards)
                    await self.buffer_list[stage_id].add_result(result)

                    if extracted_obs[stage_id] is not None and hasattr(
                        self.hf_model, "q_head"
                    ):
                        await self.buffer_list[stage_id].add_transition(
                            extracted_obs[stage_id], real_next_extracted_obs
                        )

                    extracted_obs[stage_id] = next_extracted_obs

                    self.send_chunk_actions(actions)

            for i in range(self.num_pipeline_stages):
                env_output = self.recv_env_output()
                next_extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                dones, rewards, real_next_extracted_obs = self.get_dones_and_rewards(
                    env_output, next_extracted_obs
                )
                await self.buffer_list[i].add(
                        "truncations", env_output["truncations"].bool().cpu().contiguous()
                    )
                await self.buffer_list[i].add(
                    "terminations", env_output["terminations"].bool().cpu().contiguous()
                )
                await self.buffer_list[i].add("dones", dones)
                if rewards is not None:
                    await self.buffer_list[i].add("rewards", rewards)

                with self.worker_timer():
                    actions, result = self.predict(next_extracted_obs)
                if "prev_values" in result:
                    await self.buffer_list[i].add(
                        "prev_values", result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    await self.buffer_list[i].add_transition(
                        extracted_obs[i], real_next_extracted_obs
                    )

        for task in tasks:
            await task

    async def sync_model_from_actor(self):
        param_state_dict = await self.recv(
            self.actor_group_name, src_rank=self._rank, async_op=True
        ).async_wait()
        self.hf_model.load_state_dict(param_state_dict)

        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()
