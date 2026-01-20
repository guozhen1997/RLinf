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

from tqdm import tqdm

from rlinf.data.io_struct import AsyncEmbodiedRolloutBuffer
from rlinf.scheduler import Channel
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf.data.embodied_io_struct import BatchEmbodiedRolloutResult, ChunkStepResult, EpisodeResult, EmbodiedRolloutEpisodeResultHandler
from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    put_tensor_device,
    split_dict_to_chunk,
    stack_list_of_dict_tensor,
)

class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    async def generate(
        self, input_channel: Channel, output_channel: Channel, replay_channel: Channel
    ):
        # Determine compute_mask based on cfg
        auto_reset = self.cfg.env.train.get("auto_reset", True)
        ignore_terminations = self.cfg.env.train.get("ignore_terminations", False)
        compute_mask = not auto_reset and not ignore_terminations
        reward_type = self.cfg.algorithm.get("reward_type", "chunk_level")
        
        self.rollout_result_handlers: list[EmbodiedRolloutEpisodeResultHandler] = [
            EmbodiedRolloutEpisodeResultHandler(
                actor_split_num=self.get_actor_split_num(), 
                channel=replay_channel,
                compute_mask=compute_mask,
                reward_type=reward_type
            ) for _ in range(self.num_pipeline_stages)
        ]

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        progress_bar = tqdm(
            total=None,
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        )

        while not self.should_stop:

            # episode_results[stage_id]
            batch_episode_results: list[BatchEmbodiedRolloutResult] = [
                BatchEmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.train.max_episode_steps,
                    collect_obs=self.cfg.rollout.get("collect_obs", True)
                ) for _ in range(self.num_pipeline_stages)
            ]

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if env_output["intervene_actions"] is not None:
                        batch_episode_results[stage_id].update_last_actions(
                            env_output["intervene_actions"], env_output["intervene_flags"]
                        )

                    if self.cfg.env.train.get("auto_reset", False):
                        # append completed episodes to buffer
                        completed_episodes = batch_episode_results[stage_id].get_completed_episodes()
                        self.rollout_result_handlers[stage_id].append_and_send_eposide_results(completed_episodes)

                    dones, rewards = self.get_dones_and_rewards(
                        env_output
                    )

                    actions, result = self.predict(env_output["obs"])

                    env_output["obs"].pop("task_descriptions", None)
                    if env_output["final_obs"] is not None:
                        env_output["final_obs"].pop("task_descriptions", None)
                    chunk_step_result = ChunkStepResult(
                        obs=env_output["obs"],
                        actions=result.get("action", None),
                        dones=dones,
                        rewards=rewards,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        forward_inputs=result["forward_inputs"],
                        final_obs=env_output["final_obs"],
                    )

                    batch_episode_results[stage_id].append_batch_result(chunk_step_result)

                    self.send_chunk_actions(output_channel, actions)

            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)

                if env_output["intervene_actions"] is not None:
                    batch_episode_results[stage_id].update_last_actions(
                        env_output["intervene_actions"], env_output["intervene_flags"]
                    )

                if self.cfg.env.train.get("auto_reset", False):
                    # append completed episodes to buffer
                    completed_episodes = batch_episode_results[stage_id].get_completed_episodes()
                    self.rollout_result_handlers[stage_id].append_and_send_eposide_results(completed_episodes)

                dones, rewards = self.get_dones_and_rewards(
                    env_output
                )

                with self.worker_timer():
                    _, result = self.predict(env_output["obs"])
                
                chunk_step_result = ChunkStepResult(
                    obs=env_output["obs"],
                    final_obs=env_output["final_obs"],
                    dones=dones,
                    rewards=rewards,
                    truncations=env_output["truncations"],
                    terminations=env_output["terminations"],
                    prev_logprobs=result["prev_logprobs"],
                    prev_values=result["prev_values"],
                    forward_inputs=result["forward_inputs"],
                )

                batch_episode_results[stage_id].append_batch_result(chunk_step_result)
                completed_episodes = batch_episode_results[stage_id].get_completed_episodes()
                self.rollout_result_handlers[stage_id].append_and_send_eposide_results(completed_episodes)

            progress_bar.update(1)

    async def stop(self):
        self.should_stop = True
        for buffer in self.buffer_list:
            await buffer.stop()
        await asyncio.gather(*self.buffer_tasks)
