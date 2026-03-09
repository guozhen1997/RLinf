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
from collections import defaultdict

import torch
from omegaconf.omegaconf import DictConfig

from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult, EnvOutput
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class AsyncEnvWorker(EnvWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._interact_task: asyncio.Task = None
        assert not self.enable_offload, "Offload not supported in AsyncEnvWorker"

    async def interact(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
        replay_channel: Channel | None = None,
    ):
        assert self._interact_task is None or self._interact_task.done(), (
            "Previous interact task is still running while a new interact call is made."
        )
        self._interact_task = asyncio.create_task(
            self._interact(input_channel, output_channel, metric_channel, replay_channel)
        )
        try:
            await self._interact_task
        except asyncio.CancelledError:
            pass

    async def _interact(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
        replay_channel: Channel | None,
    ):
        while True:
            self.rollout_results: list[EmbodiedRolloutResult] = [
                EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.train.max_episode_steps,
                )
                for _ in range(self.stage_num)
            ]
            env_metrics = defaultdict(list)
            for epoch in range(self.rollout_epoch):
                last_obs = [None for _ in range(self.stage_num)]
                env_output_list = self.bootstrap_step()
                for stage_id in range(self.stage_num):
                    env_output: EnvOutput = env_output_list[stage_id]
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        output_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                    )
                    if env_output.intervene_actions is not None:
                        self.rollout_results[stage_id].update_last_actions(
                            env_output.intervene_actions,
                            env_output.intervene_flags,
                        )

                for _ in range(self.n_train_chunk_steps):
                    for stage_id in range(self.stage_num):
                        await asyncio.sleep(0)
                        env_output = env_output_list[stage_id]
                        raw_chunk_actions = self.recv_chunk_actions(input_channel)
                        rollout_result = self.recv_rollout_results(input_channel, mode="train")
                        rewards = self.compute_bootstrap_rewards(
                            env_output, rollout_result.bootstrap_values
                        )
                        chunk_step_result = ChunkStepResult(
                            actions=rollout_result.actions,
                            prev_logprobs=rollout_result.prev_logprobs,
                            prev_values=rollout_result.prev_values,
                            forward_inputs=rollout_result.forward_inputs,
                            versions=rollout_result.versions,
                            dones=env_output.dones,
                            truncations=env_output.truncations,
                            terminations=env_output.terminations,
                            rewards=rewards,
                        )
                        self.rollout_results[stage_id].append_step_result(chunk_step_result)
                        if self.collect_transitions and last_obs[stage_id] is not None:
                            curr_obs = last_obs[stage_id]
                            next_obs = (
                                env_output.final_obs
                                if env_output.dones is not None
                                and env_output.dones.any()
                                and self.cfg.env.train.auto_reset
                                else env_output.obs
                            )
                            self.rollout_results[stage_id].append_transitions(
                                curr_obs, next_obs
                            )
                        last_obs[stage_id] = env_output.obs
                        env_output, env_info = self.env_interact_step(
                            raw_chunk_actions, stage_id
                        )
                        env_batch = env_output.to_dict()
                        self.send_env_batch(
                            output_channel,
                            {
                                "obs": env_batch["obs"],
                                "final_obs": env_batch["final_obs"],
                            },
                        )
                        env_output_list[stage_id] = env_output
                        self.record_env_metrics(env_metrics, env_info, epoch)

                self.store_last_obs_and_intervened_info(env_output_list)
                self.finish_rollout()

            for stage_id in range(self.stage_num):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], replay_channel
                )

            for key, value in env_metrics.items():
                env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            env_interact_time_metrics = self.pop_execution_times()
            env_interact_time_metrics = {
                f"time/env/{k}": v for k, v in env_interact_time_metrics.items()
            }
            metrics = {
                "rank": self._rank,
                "env": env_metrics,
                "time": env_interact_time_metrics,
            }
            metric_channel.put(metrics, async_op=True)

    async def stop(self):
        if self._interact_task is not None and not self._interact_task.done():
            self._interact_task.cancel()
