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

from collections import defaultdict

import numpy as np
import torch

from rlinf.data.io_struct import EnvOutput
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class AsyncEnvWorker(EnvWorker):
    async def evaluate(self):
        for i in range(self.stage_num):
            self.eval_simulator_list[i].start_simulator()
            self.eval_simulator_list[i].is_start = True
            extracted_obs, _, _, _, infos = self.eval_simulator_list[i].step()
            env_output = EnvOutput(
                simulator_type=self.cfg.env.eval.simulator_type,
                obs=extracted_obs,
                final_obs=infos["final_observation"]
                if "final_observation" in infos
                else None,
            )
            await self.send_env_batch(env_output.to_dict(), mode="eval")

        eval_metrics = defaultdict(list)

        for eval_step in range(self.cfg.algorithm.n_eval_chunk_steps):
            for i in range(self.stage_num):
                raw_chunk_actions = await self.recv_chunk_actions(mode="eval")
                env_output, env_info = self.env_evaluate_step(raw_chunk_actions, i)

                for key, value in env_info.items():
                    eval_metrics[key].append(value)
                if eval_step == self.cfg.algorithm.n_eval_chunk_steps - 1:
                    continue
                await self.send_env_batch(env_output.to_dict(), mode="eval")

        self.finish_rollout(mode="eval")
        for i in range(self.stage_num):
            self.eval_simulator_list[i].stop_simulator()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    async def recv_chunk_actions(self, mode="train"):
        assert mode in ["train", "eval"]
        chunk_action = []
        for gather_id in range(self.gather_num):
            chunk_action.append(
                await self.channel.get(
                    key=f"{self._action_queue_name}_{mode}_{gather_id + self._rank * self.gather_num}",
                    async_op=True,
                ).async_wait()
            )
        chunk_action = np.concatenate(chunk_action, axis=0)
        return chunk_action

    async def send_env_batch(self, env_batch, mode="train"):
        assert mode in ["train", "eval"]
        # split env_batch into num_processes chunks, each chunk contains gather_num env_batch
        for gather_id in range(self.gather_num):
            env_batch_i = self.split_env_batch(env_batch, gather_id, mode)
            await self.channel.put(
                item=env_batch_i,
                key=f"{self._obs_queue_name}_{mode}_{gather_id + self._rank * self.gather_num}",
                async_op=True,
            ).async_wait()

    async def interact(self, env_metric_channel: Channel):
        for simulator in self.simulator_list:
            simulator.start_simulator()

        for epoch in range(self.cfg.algorithm.rollout_epoch):
            env_metrics = defaultdict(list)
            env_output_list = []
            if not self.cfg.env.train.auto_reset:
                for i in range(self.stage_num):
                    self.simulator_list[i].is_start = True
                    extracted_obs, rewards, terminations, truncations, infos = (
                        self.simulator_list[i].step()
                    )
                    dones = (
                        torch.logical_or(terminations, truncations)
                        .unsqueeze(1)
                        .repeat(1, self.cfg.actor.model.num_action_chunks)
                    )
                    terminations = terminations[:, None].repeat(
                        1, self.cfg.actor.model.num_action_chunks
                    )
                    truncations = truncations[:, None].repeat(
                        1, self.cfg.actor.model.num_action_chunks
                    )

                    env_output = EnvOutput(
                        simulator_type=self.cfg.env.train.simulator_type,
                        obs=extracted_obs,
                        rewards=rewards,
                        dones=dones,
                        terminations=terminations,
                        truncations=truncations,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                    )
                    env_output_list.append(env_output)
            else:
                self.num_done_envs = 0
                self.num_succ_envs = 0
                for i in range(self.stage_num):
                    env_output = EnvOutput(
                        simulator_type=self.cfg.env.train.simulator_type,
                        obs=self.last_obs_list[i],
                        rewards=None,
                        dones=self.last_dones_list[i],
                        terminations=self.last_terminations_list[i],
                        truncations=self.last_truncations_list[i],
                    )
                    env_output_list.append(env_output)

            for stage_id in range(self.stage_num):
                env_output: EnvOutput = env_output_list[stage_id]
                await self.send_env_batch(env_output.to_dict())

            for _ in range(self.cfg.algorithm.n_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = await self.recv_chunk_actions()
                    env_output, env_info = self.env_interact_step(
                        raw_chunk_actions, stage_id
                    )
                    await self.send_env_batch(env_output.to_dict())
                    env_output_list[stage_id] = env_output
                    for key, value in env_info.items():
                        if (
                            not self.cfg.env.train.auto_reset
                            and not self.cfg.env.train.ignore_terminations
                        ):
                            if key in env_metrics and len(env_metrics[key]) > epoch:
                                env_metrics[key][epoch] = value
                            else:
                                env_metrics[key].append(value)
                        else:
                            env_metrics[key].append(value)

            for key, value in env_metrics.items():
                env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()
            env_metric_channel.put(env_metrics)

            self.last_obs_list = [env_output.obs for env_output in env_output_list]
            self.last_dones_list = [env_output.dones for env_output in env_output_list]
            self.last_truncations_list = [
                env_output.truncations for env_output in env_output_list
            ]
            self.last_terminations_list = [
                env_output.terminations for env_output in env_output_list
            ]
            self.finish_rollout()

        for simulator in self.simulator_list:
            simulator.stop_simulator()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics
