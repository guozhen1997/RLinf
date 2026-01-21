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


from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.scheduler import Channel
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    async def generate(
        self, input_channel: Channel, output_channel: Channel, replay_channel: Channel
    ):
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        while not self.should_stop:
            # rollout_results[stage_id]
            rollout_results: list[EmbodiedRolloutResult] = [
                EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.train.max_episode_steps
                )
                for _ in range(self.num_pipeline_stages)
            ]
            last_obs = [None for i in range(self.num_pipeline_stages)]
            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if env_output["intervene_actions"] is not None:
                        rollout_results[stage_id].update_last_actions(
                            env_output["intervene_actions"],
                            env_output["intervene_flags"],
                        )

                    dones, rewards = self.get_dones_and_rewards(env_output)

                    actions, result = self.predict(env_output["obs"])

                    env_output["obs"].pop("task_descriptions", None)
                    if env_output["final_obs"] is not None:
                        env_output["final_obs"].pop("task_descriptions", None)
                    chunk_step_result = ChunkStepResult(
                        actions=result.get("action", None),
                        dones=dones,
                        rewards=rewards,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        forward_inputs=result["forward_inputs"],
                    )

                    rollout_results[stage_id].append_step_result(chunk_step_result)
                    if self.collect_transitions and last_obs[stage_id] is not None:
                        curr_obs = last_obs[stage_id]
                        next_obs = (
                            env_output["final_obs"]
                            if dones.any() and self.cfg.env.train.auto_reset
                            else env_output["obs"]
                        )
                        rollout_results[stage_id].append_transitions(curr_obs, next_obs)

                    last_obs[stage_id] = env_output["obs"]

                    self.send_chunk_actions(output_channel, actions)

            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)

                if env_output["intervene_actions"] is not None:
                    rollout_results[stage_id].update_last_actions(
                        env_output["intervene_actions"], env_output["intervene_flags"]
                    )

                dones, rewards = self.get_dones_and_rewards(env_output)

                with self.worker_timer():
                    _, result = self.predict(env_output["obs"])

                chunk_step_result = ChunkStepResult(
                    dones=dones,
                    rewards=rewards,
                    truncations=env_output["truncations"],
                    terminations=env_output["terminations"],
                    prev_logprobs=None,
                    prev_values=result["prev_values"],
                    forward_inputs=None,
                )

                rollout_results[stage_id].append_step_result(chunk_step_result)
                if self.collect_transitions and last_obs[stage_id] is not None:
                    curr_obs = last_obs[stage_id]
                    next_obs = (
                        env_output["final_obs"]
                        if dones.any() and self.cfg.env.train.auto_reset
                        else env_output["obs"]
                    )
                    rollout_results[stage_id].append_transitions(curr_obs, next_obs)

            for stage_id in range(self.num_pipeline_stages):
                await self.send_rollout_trajectories(
                    rollout_results[stage_id], replay_channel
                )

    async def stop(self):
        self.should_stop = True
