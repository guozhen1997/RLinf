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

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)

_REWARD_SERVER_COMPONENT_NAME = "reward_server"
_REMOVED_REWARD_MODEL_SGLANG_KEYS = (
    "sglang_server_args",
    "sglang_router_args",
    "sglang_engine_args",
)


def _launch_sglang_router_and_server(*args, **kwargs):
    from rlinf.workers.rollout.sglang_server import launch_sglang_router_and_server

    return launch_sglang_router_and_server(*args, **kwargs)


def _validate_reward_model_has_no_sglang_serving_fields(cfg) -> None:
    reward_model_cfg = cfg.get("reward", {}).get("model", {})
    removed_keys = [
        key for key in _REMOVED_REWARD_MODEL_SGLANG_KEYS if key in reward_model_cfg
    ]
    if removed_keys:
        raise ValueError(
            "SGLang reward serving config must not live under reward.model. "
            f"Move {removed_keys} to the standard top-level router_server_args "
            "block."
        )
    if str(reward_model_cfg.get("inference_backend", "")).lower() == "sglang":
        raise ValueError(
            "reward.model.inference_backend='sglang' is no longer supported. "
            "Use reward.worker_type='api' with an OpenAI-compatible reward.api."
        )


def should_launch_managed_sglang_reward_api(cfg) -> bool:
    _validate_reward_model_has_no_sglang_serving_fields(cfg)
    reward_cfg = cfg.get("reward", {})
    if not reward_cfg.get("use_reward_model", False):
        return False
    if str(reward_cfg.get("worker_type", "model")).lower() != "api":
        return False

    api_cfg = reward_cfg.get("api", {})
    api_base = str(
        api_cfg.get("api_base") or api_cfg.get("_runtime_api_base") or ""
    ).strip()
    if api_base:
        return False
    if "router_server_args" not in cfg:
        raise ValueError(
            "reward.worker_type='api' requires either reward.api.api_base or the "
            "standard top-level router_server_args block for Ray-managed SGLang."
        )
    return True


def _resolve_reward_api_base_url(server_group, router_group) -> str:
    if router_group is not None:
        return router_group.get_router_url().wait()[0].rstrip("/")
    if server_group is not None:
        server_urls = server_group.get_server_url().wait()
        if server_urls:
            return str(server_urls[0]).rstrip("/")
    raise RuntimeError(
        "Unable to resolve reward.api._runtime_api_base from managed SGLang reward API."
    )


def launch_managed_sglang_reward_api(cfg, cluster, component_placement):
    if not should_launch_managed_sglang_reward_api(cfg):
        return None

    server_group = None
    router_group = None
    try:
        server_group, router_group = _launch_sglang_router_and_server(
            config=cfg,
            cluster=cluster,
            rollout_hardware_ranks=None,
            router_server_args=cfg.router_server_args,
            placement_strategy=component_placement.get_strategy(
                _REWARD_SERVER_COMPONENT_NAME
            ),
        )
        api_base = _resolve_reward_api_base_url(server_group, router_group)
        with open_dict(cfg.reward):
            if "api" not in cfg.reward:
                cfg.reward.api = {}
        with open_dict(cfg.reward.api):
            cfg.reward.api._runtime_api_base = api_base
        return server_group, router_group
    except Exception:
        stop_managed_sglang_reward_api((server_group, router_group))
        raise


def stop_managed_sglang_reward_api(managed_reward_api) -> None:
    if managed_reward_api is None:
        return
    server_group, router_group = managed_reward_api
    try:
        if router_group is not None:
            router_group.shutdown().wait()
    finally:
        if server_group is not None:
            server_group.shutdown().wait()


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward import EmbodiedAPIRewardWorker, EmbodiedRewardWorker

    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    use_training_pipeline = bool(cfg.runner.get("use_training_pipeline", False))

    if cfg.algorithm.loss_type == "embodied_sac":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_sac."
            )
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_dagger":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_dagger."
            )
        from rlinf.workers.actor.fsdp_dagger_policy_worker import (
            EmbodiedDAGGERFSDPPolicy,
        )

        actor_worker_cls = EmbodiedDAGGERFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_nft":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_nft."
            )
        from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

        actor_worker_cls = EmbodiedNFTFSDPPolicy
    else:
        if use_training_pipeline:
            from rlinf.workers.actor.fsdp_actor_worker_pipeline import (
                PipelineEmbodiedFSDPActor,
            )

            actor_worker_cls = PipelineEmbodiedFSDPActor
        else:
            from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

            actor_worker_cls = EmbodiedFSDPActor

    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    managed_sglang_reward_api = None
    reward_group = None
    try:
        managed_sglang_reward_api = launch_managed_sglang_reward_api(
            cfg, cluster, component_placement
        )
        if cfg.get("reward", {}).get("use_reward_model", False) and not cfg.get(
            "reward", {}
        ).get("standalone_realworld", False):
            reward_placement = component_placement.get_strategy("reward")
            reward_worker_cls = (
                EmbodiedAPIRewardWorker
                if str(cfg.reward.get("worker_type", "model")).lower() == "api"
                else EmbodiedRewardWorker
            )
            reward_group = reward_worker_cls.create_group(cfg).launch(
                cluster,
                name=cfg.reward.group_name,
                placement_strategy=reward_placement,
            )

        runner = EmbodiedRunner(
            cfg=cfg,
            actor=actor_group,
            rollout=rollout_group,
            env=env_group,
            reward=reward_group,
        )

        runner.init_workers()
        runner.run()
    finally:
        if reward_group is not None:
            reward_group.stop().wait()
        stop_managed_sglang_reward_api(managed_sglang_reward_api)


if __name__ == "__main__":
    main()
