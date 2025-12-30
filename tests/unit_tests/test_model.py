#!/usr/bin/env python3
"""
Evaluate an OpenVLA-OFT checkpoint in the RoboTwin environment and dump simple stats.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
from rlinf.models import get_model, get_vla_model_config_and_processor
# from tests.unit_tests.test_robotwin_env import create_test_config
def create_test_config():
    """Create test configuration"""
    config = OmegaConf.create(
        {
            "max_episode_steps": 200,
            "seed": 0,
            "assets_path": "/mnt/public/peihong/codes/RLinf_RoboTwin_gz",
            "seeds_path": "/mnt/public/peihong/codes/RLinf/examples/embodiment/seeds/robotwin2_eval_seeds.json",
            "auto_reset": True,
            "use_rel_reward": True,
            "use_custom_reward": False,
            "ignore_terminations": False,
            "num_images": 1,
            "num_group": 1,
            "group_size": 1,
            "use_fixed_reset_state_ids": False,
            "num_envs": 1,
            "video_cfg": {
                "save_video": True,
                "info_on_video": True,
                "video_base_dir": "./videos",
            },
            "task_config": {
                "task_name": "place_empty_cup",
                "step_lim": 200,
                "render_freq": 0,
                "episode_num": 100,
                "use_seed": False,
                "save_freq": 15,
                "embodiment": ["piper", "piper", 0.6],
                "save_path": "./data",
                "language_num": 100,
                "domain_randomization": {
                    "random_background": True,
                    "cluttered_table": True,
                    "clean_background_rate": 0.02,
                    "random_head_camera_dis": 0,
                    "random_table_height": 0.03,
                    "random_light": True,
                    "crazy_random_light_rate": 0.02,
                },
                "camera": {
                    "head_camera_type": "D435",
                    "wrist_camera_type": "D435",
                    "collect_head_camera": True,
                    "collect_wrist_camera": False,
                },
                "data_type": {
                    "rgb": True,
                    "third_view": False,
                    "depth": False,
                    "pointcloud": False,
                    "observer": False,
                    "endpose": False,
                    "qpos": True,
                    "mesh_segmentation": False,
                    "actor_segmentation": False,
                },
                "pcd_down_sample_num": 1024,
                "pcd_crop": True,
                "clear_cache_freq": 5,
                "collect_data": True,
                "eval_video_log": True,
            },
        }
    )
    return config

DEFAULT_MODEL_DIR = "/mnt/public/peihong/models/robotwin2_place_empty_cup_seed1k_sft_aloha_25chunks_10k"
DEFAULT_ASSETS_PATH = "/mnt/public/peihong/codes/RLinf_RoboTwin_gz"
DEFAULT_SEEDS_PATH = "/mnt/public/peihong/codes/RLinf/examples/embodiment/seeds/robotwin2_eval_seeds.json"


def build_env_cfg(args):
    """Reuse the unit-test config and adjust a few knobs for evaluation."""
    cfg = create_test_config()
    cfg.seed = args.seed
    cfg.auto_reset = args.auto_reset
    cfg.ignore_terminations = False
    cfg.use_custom_reward = False
    cfg.num_envs = args.num_envs
    cfg.max_episode_steps = args.max_episode_steps
    cfg.assets_path = args.assets_path
    cfg.seeds_path = args.seeds_path

    cfg.task_config.task_name = args.task_name
    cfg.task_config.step_lim = args.max_episode_steps
    cfg.task_config.episode_num = args.episodes

    cfg.video_cfg.save_video = args.save_video
    cfg.video_cfg.video_base_dir = str(Path(args.output_dir) / "videos")
    return cfg


def build_model(args):
    model_cfg = OmegaConf.create(
        {
            "model_name": "openvla_oft",
            "model_dir": args.model_dir,
            "action_dim": args.action_dim,
            "num_action_chunks": args.num_action_chunks,
            "add_value_head": False,
            "use_proprio": True,
            "proprio_dim": args.proprio_dim,
            "use_film": False,
            "center_crop": True,
            "precision": args.precision,
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "num_images_in_input": args.num_images_in_input,
            "unnorm_key": args.unnorm_key,
            "is_lora": False,
            "lora_rank": 32,
            # "lora_path": null,
            # "ckpt_path": null,
        }
    )

    model = get_model(model_cfg)

    actor_cfg = OmegaConf.create(
        {
            "model": model_cfg,
            "tokenizer": {"tokenizer_model": args.model_dir},
        }
    )
    setup_cfg = OmegaConf.create(
        {
            "actor": {"model": model_cfg},
            "runner": {"max_prompt_length": args.max_prompt_length},
        }
    )
    model_config, processor = get_vla_model_config_and_processor(actor_cfg)
    model.setup_config_and_processor(model_config, setup_cfg, processor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def run_episode(env: RoboTwinEnv, model, args):
    obs, _ = env.reset()
    episode_reward = 0.0
    success = 0.0
    length = 0.0

    for _ in range(args.max_chunk_calls):
        chunk_actions, _ = model.predict_action_batch(
            env_obs=obs,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        obs, chunk_rewards, chunk_terms, chunk_truncs, infos = env.chunk_step(
            chunk_actions
        )

        num_envs = chunk_rewards.shape[0]
        episode_reward += float(chunk_rewards.sum().item()) / max(1, num_envs)

        episode_info = infos.get("episode", {})
        if "episode_len" in episode_info:
            length = float(episode_info["episode_len"].float().mean().item())
        if "success_once" in episode_info:
            success = float(episode_info["success_once"].float().mean().item())

        done_mask = (chunk_terms[:, -1] > 0.5) | (chunk_truncs[:, -1] > 0.5)
        if done_mask.any():
            break

    return {"return": episode_reward, "length": length, "success": success}


def evaluate(env: RoboTwinEnv, model, args):
    results = []
    for ep in range(args.episodes):
        episode_stats = run_episode(env, model, args)
        episode_stats["episode"] = ep
        results.append(episode_stats)

    env.close(clear_cache=True)

    success_rate = float(np.mean([item["success"] for item in results])) if results else 0.0
    avg_return = float(np.mean([item["return"] for item in results])) if results else 0.0
    avg_length = float(np.mean([item["length"] for item in results])) if results else 0.0

    summary = {
        "episodes": len(results),
        "success_rate": success_rate,
        "avg_return": avg_return,
        "avg_length": avg_length,
    }
    return summary, results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenVLA-OFT checkpoint on RoboTwin."
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to the OpenVLA-OFT checkpoint.")
    parser.add_argument("--unnorm-key", default="place_empty_cup_1k", help="Normalization key matching the SFT dataset.")
    parser.add_argument("--task-name", default="place_empty_cup", help="RoboTwin task to evaluate.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes.")
    parser.add_argument("--max-chunk-calls", type=int, default=6, help="How many chunk_step calls to run per episode.")
    parser.add_argument("--num-action-chunks", type=int, default=25, help="Chunk horizon of the policy.")
    parser.add_argument("--action-dim", type=int, default=14, help="Action dimension expected by RoboTwin.")
    parser.add_argument("--proprio-dim", type=int, default=14, help="Proprioception dimension fed to the model.")
    parser.add_argument("--num-images-in-input", type=int, default=1, help="How many camera views are passed into the model.")
    parser.add_argument("--num-envs", type=int, default=1, help="Vectorized env count.")
    parser.add_argument("--max-episode-steps", type=int, default=200, help="Maximum env steps before truncation.")
    parser.add_argument("--auto-reset", action="store_true", help="Let the env auto-reset when done.")
    parser.add_argument("--save-video", action="store_true", help="Save rollout videos to the output directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--temperature", type=float, default=1.6, help="Sampling temperature when do_sample is enabled.")
    parser.add_argument("--top-k", dest="top_k", type=int, default=-1, help="Top-k for sampling; -1 keeps all tokens.")
    parser.add_argument("--top-p", dest="top_p", type=float, default=1.0, help="Top-p for sampling.")
    parser.add_argument("--do-sample", default=True, help="Use stochastic sampling instead of greedy decoding.")
    parser.add_argument("--precision", default="fp32", help="Model precision passed to get_model.")
    parser.add_argument("--max-prompt-length", type=int, default=512, help="Prompt length cap for the text encoder.")
    parser.add_argument("--assets-path", default=DEFAULT_ASSETS_PATH, help="RoboTwin assets path.")
    parser.add_argument("--seeds-path", default=DEFAULT_SEEDS_PATH, help="Seed list for RoboTwin resets.")
    parser.add_argument("--output-dir", default="chunk_step_logs/openvlaoft_eval", help="Where to dump the JSON stats.")
    return parser.parse_args()


def main():
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_cfg = build_env_cfg(args)
    model = build_model(args)
    env = RoboTwinEnv(env_cfg, seed_offset=0, total_num_processes=1, record_metrics=True)

    summary, episodes = evaluate(env, model, args)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = Path(args.output_dir) / f"openvlaoft_robotwin_eval_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "episodes": episodes,
                "config": {
                    "model_dir": args.model_dir,
                    "task_name": args.task_name,
                    "unnorm_key": args.unnorm_key,
                    "do_sample": args.do_sample,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                },
            },
            f,
            indent=2,
        )

    print(f"[robotwin-openvlaoft] Wrote eval stats to {output_path}")
    print(f"[robotwin-openvlaoft] Summary: {summary}")


def test_model():
    data = torch.load("/mnt/public/peihong/codes/RLinf_support/tests/unit_tests/input_debug_0.pt", weights_only=False)
    tt = data["current_inputs"][0]
    env_obs = {}
    env_obs["images"] = np.expand_dims(tt["full_image"], axis=0)
    print(env_obs["images"].shape)
    env_obs["states"] = np.expand_dims(tt["state"], axis=0)
    
    print(f"env_obs['states'].shape: {env_obs['states'].shape}")
    print(f"env_obs['states']: {env_obs['states']}")
    
    env_obs["task_descriptions"] = data["current_task_descriptions"]
    
    print(env_obs.keys())
    print(data["vla_input"].keys())
    args = parse_args()
    model = build_model(args)
    _, res = model.predict_action_batch(
        env_obs = env_obs,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    
    torch.save({
        "res": res,
    }, "/mnt/public/peihong/codes/RLinf_support/tests/unit_tests/predict_action_new.pt")
    
def test_forward():
    # data = torch.load("/mnt/public/peihong/codes/RLinf_support/tests/unit_tests/predict_action_new.pt", weights_only=False)["res"]
    # data = data["result"]["forward_inputs"]
    data = torch.load("/mnt/public/peihong/codes/RLinf_support/tests/unit_tests/new_debug.pt", weights_only=False)
    
    data["temperature"] = 1.6
    # data["input_ids"] = data["input_ids_unpad"]
    # data["attention_mask"] = data["attention_mask_unpad"]
    
    data["action_tokens"] = data["responses"] - (32000 - 256)
    # 
    args = parse_args()
    model = build_model(args)
    model.eval()

    output = model(
        input_ids=data["input_ids_unpad"].long(), 
        pixel_values=data["pixel_values"],
        attention_mask=data["attention_mask_unpad"],
        #labels=None,
        proprio=data["proprio"],
        action_tokens=data["action_tokens"]
    )
        
    # output = model.t_forward(
    #         data=data,
    #         compute_logprobs=True,
    #         compute_entropy=False,
    #         compute_values=False,
    #         use_cache=False,
    #     )
    # with torch.inference_mode():
    #     with torch.autocast("cuda", dtype=torch.bfloat16):
    #         output = model(
    #             data=data,
    #             compute_logprobs=True,
    #             compute_entropy=False,
    #             compute_values=False,
    #             use_cache=False,
    #         )
    torch.save(output, "new_debug_output_2.pt")
    print("output keys:", output.keys())
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {v}")
    
    
if __name__ == "__main__":
    # main()
    # test_model()
    test_forward()
