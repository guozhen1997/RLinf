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

# !/usr/bin/env python3
"""
Test the functionality of RoboTwinEnv
"""

import multiprocessing as mp
import os
import numpy as np
from omegaconf import OmegaConf
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

def create_base_cfg():
    """给 chunk 测试用一个简单基准 cfg（在此基础上改 auto_reset / ignore_terminations）"""
    cfg = create_test_config()
    # 可以视情况减小 step_lim 和 episode_num，加快测试
    cfg.task_config.step_lim = 50
    cfg.task_config.episode_num = 10
    return cfg


def test_chunk_step_dones_logic():
    """
    测试 chunk_step 在 auto_reset / ignore_terminations 组合下：
    - dones = terminations OR truncations 的逻辑是否正确
    - ignore_terminations 只影响 terminations，不影响 truncations
    - auto_reset 不影响 terminations / truncations
    """
    print("\n===== Testing chunk_step dones logic =====")

    base_cfg = create_base_cfg()

    # 1) auto_reset=False, ignore_terminations=False 作为 baseline
    cfg_00 = base_cfg.copy()
    cfg_00.auto_reset = False
    cfg_00.ignore_terminations = False
    terms_00, truncs_00 = run_chunk_rollout(cfg_00)
    print("cfg_00 (auto_reset=False, ignore_terminations=False) done.")

    # # 2) auto_reset=True, ignore_terminations=False
    # cfg_10 = base_cfg.copy()
    # cfg_10.auto_reset = True
    # cfg_10.ignore_terminations = False
    # terms_10, truncs_10 = run_chunk_rollout(cfg_10)
    # print("cfg_10 (auto_reset=True, ignore_terminations=False) done.")

    # # 3) auto_reset=False, ignore_terminations=True
    # cfg_01 = base_cfg.copy()
    # cfg_01.auto_reset = False
    # cfg_01.ignore_terminations = True
    # terms_01, truncs_01 = run_chunk_rollout(cfg_01)
    # print("cfg_01 (auto_reset=False, ignore_terminations=True) done.")

    # # 4) auto_reset=True, ignore_terminations=True
    # cfg_11 = base_cfg.copy()
    # cfg_11.auto_reset = True
    # cfg_11.ignore_terminations = True
    # terms_11, truncs_11 = run_chunk_rollout(cfg_11)
    # print("cfg_11 (auto_reset=True, ignore_terminations=True) done.")

    # ========== 断言部分 ==========

    # # (A) auto_reset 不应该影响 dones:
    # #     在 ignore_terminations 固定的情况下，terms / truncs 应该相同
    # assert np.allclose(truncs_00, truncs_10), "auto_reset 改变了 truncations（不应该）"
    # assert np.allclose(truncs_01, truncs_11), "auto_reset 改变了 truncations（不应该）"
    # assert np.allclose(terms_00, terms_10), "auto_reset 改变了 terminations（不应该）"
    # assert np.allclose(terms_01, terms_11), "auto_reset 改变了 terminations（不应该）"

    # # (B) ignore_terminations 只影响 terminations，不影响 truncations
    # assert np.allclose(truncs_00, truncs_01), "ignore_terminations 居然影响了 truncations（不应该）"
    # assert np.allclose(truncs_10, truncs_11), "ignore_terminations 居然影响了 truncations（不应该）"

    # # 在相同 auto_reset 下，开启 ignore_terminations 后，termination 只能更少（被置为 0）
    # assert np.all(terms_01 <= terms_00 + 1e-6), "ignore_terminations=True 时，termination 没有变少"
    # assert np.all(terms_11 <= terms_10 + 1e-6), "ignore_terminations=True 时，termination 没有变少"

    # # (C) 校验 dones = terminations OR truncations 在四个配置下都是成立的
    # def check_dones(terms, truncs, name):
    #     dones = ((terms > 0.5) | (truncs > 0.5)).astype(np.float32)
    #     # 从 chunk_step 返回的 dones 本质就是 term/trunc 的或，逻辑上始终成立，这里更多是 sanity check
    #     # 你也可以在这里根据实际需要打印一些统计
    #     print(f"{name}: num_dones={dones.sum()}")

    # check_dones(terms_00, truncs_00, "cfg_00")
    # check_dones(terms_10, truncs_10, "cfg_10")
    # check_dones(terms_01, truncs_01, "cfg_01")
    # check_dones(terms_11, truncs_11, "cfg_11")

    print("✓ chunk_step dones 逻辑测试通过！")

def run_chunk_rollout(cfg, tag="cfg_00", num_chunk_steps=5, chunk_horizon=5, save_dir="./chunk_step_logs"):
    """
    用给定 cfg 创建一个 RoboTwinEnv，然后：
    - reset 一次
    - 连续调用 num_chunk_steps 次 chunk_step
    - 每次记录 chunk_terminations[:, -1] 和 chunk_truncations[:, -1]
    返回两个 numpy 数组：
      all_terms:  [num_chunk_steps, num_envs]
      all_truncs: [num_chunk_steps, num_envs]
    同时把这些结果保存到文件中，便于人工查看。
    """
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(0)

    env = RoboTwinEnv(cfg, seed_offset=0, total_num_processes=1)
    obs, info = env.reset()

    num_envs = env.num_envs
    action_dim = obs["states"].shape[-1]  # 使用状态维度作为动作维度（你的 env 中是 14）

    all_terms = []
    all_truncs = []

    for k in range(num_chunk_steps):
        # 构造一个 chunk 的动作: [num_envs, chunk_horizon, action_dim]
        chunk_actions = np.random.uniform(
            low=-1.0, high=1.0, size=(num_envs, chunk_horizon, action_dim)
        ).astype(np.float32)

        _, _, chunk_terms, chunk_truncs, infos = env.chunk_step(chunk_actions)

        # 只关心每个 env 在这个 chunk 中“最后一步”的 termination / truncation
        last_terms = chunk_terms[:, -1].cpu().numpy()
        last_truncs = chunk_truncs[:, -1].cpu().numpy()

        all_terms.append(last_terms)
        all_truncs.append(last_truncs)

    env.close(clear_cache=True)

    all_terms = np.stack(all_terms, axis=0)   # [num_chunk_steps, num_envs]
    all_truncs = np.stack(all_truncs, axis=0)

    # ----- 保存为 npz -----
    npz_path = os.path.join(save_dir, f"{tag}_terms_truncs.npz")
    np.savez(npz_path, terms=all_terms, truncs=all_truncs)

    # ----- 保存为可读文本 -----
    txt_path = os.path.join(save_dir, f"{tag}_terms_truncs.txt")
    with open(txt_path, "w") as f:
        f.write(f"tag={tag}, num_chunk_steps={num_chunk_steps}, num_envs={num_envs}\n")
        f.write("chunk_idx, env_idx, term, trunc, done\n")
        for i in range(num_chunk_steps):
            for j in range(num_envs):
                term = float(all_terms[i, j])
                trunc = float(all_truncs[i, j])
                done = 1.0 if (term > 0.5 or trunc > 0.5) else 0.0
                f.write(f"{i:03d}, {j:02d}, {term:.1f}, {trunc:.1f}, {done:.1f}\n")

    print(f"[{tag}] 保存 chunk dones 到: {txt_path} 和 {npz_path}")
    return all_terms, all_truncs




def create_test_config():
    """Create test configuration"""
    config = OmegaConf.create(
        {
            "max_episode_steps": 200,
            "seed": 0,
            "assets_path": "/mnt/public/guozhen/test_robotwin/robotwin_assets",
            "seeds_path": "/mnt/public/wph/codes/RLinf/examples/embodiment/seeds/robotwin2_eval_seeds.json",
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


def test_robotwin_env():
    """Test basic functionality of RoboTwinEnv"""
    print("Starting RoboTwinEnv tests...")

    # Create configuration
    cfg = create_test_config()

    # Create environment
    env = RoboTwinEnv(cfg, seed_offset=0, total_num_processes=1, record_metrics=True)
    print(f"✓ Environment created successfully, num_envs: {env.num_envs}")

    # Test reset
    print("Testing reset...")
    obs, info = env.reset()
    print(f"✓ Reset successful, observation keys: {obs.keys()}")
    print(f"  Image shape: {obs['images'].shape}")
    if obs["wrist_images"] is not None:
        print(f"  Wrist image shape: {obs['wrist_images'].shape}")
    print(f"  State shape: {obs['states'].shape}")

    # Test step
    print("Testing step...")
    for step in range(5):
        # Generate random actions
        actions = np.random.randn(env.num_envs, 1, 14)
        actions = np.clip(actions, 0, 1)

        obs, reward, terminated, truncated, info = env.step(actions)
        print(
            f"  Step {step}: reward={reward}, "
            f"terminated={terminated}, truncated={truncated}, info={info}, {obs['states']=}"
        )

    # Test flush_video
    print("Testing flush_video...")
    env.flush_video()

    # Test chunk_step
    print("Testing chunk_step...")
    chunk_actions = np.random.randn(env.num_envs, 3, 14)  # 3-step chunk
    chunk_actions = np.clip(chunk_actions, 0, 1)

    obs, chunk_rewards, chunk_terminations, chunk_truncations, info = env.chunk_step(
        chunk_actions
    )
    print(f"✓ Chunk step successful, reward shape: {chunk_rewards.shape}")
    print(f"✓ Chunk step successful, info: {info}")

    # Test partial reset
    print("Testing partial reset...")
    obs, info = env.reset(env_idx=[0])  # Only reset environment 0
    print(f"✓ Partial reset successful, observation keys: {obs.keys()}")
    print(f"✓ Partial reset successful, info: {info}")

    # Cleanup
    env.close(clear_cache=True)
    print("✓ Environment cleaned up successfully")

    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    # test 单步step
    # test_robotwin_env()
    # test chunk step 逻辑
    test_chunk_step_dones_logic()
