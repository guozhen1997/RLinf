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

def create_test_config(task_name):
    """Create test configuration"""
    config = OmegaConf.create(
        {
            "max_episode_steps": 200,
            "seed": 0,
            "assets_path": "/mnt/public/peihong/codes/RLinf_RoboTwin_gz",
            # "seeds_path": "/mnt/public/peihong/codes/SimpleVLA-RL/robotwin2_eval_test_seeds.json",
            "seeds_path": None,
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
                "task_name": task_name,
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


def test_robotwin_env(env_name):
    """Test basic functionality of RoboTwinEnv"""
    print("Starting RoboTwinEnv tests...")

    # Create configuration
    cfg = create_test_config(env_name)

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

    print(f"\n🎉 {env_name}: All tests passed!")
    

import multiprocessing as mp
import traceback

def _worker_run_env_test(env_name: str, q: mp.Queue):
    """
    子进程入口：必须是 top-level function，spawn 才能 pickle。
    """
    try:
        test_robotwin_env(env_name)
        q.put(("success", None))
    except Exception:
        q.put(("error", traceback.format_exc()))


def run_test_with_timeout(env_name: str, timeout: int = 180) -> bool:
    """
    在子进程中运行测试，超时则判失败并强制终止子进程。
    """
    q = mp.Queue()
    p = mp.Process(target=_worker_run_env_test, args=(env_name, q), daemon=True)

    p.start()
    p.join(timeout)

    if p.is_alive():
        print(f"❌ {env_name}: TIMEOUT ({timeout}s)")
        p.terminate()
        p.join()
        return False

    # 子进程正常退出：从队列取结果
    if not q.empty():
        status, msg = q.get()
        if status == "success":
            print(f"✅ {env_name}: PASSED")
            return True
        else:
            print(f"❌ {env_name}: EXCEPTION\n{msg}")
            return False

    print(f"❌ {env_name}: UNKNOWN FAILURE (no message returned)")
    return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(4)
    env_names = [
        # 'handover_block',
        # 'lift_pot',
        # 'place_a2b_left',
        # 'place_a2b_right',
        # 'stack_bowls_two',
    ]
    
    tested = [
        'beat_block_hammer',
        'blocks_ranking_rgb',
        'click_bell',
        'handover_block',
        'handover_mic',
        'lift_pot',
        'move_can_pot',
        'move_pillbottle_pad',
        'move_stapler_pad',
        'pick_dual_bottles',
        'place_a2b_left',
        'place_a2b_right',
        'place_container_plate',
        'place_empty_cup',
        'place_mouse_pad',
        'place_phone_stand',
        'place_shoe',
        'put_bottles_dustbin',
        'shake_bottle',
        'stack_blocks_two',
        'stack_bowls_two',
    ]
    
    all_envs = [
        'shake_bottle_horizontally',
        'place_a2b_left',
        'move_pillbottle_pad',
        'place_bread_basket',
        'stack_bowls_two',
        'move_playingcard_away',
        'click_bell',
        'beat_block_hammer',
        'place_mouse_pad',
        'blocks_ranking_rgb',
        'handover_mic',
        'place_container_plate',
        'pick_dual_bottles',
        'place_phone_stand',
        'put_object_cabinet',
        'adjust_bottle',
        'place_burger_fries',
        'place_can_basket',
        'press_stapler',
        'place_object_stand',
        'place_dual_shoes',
        'pick_diverse_bottles',
        'scan_object',
        'place_bread_skillet',
        'move_can_pot',
        'stamp_seal',
        'place_fan',
        'rotate_qrcode',
        'open_laptop',
        'place_a2b_right',
        'open_microwave',
        'handover_block',
        'place_shoe',
        'stack_bowls_three',
        'place_empty_cup',
        'place_object_basket',
        'place_object_scale',
        'stack_blocks_three',
        'put_bottles_dustbin',
        'hanging_mug',
        'click_alarmclock',
        'stack_blocks_two',
        'shake_bottle',
        'lift_pot',
        'grab_roller',
        'dump_bin_bigbin',
        'place_cans_plasticbox',
        'blocks_ranking_size',
        'turn_switch',
        'move_stapler_pad',
    ]
    
    # for env in all_envs:
    #     if env not in tested:
    #         print(f"'{env}',")
    
    # need_to_test = [
    #     'open_laptop',
    # ]
    need_to_test = [
        # 'put_object_cabinet',
        # 'place_object_stand',
        # 'place_fan',
        # 'place_object_scale',
        # 'stack_blocks_three',
        # 'blocks_ranking_size',
    ]

    # for i in need_to_test:
    #     print(f"{i}", end=" ")
    
    for env_name in need_to_test:
        print(f"\n===== Testing {env_name} =====")
        run_test_with_timeout(env_name, timeout=60)

