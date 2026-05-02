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

"""PolaRiS environment wrapper for RLinf.

This module integrates PolaRiS (a Gaussian-splatting-based Real-to-Sim evaluation
framework built on IsaacLab) into the RLinf embodied RL infrastructure.

PolaRiS environments run inside a subprocess (via ``SubProcIsaacLabEnv``) because
Isaac Sim requires ``AppLauncher`` to be initialised before any IsaacLab imports.
The wrapper follows the same pattern as the existing IsaacLab environment
(``rlinf.envs.isaaclab``).
"""

import os
import sys

import gymnasium as gym
import numpy as np
import torch
from omegaconf import open_dict

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv


class PolarisEnv(IsaaclabBaseEnv):
    """RLinf wrapper for PolaRiS environments.

    The wrapper:
    - Launches PolaRiS inside a subprocess via ``SubProcIsaacLabEnv``.
    - Exposes the canonical RLinf observation dict (``main_images``,
      ``wrist_images``, ``states``, ``task_descriptions``).
    - Supports ``chunk_step``, metrics tracking, auto-reset and
      ``ignore_terminations`` inherited from ``IsaaclabBaseEnv``.
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        # PolaRiS currently supports only num_envs=1 per subprocess because
        # the Gaussian-splat renderer is not vectorised.  Guard against
        # accidental misconfiguration.
        if num_envs != 1:
            raise ValueError(
                f"PolarisEnv only supports num_envs=1 per worker, got {num_envs}. "
                "Adjust total_num_envs and env_world_size so that each env "
                "worker receives exactly 1 environment."
            )
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )

    # ------------------------------------------------------------------
    # IsaaclabBaseEnv abstract methods
    # ------------------------------------------------------------------

    def _make_env_function(self):
        """Return a factory that creates the PolaRiS env inside a subprocess.

        The factory is pickled and shipped to a child process by
        ``SubProcIsaacLabEnv``.  It must:
        1. Start the Isaac Sim ``AppLauncher``.
        2. Create a ``ManagerBasedRLSplatEnv`` wrapped inside a thin
           ``InnerPolarisEnv`` adapter that translates the standard
           ``reset(seed, env_ids)`` / ``step(actions)`` protocol used by
           ``SubProcIsaacLabEnv._torch_worker`` into PolaRiS-specific calls.
        3. Return ``(inner_env, sim_app)``.
        """
        cfg = self.cfg
        task_name = self.isaaclab_env_id
        num_envs = self.num_envs
        seed = self.seed

        def _make_polaris():
            import os

            for key in list(sys.modules.keys()):
                if key.startswith("polaris.") or key.startswith("isaaclab.") or key.startswith("isaacsim") or key.startswith("omni."):
                    del sys.modules[key]

            # --- 1. clean env vars & launch Isaac Sim ----------------------
            # Remove DISPLAY variable to force headless mode and avoid GLX errors
            os.environ.pop("DISPLAY", None)

            # Resolve dataset path from config or environment variable.

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            # --- 2. import PolaRiS (must be after AppLauncher) -------------
            sys.stderr.write("==============POLARIS before import polaris.environments\n")
            sys.stderr.flush()
            import polaris.environments  # noqa: F401  (registers gym envs)
            sys.stderr.write("==============POLARIS after import polaris.environments\n")
            sys.stderr.flush()
            from polaris.utils import load_eval_initial_conditions, parse_env_cfg

            dataset_path = getattr(cfg.init_params, "dataset_path", None)
            if dataset_path is None:
                dataset_path = os.environ.get("POLARIS_DATA_PATH", None)
            if dataset_path is not None:
                os.environ["POLARIS_DATA_PATH"] = str(dataset_path)

            sys.stderr.write("==============POLARIS after dataset_path\n")
            sys.stderr.flush()
            # import fuck_2

            # --- 3. build the inner env ------------------------------------
            usd_file = cfg.init_params.usd_file
            env_cfg = parse_env_cfg(
                task_name,
                usd_file=usd_file,
                device="cuda",
                num_envs=num_envs,
                use_fabric=True,
            )
            env_cfg.seed = seed

            # import fuck_3
            sys.stderr.write("==============POLARIS after env_cfg\n")
            sys.stderr.flush()

            # Override camera resolution from RLinf config if provided.
            if hasattr(cfg.init_params, "wrist_cam"):
                env_cfg.scene.wrist_cam.height = cfg.init_params.wrist_cam.height
                env_cfg.scene.wrist_cam.width = cfg.init_params.wrist_cam.width
            if hasattr(cfg.init_params, "table_cam"):
                for cam_name in ("external_cam",):
                    if hasattr(env_cfg.scene, cam_name):
                        getattr(env_cfg.scene, cam_name).height = (
                            cfg.init_params.table_cam.height
                        )
                        getattr(env_cfg.scene, cam_name).width = (
                            cfg.init_params.table_cam.width
                        )

            # import fuck_4

            sys.stderr.write("==============POLARIS before env create\n")
            sys.stderr.flush()
            real_env = gym.make(task_name, cfg=env_cfg)
            sys.stderr.write("==============POLARIS after env create\n")
            sys.stderr.flush()

            # Load language instruction and pre-defined initial conditions.
            language_instruction, initial_conditions = (
                load_eval_initial_conditions(real_env.usd_file)
            )

            # import fuck_5

            # --- 4. thin adapter ------------------------------------------
            class _InnerPolarisEnv:
                """Adapts PolaRiS's custom API to the protocol expected by
                ``SubProcIsaacLabEnv._torch_worker`` (``reset`` / ``step`` /
                ``close`` / ``device``)."""

                def __init__(self):
                    self.env = real_env
                    self.language_instruction = language_instruction
                    self.initial_conditions = initial_conditions
                    self._ic_idx = 0
                    self.device = "cuda"

                # -- standard API consumed by _torch_worker --

                def reset(self, seed=None, env_ids=None):
                    ic = self.initial_conditions[self._ic_idx]
                    obs, info = self.env.reset(
                        object_positions=ic, expensive=True
                    )
                    self._ic_idx = (self._ic_idx + 1) % len(
                        self.initial_conditions
                    )
                    return obs, info

                def step(self, actions):
                    if not isinstance(actions, torch.Tensor):
                        actions = torch.as_tensor(actions, device=self.device)
                    else:
                        actions = actions.to(self.device)
                        
                    # Pad 7D actions to 8D to bypass action dimension mismatch during pipeline testing
                    if actions.shape[-1] == 7:
                        padding = torch.zeros(actions.shape[:-1] + (1,), dtype=actions.dtype, device=self.device)
                        actions = torch.cat([actions, padding], dim=-1)
                        
                    return self.env.step(actions, expensive=True)

                def close(self):
                    self.env.close()

            sys.stderr.write("==============POLARIS before _InnerPolarisEnv\n")
            sys.stderr.flush()

            inner_env = _InnerPolarisEnv()

            sys.stderr.write("==============POLARIS after _InnerPolarisEnv\n")
            sys.stderr.flush()
            return inner_env, sim_app

        return _make_polaris

    def _wrap_obs(self, obs):
        """Convert raw PolaRiS observations into the canonical RLinf dict.

        Keys returned
        -------------
        main_images : torch.Tensor  [num_envs, H, W, 3]
            External (table) camera RGB.
        wrist_images : torch.Tensor  [num_envs, H, W, 3]
            Wrist camera RGB.
        states : torch.Tensor  [num_envs, 8]
            Robot proprioception: ``[arm_joint_pos (7), gripper_pos (1)]``.
        task_descriptions : list[str]
            Natural-language instruction repeated for every env.
        """
        # --- images from Gaussian-splat rendering --------------------------
        splat = obs.get("splat", {})
        # PolaRiS camera names vary per scene; fall back to sim RGB.
        main_img = splat.get("external_cam", splat.get("camera", None))
        wrist_img = splat.get("wrist_cam", None)

        def _to_tensor(img):
            if img is None:
                return None
            if isinstance(img, np.ndarray):
                return torch.from_numpy(img).to(self.device).unsqueeze(0)
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                return img.to(self.device)
            return None

        main_images = _to_tensor(main_img)
        wrist_images = _to_tensor(wrist_img)

        # --- proprioceptive state ------------------------------------------
        policy = obs.get("policy", {})
        arm_joint_pos = policy.get("arm_joint_pos", None)
        gripper_pos = policy.get("gripper_pos", None)

        if arm_joint_pos is not None and gripper_pos is not None:
            if isinstance(arm_joint_pos, np.ndarray):
                arm_joint_pos = torch.from_numpy(arm_joint_pos).to(self.device)
            if isinstance(gripper_pos, np.ndarray):
                gripper_pos = torch.from_numpy(gripper_pos).to(self.device)
            # Ensure batch dimension [num_envs, ...]
            if arm_joint_pos.dim() == 1:
                arm_joint_pos = arm_joint_pos.unsqueeze(0)
            if gripper_pos.dim() == 1:
                gripper_pos = gripper_pos.unsqueeze(0)
            states = torch.cat([arm_joint_pos, gripper_pos], dim=-1).float()
        else:
            # Fallback: zeros (should not happen in a correctly configured env)
            states = torch.zeros(
                (self.num_envs, 8), dtype=torch.float32, device=self.device
            )

        instruction = [self.task_description] * self.num_envs

        env_obs = {
            "task_descriptions": instruction,
            "states": states,
        }
        if main_images is not None:
            env_obs["main_images"] = main_images
        if wrist_images is not None:
            env_obs["wrist_images"] = wrist_images

        return env_obs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_num_group_envs(self):
        return self.num_envs // self.cfg.group_size
