"""
Base environment for Bridge dataset environments
"""
import os
from typing import Dict, List, Literal

import numpy as np
import sapien
import torch
from sapien.physx import PhysxMaterial
from transforms3d.quaternions import quat2mat
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR

from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseControllerConfig
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosMimicControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.agents.robots.widowx.widowx import WidowX250S
from mani_skill.utils.registration import register_env
from copy import deepcopy
from mani_skill.agents.controllers import *


BRIDGE_DATASET_ASSET_PATH = ASSET_DIR / "tasks/bridge_v2_real2sim_dataset/"
@register_agent()
class PandaBridgeDatasetFlatTable(PandaWristCam):
    """Panda arm robot with the real sense camera attached to gripper"""
    uid = "panda_bridgedataset_flat_table"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1, # -1.0,
            pos_upper=0.1, # 1.0,
            rot_lower=-0.1, # -np.pi / 2,
            rot_upper=0.1, # np.pi / 2,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=False,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True
        arm_pd_ee_target_delta_pose_body = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose_body.frame = "root_translation:body_aligned_body_rotation"

        arm_pd_ee_delta_pose_real_root_frame = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1, # -1.0,
            pos_upper=0.1, # 1.0,
            rot_lower=-0.1, # -np.pi / 2,
            rot_upper=0.1, # np.pi / 2,
            stiffness=[37.800000000000004, 29.925, 48.3, 48.3, 2.1284343434343436, 27.3, 48.3],
            damping=[10.5,  10.5,  10.5,  10.5,  0.6353535353535353, 10.5,  10.5],
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=False,
        )
        arm_pd_ee_delta_pose_real_root_frame.use_target = True
        arm_pd_ee_delta_pose_real = deepcopy(arm_pd_ee_delta_pose_real_root_frame)
        arm_pd_ee_delta_pose_real.frame = "root_translation:body_aligned_body_rotation"

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_ee_body_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose_body, gripper=gripper_pd_joint_pos
            ),
            pd_ee_body_target_delta_pose_real=dict(
                arm=arm_pd_ee_delta_pose_real, gripper=gripper_pd_joint_pos
            ),
            pd_ee_body_target_delta_pose_real_root_frame=dict(
                arm=arm_pd_ee_delta_pose_real_root_frame, gripper=gripper_pd_joint_pos
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_pos),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    
    @property
    def ee_pose_at_robot_base(self): # in robot frame(root frame)
        to_base = self.robot.pose.inv()
        return to_base * (self.tcp.pose)   

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used in the Bridge dataset
                pose=sapien.Pose([0.147, 0.028, 0.870], q=[0, 0, 0, 1])*sapien.Pose(
                    [0, -0.16, 0.36], # 0, -0.16, 0.36
                    [0.8992917, -0.09263245, 0.35892478, 0.23209205],
                ),
                width=640,
                height=480,
                # entity_uid="panda_link0",
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),  # logitech C920
            ),
            CameraConfig(
                uid="c19_front_view",
                pose=sapien.Pose([0.1840, 0.2000, 1.4000], q=[0.2541,  0.3510,  0.1361, -0.8909]),
                width=640,
                height=480,
                fov=0.81,
                near=0.1,
                far=1000,
            ),
        ]

# Tuned for the sink setup
@register_agent()
class PandaBridgeDatasetSink(PandaBridgeDatasetFlatTable):
    """Panda arm robot with the real sense camera attached to gripper"""
    uid = "panda_bridgedataset_sink"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used for real evaluation for the sink setup
                pose=sapien.Pose([0.127, 0.060, 0.85], q=[0, 0, 0, 1])*sapien.Pose(
                    [-0.00300001, -0.21, 0.39],
                    [-0.907313, 0.0782, -0.36434, -0.194741] # wxyz
                ),
                width=640,
                # fov=1.5,
                height=480,
                near=0.01,
                far=10,
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),
            ),
            CameraConfig(
                uid = "near_third_view", 
                pose = sapien.Pose(
                    [0.116329, 0.52049, 1.23331], [0.47342, 0.1204, 0.0655081, -0.870107]
                ),
                width=640, 
                height=480, 
                fov=1.57,
                near=0.1,
                far=1000
            ),
        ]


class PandaBaseBridgeEnv(BaseDigitalTwinEnv):
    """Base Digital Twin environment for digital twins of the BridgeData v2"""

    MODEL_JSON = "info_bridge_custom_v0.json"
    SUPPORTED_OBS_MODES = ["rgb+segmentation"]
    SUPPORTED_REWARD_MODES = ["none"]
    scene_setting: Literal["flat_table", "sink"] = "flat_table"

    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5

    def __init__(
        self,
        obj_names: List[str],
        xyz_configs: torch.Tensor,
        quat_configs: torch.Tensor,
        **kwargs,
    ):
        self.objs: Dict[str, Actor] = dict()
        self.obj_names = obj_names
        self.source_obj_name = obj_names[0]
        self.target_obj_name = obj_names[1]
        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs
        if self.scene_setting == "flat_table":
            self.rgb_overlay_paths = {
                "3rd_view_camera": str(
                    BRIDGE_DATASET_ASSET_PATH / "real_inpainting/bridge_real_eval_1.png"
                )
            }
            robot_cls = PandaBridgeDatasetFlatTable
        elif self.scene_setting == "sink":
            self.rgb_overlay_paths = {
                "3rd_view_camera": str(
                    BRIDGE_DATASET_ASSET_PATH / "real_inpainting/bridge_sink.png"
                )
            }
            robot_cls = PandaBridgeDatasetSink

        self.model_db: Dict[str, Dict] = io_utils.load_json(
            BRIDGE_DATASET_ASSET_PATH / "custom/" / self.MODEL_JSON
        )
        self.src_on_target_continuous_steps = torch.zeros(kwargs['num_envs'])
        super().__init__(
            robot_uids=robot_cls,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=5, spacing=20)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien.Pose([0.442614, 0.488839, 1.45059], [0.39519, 0.210508, 0.0936785, -0.889233])
        return CameraConfig("render_camera", pose, 512, 512, 1.45, 0.1, 1000)

    def _build_actor_helper(
        self,
        model_id: str,
        scale: float = 1,
        kinematic: bool = False,
        initial_pose: Pose = None,
    ):
        """helper function to build actors by ID directly and auto configure physical materials"""
        density = self.model_db[model_id].get("density", 1000)
        physical_material = PhysxMaterial(
            static_friction=self.obj_static_friction,
            dynamic_friction=self.obj_dynamic_friction,
            restitution=0.0,
        )
        builder = self.scene.create_actor_builder()
        model_dir = BRIDGE_DATASET_ASSET_PATH / "custom" / "models" / model_id

        collision_file = str(model_dir / "collision.obj")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )

        visual_file = str(model_dir / "textured.obj")
        if not os.path.exists(visual_file):
            visual_file = str(model_dir / "textured.dae")
            if not os.path.exists(visual_file):
                visual_file = str(model_dir / "textured.glb")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)
        if initial_pose is not None:
            builder.initial_pose = initial_pose
        if kinematic:
            actor = builder.build_kinematic(name=model_id)
        else:
            actor = builder.build(name=model_id)
        return actor

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [0, 0, -1],
            [2.2, 2.2, 2.2],
            shadow=False,
            shadow_scale=5,
            shadow_map_size=2048,
        )
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _load_agent(self, options: dict): # TODO
        super()._load_agent( # the initial pose will be changed by self.agent.robot.set_pose() in _initialize_episode()
            options, initial_agent_poses = sapien.Pose(p=[0.127, 0.060, 0.85], q=[0, 0, 0, 1])
        )

    def _load_scene(self, options: dict):
        # original SIMPLER envs always do this? except for open drawer task
        for i in range(self.num_envs):
            sapien_utils.set_articulation_render_material(
                self.agent.robot._objs[i], specular=0.9, roughness=0.3
            )

        # load background
        builder = self.scene.create_actor_builder()
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])
        scene_offset = np.array([-2.0634, -2.8313, 0.0])
        if self.scene_setting == "flat_table":
            scene_file = str(BRIDGE_DATASET_ASSET_PATH / "stages/bridge_table_1_v1.glb")

        elif self.scene_setting == "sink":
            scene_file = str(BRIDGE_DATASET_ASSET_PATH / "stages/bridge_table_1_v2.glb")
        builder.add_nonconvex_collision_from_file(scene_file, pose=scene_pose)
        builder.add_visual_from_file(scene_file, pose=scene_pose)

        builder.initial_pose = sapien.Pose(-scene_offset)
        builder.build_static(name="arena")

        for name in self.obj_names:
            self.objs[name] = self._build_actor_helper(name)

        self.xyz_configs = common.to_tensor(self.xyz_configs, device=self.device).to(
            torch.float32
        )
        self.quat_configs = common.to_tensor(self.quat_configs, device=self.device).to(
            torch.float32
        )

        if self.scene_setting == "sink":
            self.sink = self._build_actor_helper(
                "sink",
                kinematic=True,
                initial_pose=sapien.Pose([-0.16, 0.13, 0.88], [1, 0, 0, 0]),
            )
        # model scales
        model_scales = None
        if model_scales is None:
            model_scales = dict()
            for model_id in [self.source_obj_name, self.target_obj_name]:
                this_available_model_scales = self.model_db[model_id].get(
                    "scales", None
                )
                if this_available_model_scales is None:
                    model_scales.append(1.0)
                else:
                    model_scales[model_id] = self.np_random.choice(
                        this_available_model_scales
                    )
        self.episode_model_scales = model_scales
        model_bbox_sizes = dict()
        for model_id in [self.source_obj_name, self.target_obj_name]:
            model_info = self.model_db[model_id]
            model_scale = self.episode_model_scales[model_id]
            if "bbox" in model_info:
                bbox = model_info["bbox"]
                bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
                model_bbox_sizes[model_id] = common.to_tensor(
                    bbox_size * model_scale, device=self.device
                )
            else:
                raise ValueError(f"Model {model_id} does not have bbox info.")
        self.episode_model_bbox_sizes = model_bbox_sizes

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            if "episode_id" in options:
                if isinstance(options["episode_id"], int):
                    options["episode_id"] = torch.tensor([options["episode_id"]])
                    assert len(options["episode_id"]) == b
                pos_episode_ids = (
                    options["episode_id"]
                    % (len(self.xyz_configs) * len(self.quat_configs))
                ) // len(self.quat_configs)
                quat_episode_ids = options["episode_id"] % len(self.quat_configs)
            else:
                pos_episode_ids = torch.randint(0, len(self.xyz_configs), size=(b,))
                quat_episode_ids = torch.randint(0, len(self.quat_configs), size=(b,))
            for i, actor in enumerate(self.objs.values()):
                xyz = self.xyz_configs[pos_episode_ids, i]
                actor.set_pose( # set the pose, but not change the bouding box pose. May be a problem
                    Pose.create_from_pq(p=xyz, q=self.quat_configs[quat_episode_ids, i])
                )
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
            self._settle(0.5)
            if self.gpu_sim_enabled:
                self.scene._gpu_fetch_all()
            # Some objects need longer time to settle
            lin_vel, ang_vel = 0.0, 0.0
            for obj_name, obj in self.objs.items():
                lin_vel += torch.linalg.norm(obj.linear_velocity)
                ang_vel += torch.linalg.norm(obj.angular_velocity)
            if lin_vel > 1e-3 or ang_vel > 1e-2:
                if self.gpu_sim_enabled:
                    self.scene._gpu_apply_all()
                self._settle(6)
                if self.gpu_sim_enabled:
                    self.scene._gpu_fetch_all()
            # measured values for bridge dataset
            if self.scene_setting == "flat_table":
                qpos = np.array(
                    [0, 0.259, 0, -2.289, 0, 2.515, np.pi / 4, 0.015, 0.015]
                )
                self.agent.robot.set_pose(
                    sapien.Pose([0.3, 0.028, 0.870], q=[0, 0, 0, 1]) # 0.147, 0.028, 0.870 # modified to better view object at [0.3, 0.028, 0.870]
                )
            elif self.scene_setting == "sink":
                qpos = np.array(
                    [0.0, 0.08580994, 0.0, -2.3964953, 0, 2.5136616, 0.7859319, 0.04, 0.04]
                )
                self.agent.robot.set_pose(
                    sapien.Pose([0.3, 0.060, 0.85], q=[0, 0, 0, 1])
                )
            self.agent.reset(init_qpos=qpos)

            # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
            self.episode_source_obj_xyz_after_settle = self.objs[
                self.source_obj_name
            ].pose.p
            self.episode_target_obj_xyz_after_settle = self.objs[
                self.target_obj_name
            ].pose.p
            self.episode_obj_xyzs_after_settle = {
                obj_name: self.objs[obj_name].pose.p for obj_name in self.objs.keys()
            }
            self.episode_source_obj_bbox_world = self.episode_model_bbox_sizes[
                self.source_obj_name
            ].float()
            self.episode_target_obj_bbox_world = self.episode_model_bbox_sizes[
                self.target_obj_name
            ].float()
            self.episode_source_obj_bbox_world = (
                rotation_conversions.quaternion_to_matrix(
                    self.objs[self.source_obj_name].pose.q
                )
                @ self.episode_source_obj_bbox_world[..., None]
            )[0, :, 0]
            """source object bbox size (3, )"""
            self.episode_target_obj_bbox_world = (
                rotation_conversions.quaternion_to_matrix(
                    self.objs[self.target_obj_name].pose.q
                )
                @ self.episode_target_obj_bbox_world[..., None]
            )[0, :, 0]
            """target object bbox size (3, )"""

            # stats to track
            self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32)
            self.episode_stats = dict(
                # all_obj_keep_height=torch.zeros((b,), dtype=torch.bool),
                moved_correct_obj=torch.zeros((b,), dtype=torch.bool),
                moved_wrong_obj=torch.zeros((b,), dtype=torch.bool),
                # near_tgt_obj=torch.zeros((b,), dtype=torch.bool),
                is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool),
                # is_closest_to_tgt=torch.zeros((b,), dtype=torch.bool),
                consecutive_grasp=torch.zeros((b,), dtype=torch.bool),
            )

    def _settle(self, t: int = 0.5):
        """run the simulation for some steps to help settle the objects"""
        sim_steps = int(self.sim_freq * t / self.control_freq)
        for _ in range(sim_steps):
            self.scene.step()

    def _evaluate(
        self,
        success_require_src_completely_on_target=True,
        z_flag_required_offset=0.01,
        **kwargs,
    ):
        source_object = self.objs[self.source_obj_name]
        target_object = self.objs[self.target_obj_name]
        source_obj_pose = source_object.pose
        target_obj_pose = target_object.pose

        # whether moved the correct object
        source_obj_xy_move_dist = torch.linalg.norm(
            self.episode_source_obj_xyz_after_settle[:, :2] - source_obj_pose.p[:, :2],
            dim=1,
        )
        other_obj_xy_move_dist = []
        for obj_name in self.objs.keys():
            obj = self.objs[obj_name]
            obj_xyz_after_settle = self.episode_obj_xyzs_after_settle[obj_name]
            if obj.name == self.source_obj_name:
                continue
            other_obj_xy_move_dist.append(
                torch.linalg.norm(
                    obj_xyz_after_settle[:, :2] - obj.pose.p[:, :2], dim=1
                )
            )

        # whether the source object is grasped
        is_src_obj_grasped = self.agent.is_grasping(source_object)
        # if is_src_obj_grasped:
        self.consecutive_grasp += is_src_obj_grasped
        self.consecutive_grasp[is_src_obj_grasped == 0] = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        # whether the source object is on the target object based on bounding box position
        tgt_obj_half_length_bbox = (
            self.episode_target_obj_bbox_world / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_half_length_bbox = self.episode_source_obj_bbox_world / 2

        pos_src = source_obj_pose.p
        pos_tgt = target_obj_pose.p
        offset = pos_src - pos_tgt
        xy_flag = (
            torch.linalg.norm(offset[:, :2], dim=1)
            <= torch.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.003
        )
        z_flag = (offset[:, 2] > 0) & (
            offset[:, 2] - tgt_obj_half_length_bbox[2] - src_obj_half_length_bbox[2]
            <= z_flag_required_offset
        )
        src_on_target = xy_flag & z_flag

        if success_require_src_completely_on_target:
            # whether the source object is on the target object based on contact information
            contact_forces = self.scene.get_pairwise_contact_forces(
                source_object, target_object
            )
            net_forces = torch.linalg.norm(contact_forces, dim=1)
            src_on_target = src_on_target & (net_forces > 0.05)

        continuous_success = torch.zeros_like(src_on_target).to(self.device)
        for i in range(src_on_target.shape[0]):
            if src_on_target[i]:
                self.src_on_target_continuous_steps[i] += 1
            else:
                self.src_on_target_continuous_steps[i] = 0
            continuous_success[i] = True if self.src_on_target_continuous_steps[i]>6 else False
        success = src_on_target

        self.episode_stats["src_on_target"] = src_on_target
        self.episode_stats["is_src_obj_grasped"] = (
            self.episode_stats["is_src_obj_grasped"] | is_src_obj_grasped
        )
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] | consecutive_grasp
        )

        return dict(**self.episode_stats, success=success, continuous_success=continuous_success)

    def is_final_subtask(self):
        # whether the current subtask is the final one, only meaningful for long-horizon tasks
        return True

