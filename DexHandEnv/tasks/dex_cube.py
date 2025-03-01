import os
import sys
import time
import json

import torch
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_angle_axis

from DexHandEnv.tasks.base.vec_task import VecTask
from DexHandEnv.utils.torch_jit_utils import (
    quat_mul,
    tensor_clamp,
    to_torch,
    axisangle2quat,
)
from DexHandEnv.utils.ycb_object_utils import YCBLoader
from DexHandEnv.utils.graspnet_object_utils import GraspNetLoader
from DexHandEnv.utils.trajectory import TrajectoryManager

# x 前后，y 左右，z 上下
# 一个具有 6 自由度基座和 5 指灵巧手的机器人结构
base_joint_names = [
    "base_ARTx",
    "base_ARTy",
    "base_ARTz",
    "base_ARRx",
    "base_ARRy",
    "base_ARRz",
]
finger_joint_names = [
    "r_f_joint1_1_pos",
    "r_f_joint1_2_pos",
    "r_f_joint1_3_pos",
    "r_f_joint1_4_pos",
    "r_f_joint2_1_pos",
    "r_f_joint2_2_pos",
    "r_f_joint2_3_pos",
    "r_f_joint2_4_pos",
    "r_f_joint3_1_pos",  # fixed
    "r_f_joint3_2_pos",
    "r_f_joint3_3_pos",
    "r_f_joint3_4_pos",
    "r_f_joint4_1_pos",
    "r_f_joint4_2_pos",
    "r_f_joint4_3_pos",
    "r_f_joint4_4_pos",
    "r_f_joint5_1_pos",
    "r_f_joint5_2_pos",
    "r_f_joint5_3_pos",
    "r_f_joint5_4_pos",
]


object_pregrasp_poses = {
    "001": [1],
    "002": [1],
    "003": [1],
    "007": [0, 1],
    "010": [0],
    "012": [1],
    "016": [0, 1],
    "022": [0, 1],
    "036": [0],
    "037": [0],
    "038": [0],
    "039": [0],
    "040": [0],
    "057": [0],
    "061": [0],
    "063": [0],
    "064": [0],
    "066": [0],
    "068": [0],
    "077": [1],
    "087": [0, 1],
}


class DexCube(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg

        self.object_category = "graspnet"
        self.record_trajectory = True
        self.object_idx_buf = None  # Store the object id for each env
        self.object_height_buf = None  # Store the object height for each env
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.dex_position_noise = self.cfg["env"]["dexPositionNoise"]
        self.dex_rotation_noise = self.cfg["env"]["dexRotationNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.control_type = self.cfg["env"]["controlType"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.reward_settings = {
            "w_distance_reward": self.cfg["env"]["distanceReward"],
            "w_object_height_reward": self.cfg["env"]["objectHeightReward"],
            "w_action_penalty": self.cfg["env"]["actionPenalty"],
        }

        logger.add(sys.stderr, level="DEBUG")

        # fmt: off
        self.cfg["env"]["numObservations"] = (
            20        # Pregrasp finger positions
            + 6       # Pregrasp hand pose
            + 20 * 2  # Current qpos and qvel
            + 5       # Contact state
        )
        self.cfg["env"]["numActions"] = 20  # 6 + 20

        # Runtime state
        self.states = {}                 # Stores all environment states
        self.handles = {}                # Stores actor/body handles
        self.actions = None              # Stores joint position targets

        # Tensor placeholders
        self._root_state = None          # State of root body        (n_envs, n_actor, 13) 
        self._dof_state = None           # State of all joints       (n_envs, n_dof, 2)
        self._q = None                   # Joint positions           (n_envs, n_dof)
        self._dq = None                  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = (None)  # State of all rigid bodies (n_envs, n_bodies, 13)
        self._eef_state = None           # State of end effector
        self._pos_control = None         # Position actions
        self._dex_effort_limits = None   # Actuator effort limits for dex
        self._global_indices = (None)    # Unique indices corresponding to all envs in flattened array

        # Set up finger tip names
        # TODO: Optimize (fingertip sensor and real fingertip)
        self.fingertip_names = [
            "r_f_link1_4_tensor",
            "r_f_link2_4_tensor",
            "r_f_link3_4_tensor",
            "r_f_link4_4_tensor",
            "r_f_link5_4_tensor",
        ]
        # fmt: on
        if self.record_trajectory:
            self.trajectory_manager = TrajectoryManager()
            self.episode_counter = 1

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        if self.num_envs > 1:
            self.record_trajectory = False

        # Load pregrasp pose
        self.pregrasp_dataset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            cfg["env"]["pregraspDatasetPath"],
        )
        self.finger_joint_qpos, self.base_joint_pose = self.load_pregrasp_dataset()
        self.pregrasp_indices = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

        # Set up success statistics
        self.print_success_stat = True
        if self.num_envs > 100:
            self.print_success_stat = False
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.total_successes = 0
        self.total_resets = 0

        # Set up joint constraints
        self.joint_constraints = {
            "coupled": [
                ("r_f_joint1_4", "r_f_joint1_3"),
                ("r_f_joint2_4", "r_f_joint2_3"),
                ("r_f_joint3_4", "r_f_joint3_3"),
                ("r_f_joint4_4", "r_f_joint4_3"),
                ("r_f_joint5_4", "r_f_joint5_3"),
            ],
            "follow": [
                ("r_f_joint4_1", "r_f_joint2_1"),
            ],
            "fixed": [
                ("r_f_joint3_1", 0.0),
            ],
            "scaled": [
                ("r_f_joint5_1", "r_f_joint2_1", 2),
            ],
        }

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh()  # Refresh tensors

    def _update_tensor_views(self):
        self._q = self._dof_state.view(self.num_envs, self.num_dex_dofs, 2)[..., 0]
        self._dq = self._dof_state.view(self.num_envs, self.num_dex_dofs, 2)[..., 1]
        self._eef_state = self._rigid_body_state[:, self.ee_indices, :]
        self._dex_state = self._root_state[:, self.handles["dexhand"], :]
        self._object_state = self._root_state[:, self.handles["object"], :]

    def load_pregrasp_dataset(self):
        df = pd.read_csv(self.pregrasp_dataset_path, float_precision="round_trip")
        missing_base = [col for col in base_joint_names if col not in df.columns]
        missing_finger = [col for col in finger_joint_names if col not in df.columns]
        assert (
            len(missing_base) == 0 and len(missing_finger) == 0
        ), f"Missing columns in pregrasp dataset: {missing_base}, {missing_finger}"
        base_joint_pose = df[base_joint_names].values
        finger_joint_qpos = df[finger_joint_names].values
        assert (
            base_joint_pose.shape[1] == 6 and finger_joint_qpos.shape[1] == 20
        ), "Expected 6 base joints and 20 finger joints (including one fixed joint)"

        return (
            torch.tensor(finger_joint_qpos, dtype=torch.float32).to(self.device),
            torch.tensor(base_joint_pose, dtype=torch.float32).to(self.device),
        )

    def build_joint_mappings(self):
        """
        Build joint mappings for the DexHand
        """
        num_dofs = self.gym.get_asset_dof_count(self.dex_asset)

        self.joint_name_to_index = {}
        for i in range(num_dofs):
            dof_name = self.gym.get_asset_dof_name(self.dex_asset, i)
            self.joint_name_to_index[dof_name] = i
        logger.info(f"Joint name to index: {self.joint_name_to_index}")

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def load_object(self, object_name=None):
        """
        Load object asset based on configuration

        Returns:
            object_asset: Asset handle
            object_start_pose: Initial pose for the object
            object_height: Height of the object for initial pose (if available, else None)
        """
        # Create base asset options
        object_opts = gymapi.AssetOptions()
        object_opts.density = 500 * (
            0.8 + 0.4 * torch.rand((1,), device=self.device).item()
        )
        object_opts.fix_base_link = False
        object_opts.disable_gravity = False

        if self.object_category.lower() == "ycb":
            ycb_loader = YCBLoader(self.gym, self.sim)
            object_asset, object_start_pose, object_height = (
                ycb_loader.load_dataset_object(object_name)
            )
        elif self.object_category.lower() == "graspnet":
            grasp_net_loader = GraspNetLoader(self.gym, self.sim)
            object_asset, object_start_pose, object_height = (
                grasp_net_loader.load_dataset_object(object_name)
            )
        elif self.object_category.lower() == "cube":
            object_height = self.cfg["env"].get("object_size", 0.06)
            cube_size = [object_height] * 3
            object_asset = self.gym.create_box(self.sim, *cube_size, object_opts)
            object_start_pose = gymapi.Transform()
            object_start_pose.p = gymapi.Vec3(0.0, 0.0, object_height / 2)
        else:
            pass

        return object_asset, object_start_pose, object_height

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )
        dex_asset_file = None

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.cfg["env"]["asset"].get("assetRoot", asset_root),
            )
            dex_asset_file = self.cfg["env"]["asset"].get(
                "assetFileNamedex", dex_asset_file
            )

        # Load dex asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = 3
        asset_options.use_mesh_materials = True
        self.dex_asset = self.gym.load_asset(
            self.sim, asset_root, dex_asset_file, asset_options
        )

        # Build joint mappings
        self.build_joint_mappings()

        # Create fingertip force sensors
        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(self.dex_asset, name)
            for name in self.fingertip_names
        ]
        sensor_pose = gymapi.Transform()
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False  # for example gravity
        sensor_options.enable_constraint_solver_forces = True  # for example contacts
        sensor_options.use_world_frame = False  # report forces in world frame
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(
                self.dex_asset, ft_handle, sensor_pose, sensor_options
            )

        self.num_dex_bodies = self.gym.get_asset_rigid_body_count(self.dex_asset)
        self.num_dex_dofs = self.gym.get_asset_dof_count(self.dex_asset)
        logger.info(f"num dex dofs: {self.num_dex_dofs}")
        logger.info(f"num dex bodies: {self.num_dex_bodies}")
        dex_dof_props = self.gym.get_asset_dof_properties(self.dex_asset)
        self.dex_dof_lower_limits = []
        self.dex_dof_upper_limits = []
        self._dex_effort_limits = []
        for i in range(self.num_dex_dofs):
            dex_dof_props["driveMode"][i] = 3  # gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                dex_dof_props["stiffness"][i] = 5000
                dex_dof_props["damping"][i] = 100
            else:
                dex_dof_props["stiffness"][i] = 7000.0
                dex_dof_props["damping"][i] = 50.0

            self.dex_dof_lower_limits.append(dex_dof_props["lower"][i])
            self.dex_dof_upper_limits.append(dex_dof_props["upper"][i])
            self._dex_effort_limits.append(dex_dof_props["effort"][i])
        logger.info(f"dex dof lower limits: {self.dex_dof_lower_limits}")
        logger.info(f"dex dof upper limits: {self.dex_dof_upper_limits}")
        logger.info(f"dex effort limits: {self._dex_effort_limits}")
        logger.info(f"dex dof props: {dex_dof_props}")

        self.dex_dof_lower_limits = to_torch(
            self.dex_dof_lower_limits, device=self.device
        )
        self.dex_dof_upper_limits = to_torch(
            self.dex_dof_upper_limits, device=self.device
        )
        self._dex_effort_limits = to_torch(self._dex_effort_limits, device=self.device)

        # Create table asset
        table_thickness = 0.05
        table_pos = [0.0, 0.0, 0.735]
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        self.table_asset = self.gym.create_box(
            self.sim, *[2, 2, table_thickness], table_opts
        )
        # Define start pose for dex
        dex_start_pose = gymapi.Transform()

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array(
            [0, 0, table_thickness / 2]
        )
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define object target pose
        self.object_target_pose = torch.tensor(
            [0.0, 0.0, self._table_surface_pos[2] + 0.4],
            dtype=torch.float32,
            device=self.device,
        )

        # Compute aggregate size
        num_dex_bodies = self.gym.get_asset_rigid_body_count(self.dex_asset)
        num_dex_shapes = self.gym.get_asset_rigid_shape_count(self.dex_asset)
        max_agg_bodies = (num_dex_bodies + 2) * 2
        max_agg_shapes = (num_dex_shapes + 2) * 2

        self.dexs = []
        self.envs = []
        self.object_idx_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.object_height_buf = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        # Load object asset
        self.object_assets = []
        self.object_poses = []
        self.object_heights = []

        if self.object_category == "ycb":
            self.ycb_loader = YCBLoader(self.gym, self.sim)
            self.available_object_names = self.ycb_loader.get_available_object_names()
        elif self.object_category == "graspnet":
            self.grasp_net_loader = GraspNetLoader(self.gym, self.sim)
            self.available_object_names = (
                self.grasp_net_loader.get_available_object_names()
            )
            # when training
            # self.available_object_names = ["016"]
        elif self.object_category == "cube":
            self.available_object_names = ["cube"]
        else:
            raise ValueError(f"Object category {self.object_category} not supported")

        for obj_name in self.available_object_names:
            asset, start_pose, height = self.load_object(obj_name)
            self.object_assets.append(asset)
            self.object_poses.append(start_pose)
            self.object_heights.append(height)

        # Create environments
        for env_idx in range(self.num_envs):
            # TODO: More domain randomization
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create dex
            # if self.dex_position_noise > 0:
            #     rand_xy = self.dex_position_noise * (-1.0 + np.random.rand(2) * 2.0)
            #     dex_start_pose.p = gymapi.Vec3(
            #         0.0,
            #         0.0,
            #         0.0,
            #     )
            # if self.dex_rotation_noise > 0:
            #     rand_rot = torch.zeros(1, 3)
            #     rand_rot[:, -1] = self.dex_rotation_noise * (-1.0)
            #     new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
            #     dex_start_pose.r = gymapi.Quat(*new_quat)
            obj_idx = env_idx % len(self.available_object_names)
            dex_start_pose.p = gymapi.Vec3(
                -0.18,
                -0.0,
                table_pos[2]
                + table_thickness / 2
                + self.object_heights[obj_idx]
                + 0.10,
            )
            dex_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            self.handles["dexhand"] = self.gym.create_actor(
                env_ptr,
                self.dex_asset,
                dex_start_pose,
                "dex",
                env_idx,  # group
                1,  # mask
            )
            self.gym.set_actor_dof_properties(
                env_ptr, self.handles["dexhand"], dex_dof_props
            )

            # Create table
            self.handles["table"] = self.gym.create_actor(
                env_ptr,
                self.table_asset,
                table_start_pose,
                "table",
                env_idx,
                0,
            )

            # Create objects
            self.object_name = self.available_object_names[obj_idx]
            self.object_idx_buf[env_idx] = obj_idx
            self.object_height_buf[env_idx] = self.object_heights[obj_idx]

            self.handles["object"] = self.gym.create_actor(
                env_ptr,
                self.object_assets[obj_idx],
                self.object_poses[obj_idx],
                f"{self.object_name}",
                env_idx,
                0,
            )
            if self.object_category == "cube":
                cmap = plt.get_cmap(f"tab20")
                colors = cmap.colors
                color = colors[0]
                self.object_color = gymapi.Vec3(color[0], color[1], color[2])

                self.gym.set_rigid_body_color(
                    env_ptr,
                    self.handles["object"],
                    0,  # index of rigid body to be set
                    gymapi.MESH_VISUAL,
                    self.object_color,
                )
            rand_friction = 0.8 + 0.4 * torch.rand((1,), device=self.device).item()
            shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, self.handles["object"]
            )
            for shape_prop in shape_props:
                shape_prop.friction *= rand_friction
                shape_prop.rolling_friction *= rand_friction
                shape_prop.torsion_friction *= rand_friction
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, self.handles["object"], shape_props
            )

            dof_props = self.gym.get_actor_dof_properties(
                env_ptr, self.handles["object"]
            )
            if len(dof_props["stiffness"]) > 0:
                rand_stiffness = 0.9 + 0.2 * torch.rand((1,), device=self.device).item()
                rand_damping = 0.9 + 0.2 * torch.rand((1,), device=self.device).item()
                dof_props["stiffness"] = dof_props["stiffness"] * rand_stiffness
                dof_props["damping"] = dof_props["damping"] * rand_damping
                self.gym.set_actor_dof_properties(
                    env_ptr, self.handles["object"], dof_props
                )

            random_scale = 0.98 + 0.04 * torch.rand(1, device=self.device).item()
            self.gym.set_actor_scale(env_ptr, self.handles["object"], random_scale)

            self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.dexs.append(self.handles["dexhand"])

        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        for i in range(1, 6):
            tip_name = self.fingertip_names[i - 1]
            self.handles[f"finger_tip{i}"] = self.gym.find_actor_rigid_body_handle(
                env_ptr, self.handles["dexhand"], tip_name
            )

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )

        self.ee_indices = [self.handles[f"finger_tip{i+1}"] for i in range(5)]
        logger.info(f"EE indices: {self.ee_indices}")

        # NOTE: root_state: num_envs * (1+1+num_type_of_objects) * 13
        self._dex_state = self._root_state[:, self.handles["dexhand"], :]

        # Set up contact force
        _net_contact_forces_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_force_tensor = gymtorch.wrap_tensor(
            _net_contact_forces_tensor
        ).view(self.num_envs, -1, 3)
        logger.info(
            f"Contact force tensor dimension: {self.contact_force_tensor.shape}"
        )

        self._update_tensor_views()

        # Initialize actions
        self._pos_control = torch.zeros(
            (self.num_envs, self.num_dex_dofs), dtype=torch.float, device=self.device
        )

        # Initialize indices
        self._global_indices = torch.arange(
            self.num_envs * (1 + 1 + 1),
            dtype=torch.int32,
            device=self.device,
        ).view(
            self.num_envs, -1
        )  # order: dex, table, objects

    def _step_physics(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self._refresh()

    def reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(
                start=0, end=self.num_envs, device=self.device, dtype=torch.long
            )

        if self.record_trajectory and len(env_ids) == 1:
            self.trajectory_manager.start_recording(f"episode_{self.episode_counter}")

        for env_idx in env_ids:
            obj_name = self.available_object_names[self.object_idx_buf[env_idx].item()]
            available_poses = object_pregrasp_poses.get(obj_name, [0])
            self.pregrasp_indices[env_idx] = available_poses[
                torch.randint(len(available_poses), (1,)).item()
            ]

        # Reset object states and hand states
        self._reset_object_state(env_ids)
        self._reset_dex_dof_and_state(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def _reset_object_state(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(
                start=0, end=self.num_envs, device=self.device, dtype=torch.long
            )

        num_resets = len(env_ids)
        object_state = torch.zeros(num_resets, 13, device=self.device)

        object_state[:, 0] = 0.0
        object_state[:, 1] = 0.0
        object_state[:, 2] = (
            self._table_surface_pos[2] + self.object_height_buf[env_ids] / 2
        )
        object_state[:, 6] = 1.0

        # Randomize object pose
        num_resets = len(env_ids)
        pos_random = (
            torch.rand((num_resets, 2), device=self.device) * 0.01 - 0.005
        )  # Random xy offset between -0.01 and 0.01
        rot_random = (
            torch.rand((num_resets,), device=self.device) * np.pi / 18 - np.pi / 36
        )  # Random rotation in ±5 degrees

        object_state[:, :2] += pos_random

        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        rot_quat = quat_from_angle_axis(rot_random, z_axis)
        object_state[:, 3:7] = quat_mul(rot_quat, object_state[:, 3:7])

        # Set object state and step physics
        self._set_object_state_and_step(env_ids, object_state)

    def _set_object_state_and_step(self, env_ids, object_state):
        self._root_state[env_ids, self.handles["object"], :] = object_state

        if len(env_ids) > 0:
            multi_env_ids_objects_int32 = self._global_indices[env_ids, 2:].flatten()
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_objects_int32),
                len(multi_env_ids_objects_int32),
            )
        self._step_physics()

    def get_object_specific_pose(self, env_idx):
        pose_idx = self.pregrasp_indices[env_idx].item()
        return self.base_joint_pose[pose_idx], self.finger_joint_qpos[pose_idx]

    def _reset_dex_dof_and_state(self, env_ids):
        num_resets = len(env_ids)

        hand_poses = torch.zeros((num_resets, 6), device=self.device)
        finger_qpos = torch.zeros((num_resets, 20), device=self.device)

        for i, (env_idx) in enumerate(zip(env_ids)):
            base_pose, finger_pose = self.get_object_specific_pose(env_idx)
            hand_poses[i] = base_pose
            finger_qpos[i] = finger_pose

        # Set hand pose and DOF state, then step physics
        self._set_hand_pose_and_dof_states_and_step(env_ids, hand_poses, finger_qpos)

    def _set_hand_pose_and_dof_states_and_step(self, env_ids, hand_poses, finger_qpos):
        self._root_state[env_ids, self.handles["dexhand"], :] = self._dex_state[
            env_ids, :
        ]

        dof_pos = torch.zeros((len(env_ids), self.num_dex_dofs), device=self.device)
        dof_pos[:, :6] = hand_poses
        dof_pos[:, 6:] = finger_qpos
        dof_pos = tensor_clamp(
            dof_pos,
            self.dex_dof_lower_limits.unsqueeze(0),
            self.dex_dof_upper_limits.unsqueeze(0),
        )

        self._q[env_ids, :] = dof_pos
        self._dq[env_ids, :] = torch.zeros_like(self._dq[env_ids])
        self._pos_control[env_ids, :] = dof_pos

        if len(env_ids) > 0:
            multi_env_ids_dex_int32 = self._global_indices[
                env_ids, self.handles["dexhand"]
            ].flatten()

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_dex_int32),
                len(multi_env_ids_dex_int32),
            )
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._pos_control),
                gymtorch.unwrap_tensor(multi_env_ids_dex_int32),
                len(multi_env_ids_dex_int32),
            )

            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(multi_env_ids_dex_int32),
                len(multi_env_ids_dex_int32),
            )

        self._step_physics()

    def _update_states(self):
        ee_contact_force = self.contact_force_tensor[:, self.ee_indices, :]
        ee_contact_force_norm = ee_contact_force.norm(dim=-1)
        ee_contact_state = ee_contact_force_norm > 0.01
        finger_to_object_dist = torch.stack(
            [
                torch.linalg.norm(
                    self._eef_state[:, i, :3] - self._object_state[:, :3], dim=-1
                )
                for i in range(5)
            ],
            dim=-1,
        )
        # if self.num_envs == 1:
        #     print(finger_to_object_dist)
        self.states.update(
            {
                # dex state
                "q": self._q[:, 6:],
                "dq": self._dq[:, 6:],
                "eef_pos": self._eef_state[:, :, :3].reshape(self.num_envs, -1),
                "eef_quat": self._eef_state[:, :, 3:7].reshape(self.num_envs, -1),
                "eef_vel": self._eef_state[:, :, 7:].reshape(self.num_envs, -1),
                # dex force
                "ee_contact_state": ee_contact_state,
                # object state
                "object_pos": self._object_state[:, :3],
                "object_quat": self._object_state[:, 3:7],
                # pregrasp pose (fixed in one episode)
                "finger_joint_qpos": self.finger_joint_qpos[self.pregrasp_indices],
                "base_joint_pose": self.base_joint_pose[self.pregrasp_indices],
                "finger_to_object_dist": finger_to_object_dist,
                # 5 physics steps per RL step
                "episode_time": self.progress_buf.unsqueeze(-1) * self.dt * 3,
            }
        )

    def _refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_tensor_views()
        self._update_states()

    def compute_observations(self):
        self._refresh()
        obs = [
            "finger_joint_qpos",
            "base_joint_pose",
            "q",
            "dq",
            "ee_contact_state",
        ]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        self.obs_buf += torch.randn_like(self.obs_buf) * 0.01

        if self.record_trajectory:
            save_dir = "trajectories"
            os.makedirs(save_dir, exist_ok=True)

            filename = os.path.join(
                save_dir, f"robot_state_{self.episode_counter}.json"
            )

            data_to_save = {
                "timestamp": time.time(),
                "q": self.states["q"].cpu().tolist(),
                "dq": self.states["dq"].cpu().tolist(),
                "ee_contact_state": self.states["ee_contact_state"].cpu().tolist(),
            }

            with open(filename, "a") as f:
                json.dump(data_to_save, f, indent=4)

    def compute_reward(self):
        self.rew_buf[:], reward_terms = compute_dex_reward(
            self.actions,
            self.states,
            self.reward_settings,
            self.object_height_buf,
            self.object_target_pose,
        )

        for term, value in reward_terms.items():
            mean_value = value.mean().item()
            self.extras[f"rewards/{term}"] = mean_value

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            self.total_successes = (
                self.total_successes + (self.successes * self.reset_buf).sum()
            )
            if self.total_resets > 0:
                print(
                    "Total success rate = {:.2f}".format(
                        self.total_successes / self.total_resets
                    )
                )

    def _apply_joint_constraints(self):
        for end_joint, start_joint in self.joint_constraints["coupled"]:
            end_idx = self.joint_name_to_index[end_joint]
            start_idx = self.joint_name_to_index[start_joint]
            self._pos_control[:, end_idx] = self._pos_control[:, start_idx]

        for follower, leader in self.joint_constraints["follow"]:
            follower_idx = self.joint_name_to_index[follower]
            leader_idx = self.joint_name_to_index[leader]
            self._pos_control[:, follower_idx] = self._pos_control[:, leader_idx]

        for joint, value in self.joint_constraints["fixed"]:
            joint_idx = self.joint_name_to_index[joint]
            self._pos_control[:, joint_idx] = value

        for scaled, source, scale_factor in self.joint_constraints["scaled"]:
            scaled_idx = self.joint_name_to_index[scaled]
            source_idx = self.joint_name_to_index[source]
            self._pos_control[:, scaled_idx] = (
                self._pos_control[:, source_idx] * scale_factor
            )

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        self.actions[:, 6:] += torch.randn_like(self.actions[:, 6:]) * 0.02

        env_times = self.dt * self.progress_buf * 3

        descending_mask = env_times < 2
        grasping_mask = (env_times >= 2) & (env_times < 4)
        ascending_mask = env_times >= 4

        qpos_increments = self.actions.clone()
        qpos_increments *= 0.08

        if descending_mask.any():
            descending_envs = torch.where(descending_mask)[0]
            descent_speeds = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            for env_idx in descending_envs:
                if self.pregrasp_indices[env_idx] == 0:
                    descent_speeds[env_idx] = 0.003
                elif self.pregrasp_indices[env_idx] == 1:
                    descent_speeds[env_idx] = 0.006
                else:
                    assert False, "Invalid pregrasp index"
            self._pos_control[descending_envs, 2] -= descent_speeds[descending_envs]
            qpos_increments = 0

        if grasping_mask.any():
            grasping_envs = torch.where(grasping_mask)[0]
            target_heights = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            for env_idx in grasping_envs:
                if self.pregrasp_indices[env_idx] == 0:
                    target_heights[env_idx] = -0.003 * 20
                elif self.pregrasp_indices[env_idx] == 1:
                    target_heights[env_idx] = -0.006 * 20
                else:
                    assert False, "Invalid pregrasp index"
            self._pos_control[grasping_envs, 2] = target_heights[grasping_envs]

        if ascending_mask.any():
            ascending_envs = torch.where(ascending_mask)[0]
            self._pos_control[ascending_envs, 2] += 0.005
            qpos_increments = 0

        self._pos_control[:, 6:] = self._q[:, 6:] + qpos_increments

        # Apply joint constraints and limits
        self._apply_joint_constraints()
        self._pos_control = torch.clamp(
            self._pos_control,
            self.dex_dof_lower_limits.unsqueeze(0),
            self.dex_dof_upper_limits.unsqueeze(0),
        )

        if self.record_trajectory:
            save_dir = "trajectories"
            os.makedirs(save_dir, exist_ok=True)

            filename = os.path.join(
                save_dir, f"robot_state_{self.episode_counter}.json"
            )

            data_to_save = {
                "timestamp": time.time(),
                "self._pos_control": self._pos_control[0].cpu().numpy().tolist(),
            }

            with open(filename, "a") as f:
                json.dump(data_to_save, f, indent=4)

        if self.record_trajectory:
            self.trajectory_manager.record_step(
                self._pos_control[0], base_joint_names + finger_joint_names
            )

        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self._pos_control)
        )

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.reset_idx(env_ids)

        self.compute_observations()
        self.check_termination()
        self.compute_reward()
        self.debug_visualize()

    def debug_visualize(self):
        """
        Draw thumb finger tip in red if in contact with object
        """
        env_index = 0
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            rigid_body_list = np.array(self.fingertip_handles)
            # draw contact force, FIXME: cpu only for isaacgym
            # self.gym.draw_env_rigid_contacts(
            #     self.viewer, self.envs[env_index], gymapi.Vec3(1, 0, 0), 5, False
            # )
            # set finger colors
            for i in range(5):
                if self.states["ee_contact_state"][env_index, i]:
                    self.gym.set_rigid_body_color(
                        self.envs[env_index],
                        self.dexs[env_index],
                        # index of rigid body to be set
                        rigid_body_list[i] - 1,
                        gymapi.MESH_VISUAL,
                        gymapi.Vec3(1, 0, 0),  # red
                    )
                else:
                    self.gym.set_rigid_body_color(
                        self.envs[env_index],
                        self.dexs[env_index],
                        rigid_body_list[i] - 1,
                        gymapi.MESH_VISUAL,
                        gymapi.Vec3(0, 0, 0),
                    )

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"].reshape(self.num_envs, -1, 3)
            eef_rot = self.states["eef_quat"].reshape(self.num_envs, -1, 4)

            self.gym.add_lines(
                self.viewer,
                self.envs[env_index],
                1,
                [0, 0, 0.01, 0.2, 0, 0.01],
                [0.85, 0.1, 0.1],
            )
            self.gym.add_lines(
                self.viewer,
                self.envs[env_index],
                1,
                [0, 0, 0.01, 0, 0.2, 0.01],
                [0.1, 0.85, 0.1],
            )
            self.gym.add_lines(
                self.viewer,
                self.envs[env_index],
                1,
                [0, 0, 0.01, 0, 0, 0.2],
                [0.1, 0.1, 0.85],
            )

    def check_termination(self):
        """
        Check if the episode should be terminated
        """
        object_height = (
            self._object_state[:, 2]
            - self._table_surface_pos[2]
            - self.object_height_buf / 2
        )

        timeout = self.progress_buf >= self.max_episode_length
        self.reset_buf |= timeout

        self.successes = torch.where(
            object_height > 0.05, torch.ones_like(self.successes), self.successes
        )

        if self.record_trajectory and timeout.any():
            self.trajectory_manager.stop_recording()
            self.trajectory_manager.save_trajectory()
            self.episode_counter += 1

    #####################################################################
    ### ======================keyboard functions=======================###
    #####################################################################

    def subscribe_keyboard_event(self):
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ENTER, "lock viewer to robot"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_P, "reset the environment"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Q, "exit camera follow mode"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_R, "record video"
        )

    def check_keyboard_event(self, action, value):
        if action == "lock viewer to robot" and value > 0:
            self.lock_viewer_to_robot = (self.lock_viewer_to_robot + 1) % 3
        elif action == "reset the environment" and value > 0:
            self.reset_idx(torch.tensor([self.follow_robot_index], device=self.device))
        elif action == "exit camera follow mode" and value > 0:
            self.lock_viewer_to_robot = 0

    def viewer_follow(self):
        """
        Callback called before rendering the scene
        Default behaviour: Follow robot
        """
        if self.lock_viewer_to_robot == 0:
            return
        distance = 0
        if self.lock_viewer_to_robot == 1:
            distance = torch.tensor(
                [-1.4, 0, 0.6], device=self.device, requires_grad=False
            )
        elif self.lock_viewer_to_robot == 2:
            distance = torch.tensor(
                [0, -1, 0.8], device=self.device, requires_grad=False
            )
        pos = self._dex_state[self.follow_robot_index, 0:3] + distance
        lookat = self._dex_state[self.follow_robot_index, 0:3]
        cam_pos = gymapi.Vec3(pos[0], pos[1], pos[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


#####################################################################
### =========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_dex_reward(
    actions: torch.Tensor,
    states: Dict[str, torch.Tensor],
    reward_settings: Dict[str, float],
    object_height_buf: torch.Tensor,
    object_target_pose: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # ------------------- Distance Reward -------------------
    finger_to_object_dist = states["finger_to_object_dist"]
    finger_to_object_dist = torch.clamp(finger_to_object_dist, 0.03, 0.8)

    finger_weights = torch.tensor([2, 1, 1, 1, 1], device=finger_to_object_dist.device)
    distance_reward = (
        reward_settings["w_distance_reward"]
        * (1.0 / (0.06 + finger_to_object_dist))
        * finger_weights
    )
    distance_reward = distance_reward.sum(dim=-1) / finger_weights.sum()

    # ------------------- Height Reward -------------------
    object_height = (
        states["object_pos"][:, 2]
        - reward_settings["table_height"]
        - object_height_buf / 2
    )
    height_reward = torch.where(
        (object_height > 0.004), 10.0 + object_height * 100, 0.0
    )

    rewards = distance_reward + height_reward
    reward_terms = {
        "distance_reward": distance_reward,
        "height_reward": height_reward,
    }

    return rewards, reward_terms
