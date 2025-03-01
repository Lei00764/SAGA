import os
from pathlib import Path

from isaacgym import gymapi

object_heights = {
    "000": 0.2134,
    "001": 0.1760,
    "002": 0.1018,
    "003": 0.1913,
    "004": 0.0835,
    "005": 0.0367,
    "006": 0.0550,
    "007": 0.0813,
    "008": 0.0573,
    "009": 0.0157,
    "010": 0.1019,
    "011": 0.0457,
    "012": 0.0719,
    "013": 0.0530,
    "014": 0.0586,
    "015": 0.0657,
    "016": 0.0713,
    "017": 0.0531,
    "018": 0.2145,
    "019": 0.2162,
    "020": 0.2180,
    "021": 0.0556,
    "022": 0.0750,
    "023": 0.0668,
    "024": 0.1465,
    "025": 0.0592,
    "026": 0.0663,
    "027": 0.0225,
    "028": 0.1177,
    "029": 0.0303,
    "030": 0.0303,
    "031": 0.0376,
    "032": 0.0774,
    "033": 0.1528,
    "034": 0.0315,
    "035": 0.0438,
    "036": 0.0353,
    "037": 0.0405,
    "038": 0.0375,
    "039": 0.0392,
    "040": 0.0540,
    "041": 0.0374,
    "042": 0.0487,
    "043": 0.0238,
    "044": 0.0683,
    "045": 0.0594,
    "046": 0.0698,
    "047": 0.1066,
    "048": 0.0537,
    "049": 0.0506,
    "050": 0.0569,
    "051": 0.0602,
    "052": 0.0612,
    "053": 0.0339,
    "054": 0.0326,
    "055": 0.0356,
    "056": 0.0804,
    "057": 0.0752,
    "058": 0.0587,
    "059": 0.0320,
    "060": 0.0974,
    "061": 0.0509,
    "062": 0.0489,
    "063": 0.0494,
    "064": 0.0483,
    "065": 0.0443,
    "066": 0.0437,
    "067": 0.0415,
    "068": 0.0422,
    "069": 0.0472,
    "070": 0.1153,
    "071": 0.0782,
    "072": 0.0623,
    "073": 0.0713,
    "074": 0.2690,
    "075": 0.1190,
    "076": 0.0995,
    "077": 0.1034,
    "078": 0.0980,
    "079": 0.1184,
    "080": 0.0988,
    "081": 0.1211,
    "082": 0.0741,
    "083": 0.0971,
    "084": 0.1139,
    "085": 0.1087,
    "086": 0.1004,
    "087": 0.1083,
}


class GraspNetLoader:
    def __init__(self, gym, sim):
        self.gym = gym
        self.sim = sim
        self.dataset_root = self.get_dataset_root_dir()

    def get_dataset_root_dir(self):
        current_dir = Path(__file__).parent
        dataset_dir = os.path.join(current_dir.parent.parent, "assets", "GraspNet")
        return dataset_dir

    def get_available_object_names(self):
        """
        Get available object names in the GraspNet dataset

        Returns:
            list: List of object names
        """
        # return [
        #     name for name, height in object_heights.items() if 0.08 <= height <= 0.1
        # ]
        return [
            # "001",
            "002",
            # "003",
            # "004",
            # "005", (unseen)
            # "006", (unseen)
            "007",
            # "008",
            # "009",
            # "010",
            # "011",
            # "012", (unseen)
            # "013",
            # "014",
            # "015", (unseen)
            "016",
            # "017",
            # "018",
            # "019",
            # "020",
            # "021",
            "022",
            # "023",
            # "024",
            "025",
            # "026",
            # "027",
            # "028",
            # "029",
            # "030",
            # "031",
            # "032",
            # "033",
            # "034",
            "035",
            "036",
            "037",
            "038",
            # "039",
            # "040",
            # "041",
            "042",
            # "043",
            "044",
            # "045", (unseen)
            # "046",
            # "047",
            # "048",
            # "049",
            # "050",
            # "051",
            # "052",
            # "053",
            # "054",
            # "055",
            # "056",
            "057",
            # "058", (unseen)
            # "059",
            # "060",
            "061",
            "062",
            "063",
            "064",
            "065",
            "066",
            # "067",
            "068",
            # "069",
            # "070",
            "071",
            # "072",
            # "073",
            # "074",
            # "075",
            # "076",
            "077",
            # "078", (unseen)
            # "079",
            # "080",
            # "081",
            # "082",
            # "083",
            # "084",
            # "085",
            # "086",
            "087",
        ]

    def load_dataset_object(self, object_name):
        """
        Load GraspNet object into IsaacGym environment

        Args:
            object_name (str): Name of GraspNet object (without prefix numbers)
            density (float): Density of the object

        Returns:
            object_asset: Asset handle for the loaded object
        """
        urdf_file = f"urdfs/{object_name}.urdf"

        # Create basic asset options
        asset_options = gymapi.AssetOptions()
        asset_options.density = 500
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 300000
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        dataset_root = Path(self.dataset_root)
        if not (dataset_root / urdf_file).exists():
            raise FileNotFoundError(f"URDF file not found: {dataset_root / urdf_file}")

        object_asset = self.gym.load_asset(
            self.sim, str(self.dataset_root), str(urdf_file), asset_options
        )

        object_pose = gymapi.Transform()
        object_height = self.get_object_height(object_name)

        return object_asset, object_pose, object_height

    def get_object_height(self, object_name):
        return object_heights[object_name]


def main():
    gym = gymapi.acquire_gym()

    # Create sim
    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = os.cpu_count()
    sim_params.physx.use_gpu = True

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    # Set camera pose
    cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Create ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # Load object
    loader = GraspNetLoader(gym, sim)
    object_name = "000"
    object_asset, object_pose, object_height = loader.load_dataset_object(object_name)

    # Create environment
    env_spacing = 2.0
    lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    num_envs = 1

    # Create env
    env = gym.create_env(sim, lower, upper, 1)

    # Set initial pose for object
    object_pose.p = gymapi.Vec3(0.0, 0.0, object_height)
    object_pose.r = gymapi.Quat(0, 0, 0, 1)

    # Create actor
    actor_handle = gym.create_actor(env, object_asset, object_pose, "object", 0, 1)

    # Simulation loop
    while not gym.query_viewer_has_closed(viewer):
        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time
        gym.sync_frame_time(sim)

    # Cleanup
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
