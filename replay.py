import os
import sys
import numpy as np
import time
from tqdm import tqdm
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import trimesh
import torch
import pickle
from scipy.spatial.transform import Rotation as R
# from pytorch3d.transforms import axis_angle_to_quaternion
import pytorch_kinematics as pk
import open3d as o3d


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (torch.sin(half_angles[~small_angles]) / angles[~small_angles])
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (0.5 - (angles[small_angles] * angles[small_angles]) / 48)
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    return quaternions


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {
        "name": "--controller",
        "type": str,
        "default": "ik",
        "help": "Controller to use for Franka. Options are {ik, osc}"
    },
    {
        "name": "--show_axis",
        "action": "store_true",
        "help": "Visualize DOF axis"
    },
    {
        "name": "--speed_scale",
        "type": float,
        "default": 1.0,
        "help": "Animation speed scale"
    },
    {
        "name": "--num_envs",
        "type": int,
        "default": 256,
        "help": "Number of environments to create"
    },
    {
        "name": "--seq",
        "type": str,
        "default": "data/UniHSI_retargeted_data/1284/amp_kinematic_motions/1284_demo_motion_0.npz",
        "help": "Number of environments to create"
    },
]
args = gymutil.parse_arguments(
    description="test",
    custom_parameters=custom_parameters,
)

# set torch device
# ipdb.set_trace()
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

######################## configure sim ########################
sim_params = gymapi.SimParams()
sim_fps = 10
sim_params.dt = dt = 1.0 / sim_fps

# set ground normal
gymutil.parse_sim_config({"gravity": [0.0, 0.0, -9.81], "up_axis": 1}, sim_params)  # 0 is y, 1 is z

if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")
###############################################################

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

show = True
save = True
save_fps = 10
if show:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")

asset_root = "./unihsi/data/assets"

# load amp asset
amp_asset_file = "mjcf/amp_humanoid.xml"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = False
amp_asset = gym.load_asset(sim, asset_root, amp_asset_file, asset_options)
amp_dof_names = gym.get_asset_dof_names(amp_asset)

amp_dof_props = gym.get_asset_dof_properties(amp_asset)
amp_num_dofs = gym.get_asset_dof_count(amp_asset)
amp_dof_states = np.zeros(amp_num_dofs, dtype=gymapi.DofState.dtype)
amp_dof_types = [gym.get_asset_dof_type(amp_asset, i) for i in range(amp_num_dofs)]
amp_dof_positions = amp_dof_states['pos']
amp_lower_limits = amp_dof_props["lower"]
amp_upper_limits = amp_dof_props["upper"]
amp_ranges = amp_upper_limits - amp_lower_limits
amp_mids = 0.3 * (amp_upper_limits + amp_lower_limits)
amp_stiffnesses = amp_dof_props['stiffness']
amp_dampings = amp_dof_props['damping']
amp_armatures = amp_dof_props['armature']
amp_has_limits = amp_dof_props['hasLimits']
amp_dof_props['hasLimits'] = np.array([True] * amp_num_dofs)

num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

envs = []
actor_handles = []
joint_handles = {}

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# load amp motion data
with open(args.seq, 'rb') as f:
    motion_data_unihsi = pickle.load(f)

walk = motion_data_unihsi['sit']
sit = motion_data_unihsi['sit']
humanoid_root_states = np.concatenate([walk['humanoid_root_states'], sit['humanoid_root_states']], axis=0)  # [N, 13]
dof_states = np.concatenate([walk['dof_states'], sit['dof_states']], axis=0)  # [N, 28, 2]
rigid_body_states = np.concatenate([walk['rigid_body_states'], sit['rigid_body_states']], axis=0)  # [N, 15, 13]
if motion_data_unihsi['object_type'] == 'bed':
    lie = motion_data_unihsi['lie']
    humanoid_root_states = np.concatenate([humanoid_root_states, lie['humanoid_root_states']], axis=0)
    dof_states = np.concatenate([dof_states, lie['dof_states']], axis=0)
    rigid_body_states = np.concatenate([rigid_body_states, lie['rigid_body_states']], axis=0)

initial_pose_ori = [0, 0, 0, 1]  # (x, y, z, w)

# set object params
object_asset_options = gymapi.AssetOptions()
object_asset_options.fix_base_link = True
object_asset_options.use_mesh_materials = True
object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
object_asset_options.override_inertia = True
object_asset_options.override_com = True
object_asset_options.vhacd_enabled = False
object_pose = gymapi.Transform()
object_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
# object_asset_root = "/home/liuyun/Humanoid_IL_Benchmark/humanplus/HST/legged_gym/legged_gym/envs/assets"

# load objects
obj_file = os.path.join(os.path.dirname(os.path.dirname(args.seq)), "scene_mesh.obj")
object_names = [obj_file]
object_list = []
for object_name in object_names:
    object_collision_mesh_o3d = o3d.io.read_triangle_mesh(object_name)  # 用trimesh读则需要区分TriangleMesh和Scene
    object_collision_mesh = trimesh.Trimesh(vertices=np.float32(object_collision_mesh_o3d.vertices),
                                            faces=np.int32(object_collision_mesh_o3d.triangles))
    object_vertices, object_faces = np.float32(object_collision_mesh.vertices).copy(), np.uint32(
        object_collision_mesh.faces).copy()
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = object_vertices.shape[0]
    tm_params.nb_triangles = object_faces.shape[0]
    # tm_params.transform.r = gymapi.Quat.from_euler_zyx(np.pi / 2, 0, -np.pi / 2)
    object_list.append({"vertices": object_vertices, "faces": object_faces, "tm_params": tm_params})

env_object_ids = np.random.randint(0, len(object_list), num_envs)
assert num_envs == 1
env_origins = torch.zeros(num_envs, 3, device=device, requires_grad=False)
# object info
object_center_position = torch.zeros(num_envs, 3, dtype=torch.float, device=device, requires_grad=False)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add object
    object_info = object_list[env_object_ids[i]]
    env_ori = env_origins[i].detach().cpu().numpy()
    object_info["tm_params"].transform.p.x = object_center_position[i, 0] = env_ori[0] + 0.0
    object_info["tm_params"].transform.p.y = object_center_position[i, 1] = env_ori[1] + 0.0
    object_info["tm_params"].transform.p.z = object_center_position[i, 2] = 0.0
    gym.add_triangle_mesh(sim, object_info["vertices"].flatten(), object_info["faces"].flatten(),
                          object_info["tm_params"])

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.05)
    pose.r = gymapi.Quat(*initial_pose_ori)

    actor_handle = gym.create_actor(env, amp_asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, amp_dof_states, gymapi.STATE_ALL)

# position the camera
if show:
    # right view
    # cam_pos = gymapi.Vec3(3, 2.0, 0)
    # cam_target = gymapi.Vec3(-3, 0, 0)
    cam_pos = gymapi.Vec3(0, -3, 2.0)
    cam_target = gymapi.Vec3(0, 3, 0)
    # front view
    # cam_pos = gymapi.Vec3(0, 2.0, -2)
    # cam_target = gymapi.Vec3(0, 0, 2)
    gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)

gym.prepare_sim(sim)
for i in tqdm(range(motion_data.shape[0])):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # set global pose
    t = torch.from_numpy(motion_global_translations[i])
    q_wxyz = axis_angle_to_quaternion(torch.from_numpy(motion_global_rotations[i]))  # (w, x, y, z)
    q_xyzw = torch.tensor([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).to(dtype=torch.float32)  # (x, y, z, w)
    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(actor_root_state)
    root_states[:, :3] = t
    root_states[:, 3:7] = q_xyzw
    root_reset_actors_indices = torch.tensor([gym.get_actor_index(envs[0], actor_handles[0],
                                                                  gymapi.DOMAIN_SIM)]).to(dtype=torch.int32)
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states),
                                            gymtorch.unwrap_tensor(root_reset_actors_indices), 1)

    # set joint angles
    for j in range(motion_data.shape[1]):
        amp_dof_positions[j] = motion_data[i, j]
    gym.set_actor_dof_states(envs[0], actor_handles[0], amp_dof_states, gymapi.STATE_POS)

    # humanoid pose
    print("pelvis global pose =", pose.p, pose.r)
    print("19DoF local poses =", amp_dof_positions)
    print("19DoF joint names =", amp_dof_names)
    joint_pose = gym.get_actor_joint_transforms(
        envs[0], actor_handles[0]
    )  # len = 24, item: (3D translation vector P, 4D quaternion Q (x, y, z, w)), in world space, 含义是：沿这个关节的转动 = 沿世界系的P处的frame Q的x轴正向的转动
    joint_names = gym.get_actor_joint_names(
        envs[0], actor_handles[0]
    )  # 'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint', 'torso_joint', 'd435_left_imager_joint', 'd435_rgb_module_joint', 'imu_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'logo_joint', 'mid360_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
    print("24D joint global poses =", joint_pose, len(joint_pose), joint_pose[20][0])
    print("24D joint names =", joint_names)

    # # differentiable FK
    # chain = pk.build_serial_chain_from_urdf(open("../assets/amp_description/urdf/amp.urdf", "rb").read(), "left_ankle_link")
    # print(chain)
    # print(chain.get_joint_parameter_names())
    # th = amp_dof_positions.copy()
    # ret = chain.forward_kinematics(th, end_only=False)
    # print(ret)
    # assert False

    if show:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.clear_lines(viewer)
    gym.sync_frame_time(sim)

    if show and gym.query_viewer_has_closed(viewer):
        break

if show:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
