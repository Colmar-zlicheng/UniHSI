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
import json
import pickle
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def get_bbox(pts):
    x_min, y_min, z_min = list(pts.min(axis=0))
    x_max, y_max, z_max = list(pts.max(axis=0))
    return [x_min, y_min, z_min, x_max, y_max, z_max]

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
        {
        "name": "--save",
        "action": "store_true",
        "help": "save video"
    },
    {
        "name": "--save_path",
        "type": str,
        "default": "tmp/viz_retarget.mp4",
        "help": "save path"
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

show = True if not args.save else False
# save = True
save_fps = 20
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

walk = motion_data_unihsi['walk']
if motion_data_unihsi['object_type'] == 'walk':
    humanoid_root_states = walk['humanoid_root_states'].numpy()
    dof_states = walk['dof_states'].numpy()
    rigid_body_states = walk['rigid_body_states'].numpy()
else:
    sit = motion_data_unihsi['sit']
    humanoid_root_states = np.concatenate([walk['humanoid_root_states'], sit['humanoid_root_states']],
                                          axis=0)  # [N, 13]
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

# load objects
if motion_data_unihsi['object_type'] == 'walk':
    object_meta_info = json.load(open(os.path.join(os.path.dirname(args.seq), "meta.json"), "r"))
    obj_file = os.path.join("data/CORE4D_retargeted_data_touchpoint", object_meta_info['seq_id'], "scene_mesh.obj")
    obj_mesh = o3d.io.read_triangle_mesh(obj_file)

else:
    object_meta_info = json.load(open(os.path.join(os.path.dirname(args.seq), "meta.json"), "r"))
    partnet_id = os.path.basename(os.path.dirname(os.path.dirname(args.seq)))

    if partnet_id in os.listdir('data/partnet'):
        obj_file = os.path.join('data/partnet', partnet_id)
    else:
        obj_file = os.path.join('data/partnet_add', partnet_id)
    obj_file = os.path.join(obj_file, 'models/model_normalized.obj')
    obj_mesh = o3d.io.read_triangle_mesh(obj_file)  # 用trimesh读则需要区分TriangleMesh和Scene
    for r in object_meta_info["rotate"]:
        R = obj_mesh.get_rotation_matrix_from_xyz(r)
        obj_mesh.rotate(R, center=(0, 0, 0))

    # rescale
    scale_factors = object_meta_info["scale"]
    if (isinstance(scale_factors, int)) or (isinstance(scale_factors, float)):
        print("[warning] the scale is a scalar, not a list !!!")
        scale_factors = [scale_factors, scale_factors, scale_factors]
    T_scale = np.eye(4)
    T_scale[:3, :3] = np.diag(scale_factors)
    T_to_origin = np.eye(4)
    T_to_origin[:3, 3] = -obj_mesh.get_center()
    T_back = -T_to_origin
    T = T_back @ T_scale @ T_to_origin
    obj_mesh.transform(T)

    obj_mesh_v = np.float32(obj_mesh.vertices)
    obj_mesh.translate((0, 0, -obj_mesh_v[:, 2].min()))
    obj_mesh.translate(object_meta_info["transfer"])

object_collision_mesh = trimesh.Trimesh(vertices=np.float32(obj_mesh.vertices), faces=np.int32(obj_mesh.triangles))
object_vertices, object_faces = np.float32(object_collision_mesh.vertices).copy(), np.uint32(
    object_collision_mesh.faces).copy()
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = object_vertices.shape[0]
tm_params.nb_triangles = object_faces.shape[0]
# tm_params.transform.r = gymapi.Quat.from_euler_zyx(np.pi / 2, 0, -np.pi / 2)
object_list = [{"vertices": object_vertices, "faces": object_faces, "tm_params": tm_params}]

bbox = get_bbox(object_vertices)
cx, cy = (bbox[0] + bbox[3]) / 2, (bbox[1] + bbox[4]) / 2

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
    # if not motion_data_unihsi['object_type'] == 'walk':
    object_info = object_list[env_object_ids[i]]
    env_ori = env_origins[i].detach().cpu().numpy()
    object_info["tm_params"].transform.p.x = object_center_position[i, 0] = env_ori[0] - cx
    object_info["tm_params"].transform.p.y = object_center_position[i, 1] = env_ori[1] - cy
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
    cam_pos = gymapi.Vec3(0, -3, 2.0)
    cam_target = gymapi.Vec3(0, 3, 0)
    # front view
    gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)

if args.save:
    camera_props = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.width = 1920
    camera_props.height = 1080

    camera_handle = gym.create_camera_sensor(envs[0], camera_props)
    camera_transform = gymapi.Transform()
    camera_transform.p = gymapi.Vec3(0, -3, 2.0)
    camera_transform.r = gymapi.Quat.from_euler_zyx(0.0, np.radians(30), np.radians(90))
    gym.set_camera_transform(camera_handle, envs[0], camera_transform)

    images = []

# humanoid_root_states # [N, 13]
# dof_states # [N, 28, 2]
# rigid_body_states  # [N, 15, 13]
amp_dof_states = np.zeros(28, dtype=gymapi.DofState.dtype)

gym.prepare_sim(sim)
for i in tqdm(range(humanoid_root_states.shape[0])):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # set global pose
    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(actor_root_state)
    root_states = torch.from_numpy(humanoid_root_states[i]).unsqueeze(0)
    root_states[:,0] -= cx
    root_states[:,1] -= cy
    root_reset_actors_indices = torch.tensor([gym.get_actor_index(envs[0], actor_handles[0],
                                                                  gymapi.DOMAIN_SIM)]).to(dtype=torch.int32)
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states),
                                            gymtorch.unwrap_tensor(root_reset_actors_indices), 1)

    # set joint angles
    amp_dof_states["pos"] = dof_states[i, :, 0]
    gym.set_actor_dof_states(envs[0], actor_handles[0], amp_dof_states, gymapi.STATE_POS)

    if show:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.clear_lines(viewer)

    if args.save:
        if not show:
            gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)
        camera_rgba_tensor = gym.get_camera_image_gpu_tensor(sim, envs[0], camera_handle, gymapi.IMAGE_COLOR)
        torch_camera_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
        images.append(torch_camera_rgba_tensor.clone())
        gym.end_access_image_tensors(sim)

    gym.sync_frame_time(sim)

    if show and gym.query_viewer_has_closed(viewer):
        break

if show:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

if args.save:
    import torchvision

    frames = torch.stack(images).detach().cpu()
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))
    torchvision.io.write_video(args.save_path, frames[..., :3], fps=save_fps, video_codec="libx264")