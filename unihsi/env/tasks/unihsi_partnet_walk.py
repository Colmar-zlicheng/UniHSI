# import wandb
import os
import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
import open3d as o3d

import json
import shutil
import pickle

from .unihsi_partnet import UniHSI_PartNet
from .unihsi_partnet_bkp import UniHSI_PartNet_BKP
from .humanoid_amp_task import HumanoidAMPTask


class UniHSI_PartNet_WALK(UniHSI_PartNet):

    def _init_saving(self):
        self.try_num = 0
        self.max_try = 3
        self.fulfill_threshold = 0.1
        self.save_dict = {}
        self.humanoid_root_states_list = []
        self.dof_states_list = []
        self.rigid_body_states_list = []
        self.if_lie = False
        self.save_root = self.cfg["env"]["save_root"]
        assert self.save_root is not None

        self.start_point = self.plan_items[0]['obj']['start_point']

    def get_save_dir(self):
        save_dir = os.path.join(self.save_root, self.pid)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        return save_dir

    def _create_mesh_ground(self):
        self.plan_number = len(self.sceneplan)
        min_mesh_dict = self._load_mesh()
        pcd_list = self._load_pcd(min_mesh_dict)

        self._get_pcd_parts(pcd_list)
        # self.scene_idx = torch.tensor([[0]]).to(self.device)

        _x = np.arange(0, self.local_scale)
        _y = np.arange(0, self.local_scale)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        mesh_grid = np.stack([x, y, y], axis=-1)  # 3rd dim meaningless
        self.mesh_pos = torch.from_numpy(mesh_grid).to(self.device) * self.local_interval
        self.humanoid_in_mesh = torch.tensor(
            [self.local_interval * (self.local_scale - 1) / 4, self.local_interval * (self.local_scale - 1) / 2,
             0]).to(self.device)

    def _save_in_reset(self, reset, fulfill):
        if reset:
            if len(self.humanoid_root_states_list) > 0:
                states = {
                    "humanoid_root_states": torch.stack(self.humanoid_root_states_list[1:], dim=0),
                    "dof_states": torch.stack(self.dof_states_list[1:], dim=0),
                    "rigid_body_states": torch.stack(self.rigid_body_states_list[1:], dim=0)
                }
                self.save_dict["walk"] = states

            self.humanoid_root_states_list = []
            self.dof_states_list = []
            self.rigid_body_states_list = []

            if fulfill or self.try_num == self.max_try:

                self.save_dict["fulfill"] = fulfill.item()
                self.save_dict["pid"] = self.pid
                self.save_dict["object_type"] = self.otype

                save_dir = self.get_save_dir()

                with open(os.path.join(save_dir, f"demo_motion_{self.count}.pkl"), 'wb') as f:
                    pickle.dump(self.save_dict, f)

                if self.count == 0:
                    with open(os.path.join(save_dir, "meta.json"), 'w') as f:
                        json.dump(self.obj_info, f, indent=4)

                self.count += 1
                if self.count >= self.n_count:
                    if self.headless == False:
                        self.gym.destroy_viewer(self.viewer)
                    self.gym.destroy_sim(self.sim)
                    exit(0)
            else:
                self.try_num += 1
                self.save_dict = {}
                self.humanoid_root_states_list = []
                self.dof_states_list = []
                self.rigid_body_states_list = []

        else:
            self.humanoid_root_states_list = []
            self.dof_states_list = []
            self.rigid_body_states_list = []

    def _reset_actors(self, env_ids):
        if len(env_ids) > 0:
            success = (self.location_diff_buf[env_ids] < 0.1) & ~self.big_force[env_ids]

        self._reset_target(env_ids, success)

    def _reset_target(self, env_ids, success):

        contact_type_steps = self.contact_type_step[self.scene_for_env, self.step_mode]
        contact_valid_steps = self.contact_valid_step[self.scene_for_env, self.step_mode]
        fulfill = ((contact_valid_steps & \
                     (((contact_type_steps) & (self.joint_diff_buff < self.fulfill_threshold)) | (((~contact_type_steps) & (self.joint_diff_buff >= 0.05))))) \
                        | (~contact_valid_steps))[env_ids] & (success[:, None]) # need add contact direction
        fulfill = torch.all(fulfill, dim=-1)

        self.step_mode[env_ids[fulfill]] += 1

        max_step = self.step_mode[env_ids] == self.max_steps[self.scene_for_env][env_ids]

        reset = ~fulfill | max_step
        HumanoidAMPTask._reset_actors(self, env_ids[reset])

        self.still_buf[env_ids[reset | fulfill]] = 0

        rand_rot_theta = 2 * np.pi * torch.rand([self.num_envs], device=self.device)
        axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis)
        self._humanoid_root_states[env_ids[reset], 3:7] = rand_rot[env_ids[reset]]

        # dist_max = 4
        # dist_min = 2

        # rand_dist_y = (dist_max - dist_min) * torch.rand([self.num_envs], device=self.device) + dist_min
        # rand_dist_x = (dist_max - dist_min) * torch.rand([self.num_envs], device=self.device) + dist_min
        # x_sign = torch.from_numpy(np.random.choice((-1, 1), [self.num_envs])).to(self.device)
        # y_sign = torch.from_numpy(np.random.choice((-1, 1), [self.num_envs])).to(self.device)

        self._humanoid_root_states[env_ids[reset], 0] += self.x_offset[env_ids[reset]] + self.start_point[0]
        self._humanoid_root_states[env_ids[reset], 1] += self.y_offset[env_ids[reset]] + self.start_point[1]
        self.step_mode[env_ids[reset]] = 0

        stand_point_choice = torch.from_numpy(np.random.choice((0, 1, 2, 3), [self.num_envs])).to(self.device)
        self.stand_point_choice[env_ids[reset]] = stand_point_choice[env_ids[reset]]

        self.contact_type = self.contact_type_step[self.scene_for_env, self.step_mode]
        self.contact_valid = self.contact_valid_step[self.scene_for_env, self.step_mode]
        self.contact_direction = self.contact_direction_step[self.scene_for_env, self.step_mode]

        self.stand_point = self.scene_stand_point[range(len(self.step_mode)), self.step_mode, self.stand_point_choice]

        self.envs_obj_pcd_buffer[env_ids] = self.obj_pcd_buffer[self.scene_for_env[env_ids], self.step_mode[env_ids]]
        self.envs_obj_pcd_buffer[env_ids] = torch.einsum(
            "nmoe,neg->nmog", self.envs_obj_pcd_buffer[env_ids],
            self.obj_rotate_matrix[self.env_scene_idx_row, self.env_scene_idx_col][env_ids])
        self.envs_obj_pcd_buffer[env_ids, ..., 0] += self.x_offset[:, None, None][env_ids] + self.rand_dist_x[
            self.env_scene_idx_row, self.env_scene_idx_col][..., None, None][env_ids]
        self.envs_obj_pcd_buffer[env_ids, ..., 1] += self.y_offset[:, None, None][env_ids] + self.rand_dist_y[
            self.env_scene_idx_row, self.env_scene_idx_col][..., None, None][env_ids]
        self.envs_obj_pcd_buffer[env_ids, ..., 2] += self.rand_dist_z[self.env_scene_idx_row,
                                                                      self.env_scene_idx_col][..., None, None][env_ids]

        self._save_in_reset(reset, fulfill)
