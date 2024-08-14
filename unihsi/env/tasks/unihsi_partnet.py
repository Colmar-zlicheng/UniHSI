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

from .unihsi_partnet_bkp import UniHSI_PartNet_BKP, VaryPoint


class UniHSI_PartNet(UniHSI_PartNet_BKP):

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

    def _extra_load_meshinfo(self, pid, obj_id, obj):
        self.pid = pid
        self.otype = obj['name']
        self.n_count = obj['count']
        self.count = 0
        self.obj_info = obj.copy()
        del self.obj_info['count']
        self.obj_info["stand_point"] = self.obj_info["stand_point"][0]

    def get_save_dir(self):
        save_dir = os.path.join(self.save_root, self.pid)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        return save_dir

    def _save_in_reset(self, reset, fulfill):
        if reset:
            if len(self.humanoid_root_states_list) > 0:
                states = {
                    "humanoid_root_states": torch.stack(self.humanoid_root_states_list, dim=0),
                    "dof_states": torch.stack(self.dof_states_list, dim=0),
                    "rigid_body_states": torch.stack(self.rigid_body_states_list, dim=0)
                }
                if self.if_lie:
                    self.save_dict["lie"] = states
                else:
                    self.save_dict["sit"] = states
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
            if fulfill:
                states = {
                    "humanoid_root_states": torch.stack(self.humanoid_root_states_list[1:], dim=0),
                    "dof_states": torch.stack(self.dof_states_list[1:], dim=0),
                    "rigid_body_states": torch.stack(self.rigid_body_states_list[1:], dim=0)
                }
                if not self.if_lie:
                    self.save_dict["walk"] = states
                else:
                    self.save_dict["sit"] = states
                    self.if_lie = False

                self.humanoid_root_states_list = []
                self.dof_states_list = []
                self.rigid_body_states_list = []

                if self.otype == 'bed':
                    self.if_lie = True

            else:
                self.humanoid_root_states_list = []
                self.dof_states_list = []
                self.rigid_body_states_list = []

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        # print(self._humanoid_root_states.shape) # [1, 13]
        # print(type(self._humanoid_root_states))  # [1, 13]
        # print(self._dof_pos.shape) # [1, 28]
        # print(self._dof_vel.shape)  # [1, 28]
        # print(self._dof_state.shape)  # [28, 2]
        # print(self._rigid_body_state.shape) # [15, 13]
        self.humanoid_root_states_list.append((self._humanoid_root_states).squeeze(0).detach().cpu())
        self.dof_states_list.append((self._dof_state).detach().cpu())
        self.rigid_body_states_list.append((self._rigid_body_state).detach().cpu())
        return
