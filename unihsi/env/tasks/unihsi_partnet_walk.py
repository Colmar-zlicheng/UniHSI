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


class UniHSI_PartNet_WALK(UniHSI_PartNet):

    def get_save_dir(self):
        save_dir = os.path.join(self.save_root, self.pid)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        return save_dir

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

