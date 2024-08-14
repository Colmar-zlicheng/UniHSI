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


class UniHSI_PartNet_AUG(UniHSI_PartNet):

    def operate_mesh_with(mesh, obj):
        for r in obj['rotate']:
            R = mesh.get_rotation_matrix_from_xyz(r)
            mesh.rotate(R, center=(0, 0, 0))
        mesh.scale(obj['scale'], center=mesh.get_center())
        mesh_vertices_single = np.asarray(mesh.vertices).astype(np.float32())
        mesh.translate((0, 0, -mesh_vertices_single[:, 2].min()))
        mesh.translate(obj['transfer'])  #  not collision with init human
        # mesh.translate(obj['multi_obj_offset'])
        return mesh

    def _extra_load_meshinfo(self, pid, obj_id, obj):
        self.pid = pid
        self.otype = obj['name']
        self.n_count = obj['count']
        self.aug_count = obj['aug_count']
        self.count = 0
        self.obj_info = obj.copy()
        del self.obj_info['count']
        del self.obj_info['aug_count']
        self.obj_info["stand_point"] = self.obj_info["stand_point"][0]

    def get_save_dir(self):
        save_dir = os.path.join(self.save_root, self.pid, self.aug_count)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
