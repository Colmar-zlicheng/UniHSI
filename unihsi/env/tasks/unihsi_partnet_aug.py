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

    def operate_mesh_with(self, mesh, obj):
        for r in obj['rotate']:
            R = mesh.get_rotation_matrix_from_xyz(r)
            mesh.rotate(R, center=(0, 0, 0))

        # 定义各轴的缩放因子
        scale_factors = np.array(obj['scale'])
        # 计算网格的中心
        center = mesh.get_center()
        # 构建缩放矩阵
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] = np.diag(scale_factors)
        # 构建平移到原点的矩阵
        translate_to_origin = np.eye(4)
        translate_to_origin[:3, 3] = -center
        # 构建从原点平移回中心的矩阵
        translate_back = np.eye(4)
        translate_back[:3, 3] = center
        # 将这些变换矩阵组合在一起
        transformation = translate_back @ scale_matrix @ translate_to_origin
        # 应用变换矩阵对网格进行非均匀缩放
        mesh.transform(transformation)

        mesh_vertices_single = np.asarray(mesh.vertices).astype(np.float32())
        mesh.translate((0, 0, -mesh_vertices_single[:, 2].min()))
        mesh.translate(obj['transfer'])  #  not collision with init human
        # mesh.translate(obj['multi_obj_offset'])
        return mesh

    def operate_pcd_with(self, pcd, obj):
        # Apply rotations
        for r in obj['rotate']:
            R = pcd.get_rotation_matrix_from_xyz(r)
            pcd.rotate(R, center=(0, 0, 0))

        # Non-uniform scaling
        scale_factors = np.array(obj['scale'])
        # Get the center of the point cloud
        center = pcd.get_center()
        # Construct the scaling matrix
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] = np.diag(scale_factors)
        # Translate to origin, apply scaling, then translate back
        translate_to_origin = np.eye(4)
        translate_to_origin[:3, 3] = -center
        translate_back = np.eye(4)
        translate_back[:3, 3] = center
        # Combine transformations
        transformation = translate_back @ scale_matrix @ translate_to_origin
        # Apply the transformation to the point cloud
        pcd.transform(transformation)
        return pcd

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
        save_dir = os.path.join(self.save_root, self.pid, f"{self.aug_count}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        return save_dir
