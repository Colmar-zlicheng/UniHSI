import os
import json
import argparse
import random
import numpy as np
import open3d as o3d
from tqdm import tqdm

PARTNET_CHAIR_BED_PATH = "docs/partnet_chair_bed.json"
PARTNET_ADD_REGULAR_BED_PATH = "docs/partnet_add_regular_bed.json"


class Chair:

    def __init__(self, obj_id):
        self.name = "chair"
        self.obj_id = obj_id
        self.scale = 1.2
        trans_x = 1.2
        trans_y = -0.8
        self.transfer = [trans_x, trans_y, 0]
        self.rotate = [[1.5707963267948966, 0, 0], [0, 0, -1.5707963267948966]]
        self.stand_point = [trans_x + 0.7, trans_y, 0.86]

        self.surface_id, surface_type = self.get_parnet_chair_seat_surface_id(obj_id)
        self.contact_pairs = [[["chair000", "none", "none", "none", "none"]],
                              [["chair000", f"{surface_type}{self.surface_id}", "pelvis", "contact", "up"]]]

    @staticmethod
    def get_parnet_chair_seat_surface_id(obj_id, partnet_root="data/partnet_add"):
        with open(os.path.join(partnet_root, obj_id, "result.json"), 'r') as f:
            result = json.load(f)
        children = result[0]["children"]
        surface_id = -1
        surface_type = None
        for child in children:
            if child["text"] == "Chair Seat":
                for c in child["children"]:
                    if c["text"] == "Seat Surface":
                        surface_id = c["children"][0]["id"]
                        surface_type = c["children"][0]["name"]
        return surface_id, surface_type


class Bed:

    def __init__(self, obj_id):
        self.name = "bed"
        self.obj_id = obj_id
        self.scale = 3.0
        self.transfer = [3, -2, 0]
        self.rotate = [[1.5707963267948966, 0, 0], [0, 0, -1.5707963267948966]]
        self.stand_point = self.get_bed_stand_point(obj_id)

        self.surface_id, surface_type, self.pillow_id = self.get_parnet_bed_mattress_id(obj_id)
        if self.pillow_id == -1:
            head_contact = f"{surface_type}{self.surface_id}"
        else:
            head_contact = f"pillow{self.pillow_id}"

        self.contact_pairs = [[["bed000", "none", "none", "none", "none"]],
                              [["bed000", f"{surface_type}{self.surface_id}", "pelvis", "contact", "up"],
                               ["bed000", "floor", "left_foot", "contact", "none"],
                               ["bed000", "floor", "right_foot", "contact", "none"],
                               ["bed000", f"{surface_type}{self.surface_id}", "head", "not contact", "none"]],
                              [["bed000", f"{surface_type}{self.surface_id}", "pelvis", "contact", "up"],
                               ["bed000", f"{surface_type}{self.surface_id}", "left_foot", "contact", "none"],
                               ["bed000", f"{surface_type}{self.surface_id}", "right_foot", "contact", "none"],
                               ["bed000", head_contact, "head", "contact", "none"]]]

    def get_bed_stand_point(self, obj_id, partnet_root="data/partnet_add"):
        stand_point = [None, None, 0.86]

        mesh = o3d.io.read_triangle_mesh(os.path.join(partnet_root, obj_id, 'models/model_normalized.obj'))
        for r in self.rotate:
            R = mesh.get_rotation_matrix_from_xyz(r)
            mesh.rotate(R, center=(0, 0, 0))
        mesh.scale(self.scale, center=mesh.get_center())
        mesh_vertices_single = np.asarray(mesh.vertices).astype(np.float32())
        mesh.translate((0, 0, -mesh_vertices_single[:, 2].min()))
        mesh.translate(self.transfer)
        mesh_vertices_single = np.asarray(mesh.vertices).astype(np.float32())

        # stand_point[0] = np.mean(mesh_vertices_single, 0)[0] + (0.1 + 0.1 * np.random.rand(1)[0]) * (
        # np.max(mesh_vertices_single, 0)[0] - np.min(mesh_vertices_single, 0)[0])
        stand_point[0] = (np.max(mesh_vertices_single, 0)[0].item() + np.min(mesh_vertices_single, 0)[0]) / 2
        stand_point[1] = np.max(mesh_vertices_single, 0)[1] + 0.3

        return stand_point

    @staticmethod
    def get_parnet_bed_mattress_id(obj_id, partnet_root="data/partnet_add"):
        with open(os.path.join(partnet_root, obj_id, "result.json"), 'r') as f:
            result = json.load(f)

        children = result[0]["children"][0]
        assert children["text"] == "Regular bed"
        children = children["children"][0]
        assert children["text"] == "Bed Unit"
        children = children["children"]

        surface_id = -1
        pillow_id = -1
        mattress_id = -1
        blanket_id = -1

        for child in children:
            if child["text"] == "Bed sleep area":
                for c in child["children"]:
                    if c["text"] == "Pillow" and pillow_id == -1:
                        pillow_id = c["id"]
                    if c["text"] == "Mattress":
                        mattress_id = c["id"]
                    if c["text"] == "Blanket":
                        blanket_id = c["id"]

        if blanket_id == -1:
            surface_id = mattress_id
            surface_type = "mattress"
        else:
            if mattress_id == -1:
                surface_id = blanket_id
                surface_type = "blanket"
            else:
                surface_id = mattress_id
                surface_type = "mattress"

        return surface_id, surface_type, pillow_id


def parse_dict(obj_class):
    return {
        "obj": {
            "000": {
                "id": obj_class.obj_id,
                "rotate": obj_class.rotate,
                "scale": obj_class.scale,
                "transfer": obj_class.transfer,
                "stand_point": [obj_class.stand_point] * 4,
                "name": obj_class.name,
            },
        },
        "contact_pairs": obj_class.contact_pairs,
    }


def get_chair_info(args):
    with open(PARTNET_CHAIR_BED_PATH, 'r') as f:
        partnet_chair_bed = json.load(f)
    parnet_unihsi = os.listdir("data/partnet")
    partnet_chair = []
    for chair_bed in partnet_chair_bed:
        if chair_bed[1] == "chair":
            if chair_bed[0] not in parnet_unihsi:
                partnet_chair.append(chair_bed[0])
    if args.num != -1:
        partnet_chair = random.sample(partnet_chair, args.num)

    save_dict = {}

    key_id = 0
    for i in tqdm(range(len(partnet_chair))):
        chair_class = Chair(partnet_chair[i])
        if chair_class.surface_id == -1:
            print("failed chair: ", partnet_chair[i])
            continue
        save_dict[str(key_id).rjust(4, '0')] = parse_dict(chair_class)
        key_id += 1

    with open(args.save_path, 'w') as f:
        json.dump(save_dict, f, indent=4)


def get_bed_info(args):
    with open(PARTNET_ADD_REGULAR_BED_PATH, 'r') as f:
        partnet_bed = json.load(f)
    if args.num != -1:
        partnet_bed = random.sample(partnet_bed, args.num)

    save_dict = {}

    key_id = 0
    for i in tqdm(range(len(partnet_bed))):
        bed_class = Bed(partnet_bed[i])
        if bed_class.surface_id == -1:
            print("failed bed: ", partnet_bed[i])
            continue
        save_dict[str(key_id).rjust(4, '0')] = parse_dict(bed_class)
        key_id += 1

    with open(args.save_path, 'w') as f:
        json.dump(save_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', '--object_type', default='chair', choices=['chair', 'bed'])
    parser.add_argument('-n', '--num', type=int, required=True)
    parser.add_argument('-s', '--save_path', type=str, required=True)
    arg = parser.parse_args()

    if arg.object_type == "chair":
        get_chair_info(arg)
    elif arg.object_type == "bed":
        get_bed_info(arg)
    else:
        raise ValueError()
