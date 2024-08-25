import os
import json
import argparse
import random
import numpy as np
from tqdm import tqdm

PARTNET_CHAIR_BED_PATH = "docs/partnet_chair_bed.json"


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

        self.surface_id = self.get_parnet_chair_seat_surface_id(obj_id)
        self.contact_pairs = [[["chair000", "none", "none", "none", "none"]],
                              [["chair000", f"seat_soft_surface{self.surface_id}", "pelvis", "contact", "up"]]]

    @staticmethod
    def get_parnet_chair_seat_surface_id(obj_id, partnet_root="data/partnet_add"):
        with open(os.path.join(partnet_root, obj_id, "result.json"), 'r') as f:
            result = json.load(f)
        children = result[0]["children"]
        surface_id = -1
        for child in children:
            if child["text"] == "Chair Seat":
                surface_id = child["children"][0]["id"]
        return surface_id


def parse_dict(obj_class):
    return {
        "obj": {
            "000": {
                "id": obj_class.obj_id,
                "rotate": obj_class.rotate,
                "scale": obj_class.scale,
                "transfer": obj_class.transfer,
                "stand_point": [obj_class.stand_point] * 4,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', '--object_type', default='chair', choices=['chair'])
    parser.add_argument('-n', '--num', type=int, required=True)
    parser.add_argument('-s', '--save_path', type=str, required=True)
    arg = parser.parse_args()

    if arg.object_type == "chair":
        get_chair_info(arg)
    else:
        raise ValueError()
