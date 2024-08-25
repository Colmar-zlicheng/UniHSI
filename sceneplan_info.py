import os
import json
import argparse
import random
import numpy as np
from tqdm import tqdm

PARTNET_CHAIR_BED_PATH = "docs/partnet_chair_bed.json"


class chair:
    name = "chair"
    obj_id = "{obj_id}"
    scale = None
    transfer = [None, -2, 0]
    rotate = [[1.5707963267948966, 0, 0], [0, 0, -1.5707963267948966]]
    stand_point = [None, None, 0.86]
    contact_pairs = [[["chair000", "none", "none", "none", "none"]],
                     [["chair000", "seat_soft_surface{surface_id}", "pelvis", "contact", "up"]]]


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


def get_parnet_chair_seat_surfave_id(obj_id):
    pass


def get_chair_info(args):
    with open(PARTNET_CHAIR_BED_PATH, 'r') as f:
        partnet_chair_bed = json.load(f)
    partnet_chair = []
    for chair_bed in partnet_chair_bed:
        if chair_bed[1] == "chair":
            partnet_chair.append(chair_bed[0])
    if arg.num != -1:
        partnet_chair = random.sample(partnet_chair, args.num)

    save_dict = {}
    for i in tqdm(range(len(partnet_chair))):
        chair_class = chair
        obj_id = partnet_chair[i]
        chair_class.obj_id = chair_class.obj_id.format(obj_id=obj_id)
        surface_id = get_parnet_chair_seat_surfave_id(obj_id)
        chair_class.contact_pairs[1][0][1] = chair_class.contact_pairs[1][0][1].format(surface_id=surface_id)

        save_dict[str(i).rjust(4, '0')] = parse_dict(chair_class)

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
