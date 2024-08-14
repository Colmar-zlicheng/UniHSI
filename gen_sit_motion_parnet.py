import os
import argparse
import json
import numpy as np


def trans_to_center(obj_value):
    transfer = np.array(obj_value["obj"]["000"]["transfer"])
    stand_point = np.array(obj_value["obj"]["000"]["stand_point"])
    stand_point = stand_point - transfer[None, :]
    obj_value["obj"]["000"]["transfer"] = [0, 0, 0]
    obj_value["obj"]["000"]["stand_point"] = np.ndarray.tolist(stand_point)
    return obj_value


def aug_mesh(a, obj_value):
    obj_value["obj"]["000"]['aug_count'] = a
    return obj_value


def run_cmd(args, i, a, value_use, key, tmp_path):
    value_use = trans_to_center(value_use)
    tmp_dict = {key: value_use}

    with open(tmp_path, 'w') as f:
        json.dump(tmp_dict, f, indent=4)

    viz = "" if args.viz else "--headless"

    print(f"\033[91mbegin {i}/{len_objs}: {a}\033[0m")
    os.system(
        f"CUDA_VISIBLE_DEVICES={args.gpu_id} python3 unihsi/run.py --obj_file {tmp_path} --task {args.task} --save_root {args.save_root} {viz} --test --num_envs 1 --cfg_env unihsi/data/cfg/humanoid_unified_interaction_scene_0.yaml --cfg_train unihsi/data/cfg/train/rlg/amp_humanoid_task_deep_layer_2we.yaml --motion_file motion_clips/chair_mo.npy --checkpoint checkpoints/Humanoid.pth"
    )
    print(f"\033[91mend {i}/{len_objs}: {a}\033[0m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', required=True)
    parser.add_argument('-t', '--tmp', type=str, required=True)
    parser.add_argument('-g', '--gpu_id', type=str, required=True)
    parser.add_argument('-n', '--num', type=int, default=5)
    parser.add_argument('-an', '--aug_num', type=int, default=10)
    parser.add_argument('-s', '--save_root', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, choices=['UniHSI_PartNet', 'UniHSI_PartNet_AUG'])
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--actions', default=["chair", "bed"])
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        all_objs = json.load(f)

    tmp_path = f"tmp/tmp{args.tmp}.json"

    len_objs = len(all_objs)
    for i in range(len_objs):
        key = str(i).rjust(4, "0")
        value = all_objs[key]

        if not value["obj"]["000"]["name"] in args.actions:
            continue

        value["obj"]["000"]["count"] = args.num

        if args.task == "UniHSI_PartNet":
            run_cmd(args, i, 0, value, key, tmp_path)
        elif args.task == "UniHSI_PartNet_AUG":
            for a in range(args.aug_num):
                value_final = aug_mesh(a, value)
                run_cmd(args, i, a, value_final, key, tmp_path)
        else:
            raise ValueError()
