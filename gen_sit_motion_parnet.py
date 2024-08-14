import os
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', required=True)
    parser.add_argument('-t', '--tmp', type=str, required=True)
    parser.add_argument('-g', '--gpu_id', type=str, required=True)
    parser.add_argument('-n', '--num', type=int, default=5)
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        all_objs = json.load(f)

    tmp_path = f"tmp/tmp{args.tmp}.json"

    len_objs = len(all_objs)
    for i in range(len_objs):
        key = str(i).rjust(4, "0")
        value = all_objs[key]

        if value["obj"]["000"]["name"] == "reach":
            continue

        value["obj"]["000"]["count"] = args.num
        tmp_dict = {key: value}
        with open(tmp_path, 'w') as f:
            json.dump(tmp_dict, f, indent=4)

        print(f"\033[91mbegin {i}/{len_objs}\033[0m")
        os.system(
            f"CUDA_VISIBLE_DEVICES={args.gpu_id} python3 unihsi/run.py --obj_file {tmp_path} --task UniHSI_PartNet --test --headless --num_envs 1 --cfg_env unihsi/data/cfg/humanoid_unified_interaction_scene_0.yaml --cfg_train unihsi/data/cfg/train/rlg/amp_humanoid_task_deep_layer_2we.yaml --motion_file motion_clips/chair_mo.npy --checkpoint checkpoints/Humanoid.pth"
        )
        print(f"\033[91mend {i}/{len_objs}\033[0m")
