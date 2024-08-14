python3 unihsi/run.py \
    --task UniHSI_PartNet_BKP\
    --test \
    --num_envs 1 \
    --cfg_env unihsi/data/cfg/humanoid_unified_interaction_scene_0.yaml \
    --cfg_train unihsi/data/cfg/train/rlg/amp_humanoid_task_deep_layer_2we.yaml \
    --motion_file motion_clips/chair_mo.npy \
    --checkpoint checkpoints/Humanoid.pth \
    --obj_file sceneplan/partnet_train_simple_dev.json
