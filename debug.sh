#!/usr/bin/env bash


MADRONA_MWGPU_KERNEL_CACHE=/home/chemist/Desktop/ICRA2026/madrona_mjx/build/kernel_cache \
MADRONA_BVH_KERNEL_CACHE=/home/chemist/Desktop/ICRA2026/madrona_mjx/build/bvh_cache \
python train_pick_cube_ppo.py \
    --env_name="PandaPickCubeCartesianModified" \
    --num_timesteps="1_000" \
    --seed="1" \
    --actuator="position" \
    --action="joint_increment" \
    --use_tb \
    --vision \
    --log_training_metrics \
