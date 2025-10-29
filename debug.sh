#!/usr/bin/env bash

python -u train_pick_cube_ppo.py \
    --env_name="PandaPickCubeCartesian3D" \
    --num_timesteps="1_000" \
    --seed="1" \
    --actuator="position" \
    --action="joint_increment" \
    --use_tb \
    --vision \
    --log_training_metrics \
