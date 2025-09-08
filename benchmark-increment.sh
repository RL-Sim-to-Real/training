#!/usr/bin/env bash

ENV_NAME="PandaPickCubeCartesianModified"
NUM_TIMESTEPS=50_000_000
SEEDS=(3 4 5)
DEVICE_ID=0

# Only include compatible pairs here:

PAIRS=(
  # "position cartesian_increment"
  # "position joint_increment"
  "torque joint"
)


for pair in "${PAIRS[@]}"; do
  set -- $pair
  actuator="$1"
  action="$2"
  # Perform process cleanup
  echo "Cleaning up processes before next run..."
  
  # With proprioception
  for seed in "${SEEDS[@]}"; do
    echo "Running actuator=$actuator action=$action seed=$seed"

    # MADRONA_MWGPU_KERNEL_CACHE=/home/nika/Desktop/Research/madrona_mjx/build/kernel_cache \
    # MADRONA_BVH_KERNEL_CACHE=/home/nika/Desktop/Research/madrona_mjx/build/bvh_cache \
    python train_pick_cube_ppo.py \
      --env_name="$ENV_NAME" \
      --num_timesteps="$NUM_TIMESTEPS" \
      --seed="$seed" \
      --actuator="$actuator" \
      --action="$action" \
      --use_tb \
      --log_training_metrics \
      --vision \
      --proprioception \
      --device_id="$DEVICE_ID" \
      --action_scale=1.0 

    
    # MADRONA_MWGPU_KERNEL_CACHE=/home/nika/Desktop/Research/madrona_mjx/build/kernel_cache \
    # MADRONA_BVH_KERNEL_CACHE=/home/nika/Desktop/Research/madrona_mjx/build/bvh_cache \
    python train_pick_cube_ppo.py \
      --env_name="$ENV_NAME" \
      --num_timesteps="$NUM_TIMESTEPS" \
      --seed="$seed" \
      --actuator="$actuator" \
      --action="$action" \
      --use_tb \
      --log_training_metrics \
      --vision \
      --device_id="$DEVICE_ID" \
      --action_scale=1.0 


  done

  # Pause before the next run to cool-down GPU
  echo "Pausing for 2 minutes..."
  echo "DO NOT INTERRUPT THIS PAUSE!"
  sleep 5
done

