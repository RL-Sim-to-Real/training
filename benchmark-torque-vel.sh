#!/bin/bash

ENV_NAME="PandaPickCubeCartesianModified"
NUM_TIMESTEPS=50_000_000
SEEDS=(1 2 3 4 5)
DEVICE_ID=1

# Only include compatible pairs here:

PAIRS=(
  "torque joint"
  "velocity joint"
)


for pair in "${PAIRS[@]}"; do
  set -- $pair
  actuator="$1"
  action="$2"
  # Perform process cleanup
  echo "Cleaning up processes before next run..."

  # With prioception
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
      --device_id=1 \
      $( [[ "$actuator" == "torque" ]] && echo "--action_scale=0.5 \\" || echo "--action_scale=1.0 \\" )


    # MADRONA_MWGPU_KERNEL_CACHE=/home/chemist/Desktop/ICRA2026/madrona_mjx/build/kernel_cache \
    # MADRONA_BVH_KERNEL_CACHE=/home/chemist/Desktop/ICRA2026/madrona_mjx/build/bvh_cache \
    python train_pick_cube_ppo.py \
      --env_name="$ENV_NAME" \
      --num_timesteps="$NUM_TIMESTEPS" \
      --seed="$seed" \
      --actuator="$actuator" \
      --action="$action" \
      --use_tb \
      --log_training_metrics \
      --vision \
      --device_id=1 \
      $( [[ "$actuator" == "torque" ]] && echo "--action_scale=0.5 \\" || echo "--action_scale=1.0 \\" )\

  done



  # Pause before the next run to cool-down GPU
  echo "Pausing for 2 minutes..."
  echo "DO NOT INTERRUPT THIS PAUSE!"
  sleep 5
done


