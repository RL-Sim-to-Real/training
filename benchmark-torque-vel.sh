#!/bin/bash
export DISPLAY=${DISPLAY:-:0}
export MUJOCO_GL=egl


ENV_NAME="PandaPickCubeCartesianModified"
NUM_TIMESTEPS=30_000_000
SEEDS=(0 1 2 3 4)
DEVICE_ID=1

# Only include compatible pairs here:

PAIRS=(
  "velocity joint"
  "torque joint"
)

for seed in "${SEEDS[@]}"; do

  
  for pair in "${PAIRS[@]}"; do
    set -- $pair
    actuator="$1"
    action="$2"
    echo "Running actuator=$actuator action=$action seed=$seed"


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
      --action_scale=1.0 

    # With propioception
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
      --action_scale=1.0 

  done



  # Pause before the next run to cool-down GPU
  echo "Pausing for 2 minutes..."
  echo "DO NOT INTERRUPT THIS PAUSE!"
  sleep 5
done


