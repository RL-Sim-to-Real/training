#!/bin/bash
export DISPLAY=${DISPLAY:-:0}
export MUJOCO_GL=egl


ENV_NAME="PandaPushCuboid"
NUM_TIMESTEPS=15_000_000
SEEDS=({0..9})
DEVICE_ID=1

# Only include compatible pairs here:

PAIRS=(
  "position cartesian_increment 0.05"
  "velocity cartesian_increment 0.05"
  "velocity joint 1"
  "position joint_increment 0.05"
)
# action_scales=(1.0 0.05 0.05 0.05)

for seed in "${SEEDS[@]}"; do

  i=0  # Initialize index counter
  for pair in "${PAIRS[@]}"; do
    set -- $pair
    actuator="$1"
    action="$2"
    action_scale="$3"
    echo "Running actuator=$actuator action=$action seed=$seed"


    # With propioception
    # MADRONA_MWGPU_KERNEL_CACHE=/home/chemist/Desktop/ICRA2026/madrona_mjx/build/kernel_cache \
    # MADRONA_BVH_KERNEL_CACHE=/home/chemist/Desktop/ICRA2026/madrona_mjx/build/bvh_cache \

    # MADRONA_MWGPU_KERNEL_CACHE=/home/nika/Desktop/Research/madrona_mjx/build/kernel_cache \
    # MADRONA_BVH_KERNEL_CACHE=/home/nika/Desktop/Research/madrona_mjx/build/bvh_cache \
    python train_push_cube_ppo.py \
      --env_name="$ENV_NAME" \
      --num_timesteps="$NUM_TIMESTEPS" \
      --seed="$seed" \
      --actuator="$actuator" \
      --action="$action" \
      --use_tb \
      --vision \
      --proprioception \
      --log_training_metrics \
      --device_id="$DEVICE_ID" \
      --action_scale="$action_scale"
    ((i++))
  done



  # Pause before the next run to cool-down GPU
  echo "Pausing for 2 minutes..."
  echo "DO NOT INTERRUPT THIS PAUSE!"
  sleep 5
done


