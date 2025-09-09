#!/usr/bin/env bash
export DISPLAY=${DISPLAY:-:0}
export MUJOCO_GL=egl

ENV_NAME="PandaPickCubeCartesianModified"
NUM_TIMESTEPS=30_000_000
SEEDS=(0 1 2 3 4)
DEVICE_ID=0

# Only include compatible pairs here:

PAIRS=(
  "position cartesian_increment"
  "position joint_increment"
)


for seed in "${SEEDS[@]}"; do
  
  for pair in "${PAIRS[@]}"; do
    set -- $pair
    actuator="$1"
    action="$2"
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
      --device_id="$DEVICE_ID" \
      --action_scale=0.02


    # With proprioception
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
      --action_scale=0.02 

    


  done

  # Pause before the next run to cool-down GPU
  echo "Pausing for 2 minutes..."
  echo "DO NOT INTERRUPT THIS PAUSE!"
  sleep 5
done

