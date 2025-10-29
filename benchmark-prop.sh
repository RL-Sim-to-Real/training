#!/bin/bash
export DISPLAY=${DISPLAY:-:0}
export MUJOCO_GL=egl


ENV_NAME="PandaPickCubeCartesian3D"
NUM_TIMESTEPS=20_000_000
SEEDS=(0)
DEVICE_ID=0

# Only include compatible pairs here:

PAIRS=(
  "velocity joint"
  # "torque joint"
  # "position cartesian_increment"
  # "position joint_increment"
)
action_scales=(1.0 0.05 0.05 0.05)

for seed in "${SEEDS[@]}"; do

  i=0  # Initialize index counter
  for pair in "${PAIRS[@]}"; do
    set -- $pair
    actuator="$1"
    action="$2"
    echo "Running actuator=$actuator action=$action seed=$seed"


    # MADRONA_MWGPU_KERNEL_CACHE=/home/chemist/Desktop/ICRA2026/madrona_mjx/build/kernel_cache \
    # MADRONA_BVH_KERNEL_CACHE=/home/chemist/Desktop/ICRA2026/madrona_mjx/build/bvh_cache \
    # python train_pick_cube_ppo.py \
    #   --env_name="$ENV_NAME" \
    #   --num_timesteps="$NUM_TIMESTEPS" \
    #   --seed="$seed" \
    #   --actuator="$actuator" \
    #   --action="$action" \
    #   --use_tb \
    #   --log_training_metrics \
    #   --vision \
    #   --device_id=1 \
    #   --action_scale="${action_scales[$i]}" 

    # With propioception
    # MADRONA_MWGPU_KERNEL_CACHE=/home/chemist/Desktop/ICRA2026/madrona_mjx/build/kernel_cache \
    # MADRONA_BVH_KERNEL_CACHE=/home/chemist/Desktop/ICRA2026/madrona_mjx/build/bvh_cache \
    python train_pick_cube_ppo.py \
      --env_name="$ENV_NAME" \
      --num_timesteps="$NUM_TIMESTEPS" \
      --seed="$seed" \
      --actuator="$actuator" \
      --action="$action" \
      --use_tb \
      --vision \
      --proprioception \
      --device_id="$DEVICE_ID" \
      --action_scale="${action_scales[$i]}"
    ((i++))
  done



  # Pause before the next run to cool-down GPU
  echo "Pausing for 2 minutes..."
  echo "DO NOT INTERRUPT THIS PAUSE!"
  sleep 5
done


