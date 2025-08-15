#!/usr/bin/env bash

ENV_NAME="PandaPickCubeCartesianModified"
NUM_TIMESTEPS=50_000_000
SEEDS=(0 1 2 3 4)

# Only include compatible pairs here:
PAIRS=(
  # "position cartesian_increment"
  "velocity joint"
  "position joint"
  # add more valid pairs...
)

for pair in "${PAIRS[@]}"; do
  set -- $pair
  actuator="$1"
  action="$2"
  # Perform process cleanup
  echo "Cleaning up processes before next run..."
  pkill -f train_pick_cube_ppo.py || true

  # Pause before the next run to cool-down GPU
  echo "Pausing for 5 seconds..."
  sleep 5
  
  for seed in "${SEEDS[@]}"; do
    echo "Running actuator=$actuator action=$action seed=$seed"
    python train_pick_cube_ppo.py \
      --env_name="$ENV_NAME" \
      --num_timesteps="$NUM_TIMESTEPS" \
      --seed="$seed" \
      --actuator="$actuator" \
      --action="$action" \
      --action_scale=1.0 \
      --use_tb \
      --log_training_metrics 

  done
done
