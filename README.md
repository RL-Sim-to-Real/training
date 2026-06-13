
FrankaVisionReach Training
```
python reacher_visual_dense.py --replay_buffer_capacity=1_000_000 --env_steps=1_000_000 --seed=0 --mode=img_prop --layer_norm --apply_weight_clip
```

FrankaReach training


```
python reacher_non_visual_dense.py --replay_buffer_capacity=1_00_000 --env_steps=1_00_000 --seed=0 --mode=prop --action_mode="velocity"
```

FrankaTrack training


```
python tracker_non_visual_dense.py --replay_buffer_capacity=1_00_000 --env_steps=1_00_000 --seed=0 --mode=prop --action_mode="velocity"
```

PandaPickCubeCartesian Training

```
python train_pick_cube_ppo.py --env_name=PandaPickCuboid \
 --num_timesteps=50_000_000 \
 --seed=0 \
 --vision \
 --log_training_metrics \
 --use_tb \
 --actuator=position \
 --action=cartesian_increment
```

## Troubleshooting

When using ROS, if you encounter this error:

```
ImportError: /lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0

```

```
 ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 ~/anaconda3/envs/[your env name here]/lib/libffi.so.7
```

## Installation Guide

Use this repository with its `dependencies/` submodules instead of cloning each dependency manually.

Clone with submodules:
```
git clone --recurse-submodules git@github.com:RL-Sim-to-Real/training.git
cd training
```

If you already cloned the repository, initialize and update submodules:
```
git submodule sync --recursive
git submodule update --init --recursive
```

This populates:
- `dependencies/brax`
- `dependencies/mujoco_playground`
- `dependencies/madrona_mjx`

Start by installing Madrona first (`dependencies/madrona_mjx`) and ensure you have `jax[cuda_local]<=0.5.3` and `flax<=0.10.6`.
**Note**: When building Madrona, ensure you use `cmake -DLOAD_VULKAN=OFF ..`
**Tested with**: CUDA 12.5

## For compute canada

```
module load StdEnv/2023 python/3.11 mujoco/3.3.0 cuda/12.2
virtualenv --no-download --clear ~/ENV && source ~/ENV/bin/activate

salloc --time=01:00:00 --gres=gpu:1 --mem=32G --cpus-per-task=4 --account=aip-ashique

```
