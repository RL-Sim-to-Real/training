
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
python train_pick_cube_ppo.py --env_name=PandaPickCubeCartesianModified \
 --num_timesteps=50_000_000 \
 --seed=0 \
 --vision \
 --log_training_metrics \
 --use_tb \
 --actuator=position \
 --action=cartesian_increment
```

## Trouble shooting

When using ROS if you encounter this error:

```
ImportError: /lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0

```

```
 ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 ~/anaconda3/envs/[your env name here]/lib/libffi.so.7
```