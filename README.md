
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
python train_jax_ppo.py --env_name=PandaPickCubeCartesian --vision --log_training_metrics
```