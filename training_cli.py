"""Brax PPO entry point shared by the original pick/push benchmark commands."""
import argparse
from datetime import datetime, timezone
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import time


def parser(default_env='PandaPickCuboid'):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env_name', choices=['PandaPickCuboid', 'PandaPushCuboid'], default=default_env)
    p.add_argument('--impl', choices=['jax', 'warp'], default='warp')
    for name, default in [('vision', True), ('proprioception', False),
                          ('domain_randomization', True), ('use_tb', False),
                          ('use_wandb', False), ('log_training_metrics', False),
                          ('run_evals', True), ('normalize_observations', True),
                          ('play_only', False), ('record_video', False)]:
        p.add_argument('--' + name, action=argparse.BooleanOptionalAction, default=default)
        p.add_argument('--no' + name, dest=name, action='store_false', help=argparse.SUPPRESS)
    ints = dict(seed=1, num_timesteps=50_000_000, num_envs=1024, num_eval_envs=128,
                num_evals=5, num_resets_per_eval=1, episode_length=0, action_repeat=1, unroll_length=10,
                num_minibatches=8, num_updates_per_batch=8, batch_size=256,
                training_metrics_steps=5_000_000, device_id=0, render_size=64,
                rollout_steps=32)
    for name, default in ints.items():
        p.add_argument('--' + name, type=int, default=default)
    for name, default in dict(action_scale=.02, reward_scaling=.1, discounting=.97,
                              learning_rate=5e-4, entropy_cost=.0075,
                              max_grad_norm=1., clipping_epsilon=.2).items():
        p.add_argument('--' + name, type=float, default=default)
    p.add_argument('--actuator', choices=['position', 'velocity'], default='position')
    p.add_argument('--action', choices=['cartesian_increment', 'joint_increment', 'joint'], default='cartesian_increment')
    p.add_argument('--suffix', default='')
    p.add_argument('--output_dir', type=Path, default=Path('results'))
    p.add_argument('--load_checkpoint_path', type=Path)
    p.add_argument('--smoke', action='store_true', help='Small PPO update, evaluation, save/reload and rollout.')
    return p


def parse_args(default_env='PandaPickCuboid', argv=None):
    p = parser(default_env)
    import sys
    raw = list(sys.argv[1:] if argv is None else argv)
    if '--smoke' in raw:
        p.set_defaults(num_envs=2, num_eval_envs=2, num_timesteps=8,
                       episode_length=4, unroll_length=2, batch_size=2,
                       num_minibatches=1, num_updates_per_batch=1,
                       num_evals=1, rollout_steps=4)
    args = p.parse_args(raw)
    for key in ('num_envs', 'num_eval_envs', 'unroll_length', 'batch_size',
                'num_minibatches', 'num_updates_per_batch', 'action_repeat', 'render_size'):
        if getattr(args, key) <= 0:
            p.error(f'--{key} must be positive')
    if args.num_timesteps < 0 or args.episode_length < 0 or args.rollout_steps < 1:
        p.error('timesteps and episode length must be nonnegative; rollout_steps must be positive')
    if (args.batch_size * args.num_minibatches) % args.num_envs:
        p.error('batch_size * num_minibatches must be divisible by num_envs')
    if args.vision and args.impl != 'warp':
        p.error('vision requires --impl=warp; use --no-vision for CPU physics')
    if args.vision and args.action_repeat != 1:
        p.error('vision training requires action_repeat=1')
    if args.action == 'joint_increment' and args.actuator != 'position':
        p.error('joint_increment actions require the position actuator')
    if args.action == 'joint' and args.actuator != 'velocity':
        p.error('joint actions require the velocity actuator')
    if args.play_only and not args.load_checkpoint_path:
        p.error('--play_only requires --load_checkpoint_path')
    return args


def main(default_env='PandaPickCuboid', argv=None):
    args = parse_args(default_env, argv)
    # Set device visibility before importing anything that initializes JAX.
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(Path('.cache/jax').resolve()))
    os.environ.setdefault('WARP_CACHE_PATH', str(Path('.cache/warp').resolve()))
    import jax
    import numpy as np
    from brax.training.agents.ppo import train as ppo, networks, networks_vision
    from simulation import make_env

    env, cfg = make_env(args)
    eval_env, _ = make_env(args, num_envs=args.num_eval_envs, seed=args.seed + 1)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    name = f'{args.env_name}-{args.action}-{args.actuator}-seed{args.seed}-{stamp}{args.suffix}'
    output = args.output_dir.resolve() / name
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(arguments=vars(args), environment=cfg.to_dict(),
                    devices=[str(d) for d in jax.devices()],
                    versions={p: importlib.metadata.version(p) for p in
                              ['jax', 'jaxlib', 'brax', 'mujoco', 'mujoco-mjx', 'flax', 'warp-lang', 'orbax-checkpoint']})
    (output / 'config.json').write_text(json.dumps(manifest, default=str, indent=2))
    writer = None
    if args.use_tb:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(str(output))
    if args.use_wandb:
        import wandb
        wandb.init(project='sim-to-real', name=name, config=json.loads(json.dumps(manifest, default=str)))

    def progress(step, metrics):
        values = {k: float(v) for k, v in metrics.items()}
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError(f'Nonfinite training metric: {values}')
        with (output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=int(step), **values)) + '\n')
        print(f'step={step} {values}', flush=True)
        if writer:
            for k, v in values.items():
                writer.add_scalar(k, v, step)
        if args.use_wandb:
            wandb.log(values, step=step)

    network = functools.partial(
        networks_vision.make_ppo_networks_vision if args.vision else networks.make_ppo_networks,
        policy_hidden_layer_sizes=(32, 32) if args.smoke else (256, 256, 256),
        value_hidden_layer_sizes=(32, 32) if args.smoke else (256, 256, 256),
        **(dict(normalise_channels=True, policy_obs_key='_prop' if args.proprioception else '',
                value_obs_key='_prop' if args.proprioception else '') if args.vision else {}))
    keys = ('num_envs', 'num_eval_envs', 'num_evals', 'num_resets_per_eval', 'unroll_length', 'batch_size',
            'num_minibatches', 'num_updates_per_batch', 'learning_rate', 'entropy_cost',
            'discounting', 'reward_scaling', 'max_grad_norm', 'clipping_epsilon',
            'normalize_observations', 'log_training_metrics', 'training_metrics_steps',
            'run_evals', 'seed')
    started = time.monotonic()
    try:
        inference, params, metrics = ppo.train(
            environment=env, eval_env=eval_env, wrap_env=False,
            max_devices_per_host=1, use_pmap_on_reset=True,
            vision=args.vision, augment_pixels=args.vision,
            num_timesteps=0 if args.play_only else args.num_timesteps,
            episode_length=cfg.episode_length, action_repeat=args.action_repeat, network_factory=network,
            progress_fn=progress, deterministic_eval=True,
            save_checkpoint_path=str(output / 'checkpoints'),
            restore_checkpoint_path=str(args.load_checkpoint_path.resolve()) if args.load_checkpoint_path else None,
            **{k: getattr(args, k) for k in keys})
        # Check the on-disk Orbax checkpoint, not only in-memory return values.
        if not args.play_only and args.num_timesteps > 0:
            from brax.training.agents.ppo import checkpoint
            checkpoints = [d for d in (output / 'checkpoints').iterdir()
                           if d.is_dir() and d.name.isdigit()]
            latest = max(checkpoints, key=lambda d: int(d.name))
            checkpoint_params = checkpoint.load(str(latest))
            original_leaves = jax.tree.leaves(params)
            restored_leaves = jax.tree.leaves(checkpoint_params)
            if len(original_leaves) != len(restored_leaves):
                raise ValueError('Checkpoint parameter structure changed on reload')
            for before, after in zip(original_leaves, restored_leaves):
                np.testing.assert_allclose(np.asarray(before), np.asarray(after), rtol=0, atol=0)
            params = checkpoint_params
        # Preserve the research parameter format, and actually reload it for inference.
        path = output / 'params.pkl'
        with path.open('wb') as f:
            pickle.dump(params, f)
        with path.open('rb') as f:
            restored = pickle.load(f)
        policy = jax.jit(inference(restored, deterministic=True))
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(args.seed + 2), args.num_envs))
        step_fn = jax.jit(env.step)
        rewards = []
        frames = []
        trajectory = []
        for i in range(args.rollout_steps):
            action, _ = policy(state.obs, jax.random.PRNGKey(i))
            state = step_fn(state, action)
            for leaf in jax.tree.leaves((state.obs, state.reward, state.data.qpos, action)):
                if not np.isfinite(np.asarray(leaf)).all():
                    raise FloatingPointError('Nonfinite restored-policy rollout')
            rewards.append(np.asarray(state.reward))
            if args.record_video:
                if args.vision:
                    pixels = np.asarray(state.obs['pixels/view_0'][0, ..., -3:])
                    frames.append((np.clip(pixels, 0, 1) * 255).astype(np.uint8))
                else:
                    trajectory.append(jax.tree.map(lambda x: x[0], state))
        if args.record_video:
            import mediapy
            import imageio_ffmpeg
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
            if not args.vision:
                frames = env.render(trajectory, width=640, height=480)
            mediapy.write_video(output / 'rollout.mp4', frames, fps=1 / env.dt)
        np.savez(output / 'rollout.npz', rewards=np.asarray(rewards), qpos=np.asarray(state.data.qpos))
        report = dict(status='passed', elapsed_seconds=time.monotonic()-started,
                      rollout_steps=args.rollout_steps, num_envs=args.num_envs,
                      impl=args.impl, vision=args.vision,
                      domain_randomization=args.domain_randomization)
        (output / 'validation.json').write_text(json.dumps(report, indent=2))
        print(f'Completed training, reload and rollout: {output}', flush=True)
    finally:
        if writer:
            writer.close()
        if args.use_wandb:
            wandb.finish()


if __name__ == '__main__':
    main()
