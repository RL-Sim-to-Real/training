"""Measure warmed, synchronized batched simulator throughput without PPO."""
import json
import os
from pathlib import Path
import time
from training_cli import parser


def main():
    p = parser()
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--report', type=Path, default=Path('results/simulation-benchmark.json'))
    args = p.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        p.error('steps and num_envs must be positive')
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('WARP_CACHE_PATH', str(Path('.cache/warp').resolve()))
    import jax
    import jax.numpy as jp
    import numpy as np
    from simulation import make_env

    env, cfg = make_env(args)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
    started = time.perf_counter()
    state = reset(keys)
    action = jp.zeros((args.num_envs, env.action_size))
    state = step(state, action)
    jax.block_until_ready(state)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(args.steps):
        state = step(state, action)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - started
    for value in jax.tree.leaves((state.obs, state.reward, state.data.qpos)):
        if not np.isfinite(np.asarray(value)).all():
            raise FloatingPointError('Nonfinite benchmark state')
    report = dict(task=args.env_name, impl=args.impl, vision=args.vision,
                  domain_randomization=args.domain_randomization, seed=args.seed,
                  num_envs=args.num_envs, steps=args.steps, compile_seconds=compile_seconds,
                  elapsed_seconds=elapsed, env_steps_per_second=args.num_envs*args.steps/elapsed,
                  physics_substeps_per_control_step=env.n_substeps,
                  devices=[str(d) for d in jax.devices()], environment=cfg.to_dict())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
