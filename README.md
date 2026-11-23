# Sim-to-real cuboid research benchmarks

The original `PandaPickCuboid` and `PandaPushCuboid` tasks now run through a shared
Brax PPO launcher. Their XML assets, Cartesian/joint controllers, reward formulas,
and task-specific horizons are retained from the research Playground fork.
Vision uses **MJX-Warp physics and MuJoCo's native Warp batch renderer**; Madrona
and the custom Brax fork are no longer required by these launchers.

## Install

Python 3.12, [uv](https://docs.astral.sh/uv/), and
[Task](https://taskfile.dev/docs/installation) v3 are required. From this checkout:

```bash
task setup MODE=cpu   # CPU state-observation testing
# On an NVIDIA Turing-or-newer GPU with an R580-or-newer driver:
task setup MODE=cuda
```

Setup fetches the pinned research Playground submodule and its pinned MuJoCo
Menagerie assets, applies `patches/playground-modern.patch`, and installs the
locked dependencies into `.venv`. It can be rerun; conflicting local submodule
changes cause an error instead of being overwritten. The patch is tracked in the
parent repository, so no unpublished submodule commit is needed. After editing
the fork, refresh it with `python3 scripts/export_playground_patch.py`.

After initial setup, recreate or synchronize `.venv` directly from the manifest
and lockfile (uv creates the environment automatically if it is missing):

```bash
uv sync --locked                # CPU
uv sync --locked --extra cuda   # NVIDIA GPU
```

Python is restricted to the tested 3.12 series. The default sync includes pytest;
use `--no-dev` for a runtime-only environment. Playground is installed editable
from the patched local submodule. Use `task setup` first on a fresh clone to
initialize and patch that submodule.

The tested dependency selection is Brax 0.14.2, JAX/JAXlib 0.8.2, Flax 0.12.2,
Orbax 0.11.30, MuJoCo/MJX 3.17.0, and Warp 1.18.0. MJX's `warp` extra supplies its
matching Warp runtime. The PyPI Warp wheel uses CUDA 13.4 and requires an
R580-or-newer driver, even though JAX uses CUDA 12 runtime packages. Do not install an unrelated standalone `mujoco-warp`
version to override it. Newer JAX releases remove an API still used by this Brax
version. `pyproject.toml` is the single dependency manifest; `uv.lock` records
the exact CPU and CUDA dependency versions.

## Train and validate

```bash
# Actual PPO update, evaluation, checkpoint saving, parameter reload and rollout:
.venv/bin/python train_pick_cube_ppo.py --smoke --impl=jax --no-vision
.venv/bin/python train_push_cube_ppo.py --smoke --impl=jax --no-vision

# Both tasks and all four benchmark actuator/action pairs, including regression tests:
task validate MODE=cpu
task validate MODE=cuda DEVICE_ID=1

# Original visual experiment style:
.venv/bin/python train_pick_cube_ppo.py \
  --env_name=PandaPickCuboid --num_timesteps=15000000 --seed=0 \
  --impl=warp --vision --proprioception --domain_randomization \
  --actuator=position --action=cartesian_increment --action_scale=.05 \
  --num_envs=1024 --num_eval_envs=128 --use_tb --log_training_metrics

# Existing ten-seed benchmark matrices (GPU 0/1, Cartesian and joint actions):
task benchmark:pick-1
```

Measure synchronized simulator throughput separately from PPO compilation:

```bash
.venv/bin/python benchmark_simulation.py --env_name=PandaPushCuboid \
  --device_id=1 --num_envs=128 --steps=100 --report=results/push-128.json
.venv/bin/python load_and_render_policy.py --run_dir=results/YOUR_RUN \
  --device_id=1 --rollout_steps=150
```

`--record_video` also exports policy observations during training's final rollout.
`--help` lists supported flags. `--no-vision`, `--no-domain_randomization`, and
`--no-proprioception` disable features. The randomization flag controls model variation and brightness;
task reset sampling and the original observation augmentation remain part of the task. `--smoke` reduces the batch and rollout
sizes; it does not establish learning convergence or reproduce the full paper
results. Vision requires `--impl=warp` and a working CUDA JAX device; CPU runs
use `--impl=jax --no-vision`. The launchers deliberately select one GPU, while
vectorizing environments within it. Multi-GPU training is not validated.

Every run gets a unique directory under `results/` containing the effective
configuration, library versions, metrics, Orbax checkpoints, `params.pkl`, a
restored-policy `rollout.npz`, and `validation.json` after successful completion.
Existing runs are never removed. Use `--load_checkpoint_path` with an individual
numeric Orbax checkpoint directory to resume; add `--play_only` to evaluate
without training. Restore with the same network/observation settings (including
`--smoke` when restoring a smoke checkpoint). Pickle parameters are for trusted
local research artifacts only. Legacy checkpoints from the old custom Brax
network are not guaranteed to match current Brax architectures.

## Migration details and research comparability

- Both launchers now honor environment count, vision, randomization, PPO and
  evaluation options. The original scripts silently ignored many of these.
- Fixed-size MuJoCo contact sensors preserve the original minimum contact
  distance predicate without accessing Warp's shared contact buffer.
- Warp uses 32 line-search iterations to avoid the original limit of 8 being exhausted.
- Nonvisual reset observations now match step observations. Finger height uses
  the geom position instead of incorrectly indexing body positions with a geom ID.
- Per-world actuator gain/bias scaling remains in [0.9, 1.1]. Randomization is
  reproducible from the experiment seed, with a separate evaluation seed.
- Visual randomization includes box color, camera position/orientation, light
  position/direction, and shadows. Native material IDs replace Madrona-specific
  negative sentinels and arbitrary texture indices. Native lighting, camera
  perturbations, and material appearance are not pixel-identical to Madrona;
  new results must be treated as a new simulator/renderer baseline.
- Policy networks use the current Brax implementation and its default Swish MLP
  activation; the old custom Brax/ReLU network is not reproduced exactly.
- Historical notebooks are not migrated entry points; use the CLI commands above.
- Robot deployment and RealSense/ROS scripts under `sim_to_real/` and `tests/`
  remain hardware-specific legacy tools. Automated tests only collect
  `tests/simulation/`; they never import or operate robot hardware.

See `VALIDATION.md` for the checks actually run and outstanding GPU validation.
Native rendering follows [MJX batch rendering](https://mujoco.readthedocs.io/en/stable/mjx.html)
and [MuJoCo contact sensors](https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor-contact).

## Task commands

All repository-owned shell launchers have been replaced by `Taskfile.yml`.
Run `task --list` to see the workflows, or `task --dry TASK` to preview commands.
Third-party dependency scripts are left upstream-owned.

```bash
task setup MODE=cuda
task validate MODE=cuda DEVICE_ID=1
task debug -- --impl=jax --no-vision
task benchmark:pick-1
task benchmark:pick-2
task benchmark:push-1
task benchmark:push-2
task benchmark:simulation DEVICE_ID=1 -- --num_envs=1024
task policy:render DEVICE_ID=1 -- --run_dir=results/YOUR_RUN
task patch:apply
task patch:export
task plot LOGDIR=logs
```

Benchmark tasks preserve their original ten seeds, action/actuator pairs, training
budgets, GPU defaults, and five-second pause after each seed. They run sequentially
and stop on failure. Override `SEEDS="0 1"`, `NUM_TIMESTEPS=1000`,
`BENCHMARK_DEVICE=1`, `COOLDOWN=0`, or `PYTHON=/path/to/python` as needed.
Options after `--` are forwarded to training/debug/render/plot commands.

`plot` preserves the historical five metric tags and run-name groups from the old
plotting script; override `GROUPS` and `LOGDIR` for a different collection. It
requires the plotting script's optional TensorBoard dependencies. The archived
torque experiment's settings are preserved in `_archive/torque-benchmark.md`;
`task archive:benchmark-prop-torque` reports its unsupported status explicitly.
