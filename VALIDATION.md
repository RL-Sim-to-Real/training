# Modernization validation

The original fork is pinned at `1f220451688327891b61bcb50e112ac012b7ecd7`.
The tracked patch is `patches/playground-modern.patch`; its tested hash and run
artifacts are recorded in `validation/results.json`.

## Completed checks

- **12 unit/native checks:** CLI validation and explicit smoke overrides,
  per-world seeded randomization, packed RGB channel decoding, and agreement
  between contact sensors and native MuJoCo contact distances.
- **4 CPU simulator integrations:** both cuboid tasks with position and velocity
  Cartesian controllers, batched reset/step, finite states, consistent observation
  shapes, deterministic reset, and episode auto-reset.
- **4 GPU vision integrations:** both tasks, model randomization on and off,
  batched rendering and stepping, finite proprioception, and visible red cuboid
  pixels. The corrected Warp camera image was also compared visually with native
  MuJoCo at the same joint/object state.
- **8 visual PPO end-to-end cases:** the matrix below. Each completed an optimizer
  update, evaluation, Orbax save and exact parameter-equality check after reload,
  pickle export/reload, finite policy rollout, TensorBoard output, and MP4 export.
  All eight exported videos were decoded successfully (4 frames, 64×64 RGB).
- **1,024-world simulation:** both tasks completed reset, warmup, and 32 synchronized
  visual control steps with domain randomization and finite observations,
  rewards, and joint positions.
- **Standalone restore:** `load_and_render_policy.py` loaded the saved visual Pick
  run and exported an eight-frame video using its saved network/configuration.
- **Normal-size networks:** Push completed 128 PPO steps with 32 training worlds,
  16 evaluation worlds and pixels-only observations. Pick completed 2,560 steps
  with 128 training worlds, 16 evaluation worlds and proprioception. Both use
  the production 256×256×256 MLP widths rather than smoke-test widths.
- CPU state-observation Pick PPO also completed training, save/reload and inference.
- Dependency consistency, shell syntax, Python compilation, and whitespace checks
  passed. The patch applies to a pristine archive of the pinned fork and passes
  the reverse-application check used by the idempotent setup script.

| Task | Actuator | Action | Action scale | Visual PPO result |
|---|---|---|---:|---|
| PickCuboid | position | cartesian_increment | 0.05 | Passed |
| PickCuboid | velocity | cartesian_increment | 0.05 | Passed |
| PickCuboid | position | joint_increment | 0.05 | Passed |
| PickCuboid | velocity | joint | 1.0 | Passed |
| PushCuboid | position | cartesian_increment | 0.05 | Passed |
| PushCuboid | velocity | cartesian_increment | 0.05 | Passed |
| PushCuboid | position | joint_increment | 0.05 | Passed |
| PushCuboid | velocity | joint | 1.0 | Passed |

## uv environment migration

The requirements and Conda manifests were replaced with `pyproject.toml` and a
single generated `uv.lock`. All previously pinned package versions were retained.
Fresh CPU and CUDA environments were independently created with `uv sync --locked`
and `uv sync --locked --extra cuda`; dependency consistency checks passed for both
(108 and 123 installed packages). The fresh CPU environment passed all 12
unit/native checks. The fresh CUDA environment imported Playground, initialized
JAX and Warp on the RTX 3090, and completed a synchronized JAX GPU computation.
Both Task setup variants were checked with `task --dry`. The full PPO matrix
above predates this manifest-only migration and was not rerun for it.

## Hardware and reproducibility

Tests used Python 3.12.3, the pinned dependencies in `pyproject.toml` and
`uv.lock`, NVIDIA driver 615.71.09, an RTX 3090 (24 GiB), and an RTX 3070 Ti (8 GiB). CUDA device ordering
is explicitly set to PCI bus order in the training entry point.

```bash
task setup MODE=cuda
task validate MODE=cuda DEVICE_ID=1
task validate MODE=cpu
```

Full-size simulator commands:

```bash
.venv/bin/python benchmark_simulation.py --env_name=PandaPickCuboid \
  --num_envs=1024 --device_id=1 --steps=32 --report=results/pick-warp-1024.json
.venv/bin/python benchmark_simulation.py --env_name=PandaPushCuboid \
  --num_envs=1024 --device_id=1 --steps=32 --report=results/push-warp-1024.json
```

Measured simulator throughput was approximately 3,480 Pick and 4,082 Push
**control steps across all environments per second**, with ten physics substeps
per control step and 64×64 pixel observations. These are functional benchmark
measurements taken while other validation jobs were active, not isolated peak
performance results or a controlled comparison against Madrona. Compilation and
warmup are excluded from those measurements.

## Limits and observed warnings

These checks establish pipeline execution, not learned task success. The full
15–20 million-step, ten-seed experiments and real-robot deployment have not been
run. Renderer/material changes and the fixes documented in README mean results
are a new simulator baseline, not numerical reproductions of old checkpoints.

The two-world Pick smoke tests produced very large KL metrics (roughly 2.5e8),
so they must not be used to assess optimization quality. In the larger Pick run
(128 worlds, 2,560 steps), KL was approximately 0.0267 and the losses were finite.
The normal-size pixels-only Push run reported KL approximately 0.000248. These
short runs still do not establish convergence or final policy quality.

The original five-iteration physics solver budget can emit an iteration-limit
warning in the larger Pick run. States remained finite; high-accuracy dynamics
or simulation-to-real agreement needs separate evaluation. Warp's line-search
budget was increased from eight to 32. JAXopt emits a maintenance deprecation,
and Orbax can emit a blocking-save performance warning; neither prevented the
completed host runs.

In this agent's sandbox, GPU devices are hidden and Orbax saving stalled. The GPU
and end-to-end checkpoint tests were therefore executed with approved host access.
This is an execution-sandbox limitation, not a reason to run normal user training
with elevated privileges.

## Taskfile migration

The repository-owned shell launchers are now represented in `Taskfile.yml`.
Validated with Task 3.54.0: all four benchmark expansions contain the original
20 runs in seed/action order, both setup modes parse, patch application remains
idempotent, debug help runs, and plotting expands to five commands. A command
recorder executed the validation workflow (pytest followed by eight smoke runs)
and a two-seed benchmark, checking CPU environment propagation and sequencing.
Invalid setup modes and the unsupported archived torque task fail explicitly.
These orchestration checks did not rerun the expensive training matrix above.
