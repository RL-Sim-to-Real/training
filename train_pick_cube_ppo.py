# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using JAX on the specified environment."""

from datetime import datetime
import functools
import json
import os
import time
import warnings

from tqdm import tqdm

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
from flax import linen
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
from orbax import checkpoint as ocp
from tensorboardX import SummaryWriter
import wandb

# mujoco_playground imports
import mujoco_playground
from mujoco_playground import manipulation
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

from mujoco_playground._src.manipulation.franka_emika_panda import randomize_vision as randomize


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "PandaPickCubeCartesianModified",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 50_000_000, "Number of timesteps"
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [256, 256],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [256, 256],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    5_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)

_ACTUATOR = flags.DEFINE_string("actuator", "torque", "Type of actuator to use.")
_ACTION_SPACE = flags.DEFINE_string(
    "action",
    "joint",
    "Type of action space to use. One of ['cartesian_increment', 'joint_increment']",
)

_ACTION_SCALE = flags.DEFINE_float(
    "action_scale",
    0.02,
    "Scale factor of the action space.",
)

_PROPRIOCEPTION = flags.DEFINE_boolean(
    "proprioception",
    False,
    "Whether to include proprioception in the observation space.",
)

_DEVICE_ID = flags.DEFINE_integer("device_id", 0, "ID of the GPU device to use.")

def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv
  os.environ["CUDA_VISIBLE_DEVICES"] = str(_DEVICE_ID.value)
  print("JAX devices:", jax.devices())
  env_name = _ENV_NAME.value
  env_cfg = manipulation.get_default_config(env_name)

  num_envs = 1024
  episode_length = int(4 / env_cfg.ctrl_dt)
  if _PROPRIOCEPTION.value:
    print("Using proprioception in the observation space.")
  # Rasterizer is less feature-complete than ray-tracing backend but stable
  config_overrides = {
      "episode_length": episode_length,
      "vision": True,
      "proprioception": _PROPRIOCEPTION.value, 
      "obs_noise.brightness": [0.75, 2.0],
      "vision_config.use_rasterizer": False,
      "vision_config.render_batch_size": num_envs,
      "vision_config.render_width": 64,
      "vision_config.render_height": 64,
      "box_init_range": 0.1, # +- 10 cm
      "action_history_length": 5,
      "success_threshold": 0.03,
      "action_scale": _ACTION_SCALE.value, 
      "actuator": _ACTUATOR.value,
      "action": _ACTION_SPACE.value,
  }

  env = manipulation.load(env_name, config=env_cfg,
                          config_overrides=config_overrides
  )
  randomization_fn = functools.partial(randomize.domain_randomize,
                                          num_worlds=num_envs
  )
  env = wrapper.wrap_for_brax_training(
      env,
      vision=True,
      num_vision_envs=num_envs,
      episode_length=episode_length,
      action_repeat=1,
      randomization_fn=randomization_fn
  )
  network_factory = functools.partial(
      ppo_networks_vision.make_ppo_networks_vision,
      policy_hidden_layer_sizes=[256, 256],
      value_hidden_layer_sizes=[256, 256],
      # activation=linen.relu, # only works with default activation right now
      normalise_channels=True,
      policy_obs_key="_prop" if env_cfg.proprioception else None, # determine wether to use proprioception
      value_obs_key="_prop" if env_cfg.proprioception else None,
  )

  ppo_params = manipulation_params.brax_vision_ppo_config(env_name)
  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  ppo_params.num_envs = num_envs
  ppo_params.num_eval_envs = num_envs
  del ppo_params.network_factory
  ppo_params.network_factory = network_factory

  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value
  
  

  print(f"Environment Config:\n{env_cfg}")
  print(f"PPO Training Parameters:\n{ppo_params}")

  # Generate unique experiment name
  now = datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{env_name}-{_ACTION_SPACE.value}-{_ACTUATOR.value}"
  if _PROPRIOCEPTION.value:
    exp_name += "-_prop"
  exp_name += f"-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    wandb.init(project="mjxrl", entity="dextrm", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": env_name})

  # Initialize TensorBoard if required
  if _USE_TB.value and not _PLAY_ONLY.value:
    writer = SummaryWriter(logdir)

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)
  
  # save seed value to path
  with open(ckpt_path / "seed.txt", "w", encoding="utf-8") as fp:
    fp.write(str(_SEED.value))
    
  times = [time.monotonic()]

  # Progress function for logging
  def progress(num_steps, metrics):
    times.append(time.monotonic())
    
    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if _RUN_EVALS.value:
      if "eval/episode_reward" in metrics:
        print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    if _LOG_TRAINING_METRICS.value:
      
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )
    
  train_fn = functools.partial(
      ppo.train,
      seed=_SEED.value,
      augment_pixels=True,
      **dict(ppo_params),
      progress_fn=progress,

  )




  # Train or load the model
  make_inference_fn, params, metrics = train_fn(environment=env)

  # save final policy params
  import pickle
  with open(logdir / f"params_general_{_ACTION_SPACE.value}-{_ACTUATOR.value}_img_aug.pkl", "wb") as f:
    pickle.dump(params, f)
  print("Done training.")


  ## Record video of final policy
  def unvmap(x):
    return jax.tree.map(lambda y: y[0], x)



  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)
  jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

  # Prepare for evaluation

  rng = jax.random.PRNGKey(123)
  rollout = []
  n_episodes = 1
  to_keep = 256

  def keep_until(state, i):
      return jax.tree.map(lambda x: x[:i], state)

  for _ in range(n_episodes):
      key_rng = jax.random.split(rng, num_envs)
      state = jit_reset(key_rng)
      rollout.append(keep_until(state, to_keep))
      for i in tqdm(range(env_cfg.episode_length)):
          act_rng, rng = jax.random.split(rng)
          act_rng = jax.random.split(act_rng, num_envs)
          ctrl, _ = jit_inference_fn(state.obs, act_rng)
          state = jit_step(state, ctrl)
          rollout.append(keep_until(state, to_keep))

  render_every = 1
  frames = env.render([unvmap(s) for s in rollout][::render_every])

  video_path = logdir / "rollout.mp4"
  media.write_video(video_path, frames, fps=1.0 / env.dt / render_every)
  print(f"Rollout video saved as '{video_path}'.")
  print("Rollout video saved as 'rollout.mp4'.")





if __name__ == "__main__":
  app.run(main)
