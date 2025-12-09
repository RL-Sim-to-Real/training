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
import os

from brax.training.acme import running_statistics
# xla_flags = os.environ.get("XLA_FLAGS", "")
# xla_flags += " --xla_gpu_triton_gemm_any=True"
# os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"



from datetime import datetime
import functools
import json

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


# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

from mujoco_playground._src.manipulation.franka_emika_panda import randomize_vision_push as randomize

# save final policy params
import pickle
import shutil

_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "PandaPushCuboid",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
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

_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 50_000_000, "Number of timesteps"
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
    True,
    "Whether to include proprioception in the observation space.",
)

_DEVICE_ID = flags.DEFINE_integer("device_id", 0, "ID of the GPU device to use.")

def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv
  print("DEVICE ID:", _DEVICE_ID.value)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(_DEVICE_ID.value)
  print("JAX devices:", jax.devices())
  env_name = _ENV_NAME.value
  env_cfg = manipulation.get_default_config(env_name)

  num_envs = 1
  episode_length = int(10 / env_cfg.ctrl_dt)
  if _PROPRIOCEPTION.value:
    print("Using proprioception in the observation space.")
  # Rasterizer is less feature-complete than ray-tracing backend but stable
  config_overrides = {
      "episode_length": episode_length,
      "vision": True,
      "proprioception": True, 
      "obs_noise.brightness": [0.75, 2.0],
      "vision_config.use_rasterizer": False,
      "vision_config.render_batch_size": num_envs,
      "vision_config.render_width": 64,
      "vision_config.render_height": 64,
      "frame_stack_size": 3, # should be equivalent to basic case
      "box_init_range": 0.03, # +- 5 cm
      "action_history_length": 5,
      "success_threshold": 0.01,  
      "action_scale": 1.0, 
      "actuator": 'velocity',
      "action": 'joint',
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

  
  






  
   # === 1) Load trained params ===
  f_path = "/home/nika/Desktop/Research/training/logs/PandaPushCuboid-joint-velocity-_prop-seed-0/params_general_joint-velocity_prop.pkl"
  with open(f_path, "rb") as f:
    params = pickle.load(f)

  # === 2) Build the SAME network factory as in train_push_cube_ppo.py ===
  def make_network_factory(env_cfg):
    return functools.partial(
        ppo_networks_vision.make_ppo_networks_vision,
        policy_hidden_layer_sizes=[256, 256, 256],
        value_hidden_layer_sizes=[256, 256, 256],
        activation=linen.relu,
        normalise_channels=True,
        policy_obs_key="_prop" if env_cfg.proprioception else None,
        value_obs_key="_prop" if env_cfg.proprioception else None,
    )

  network_factory = make_network_factory(env_cfg)

  # === 3) Use the same PPO config to decide normalization ===
  ppo_params = manipulation_params.brax_vision_ppo_config(env_name)
  # (Optional but nice: keep num_envs consistent, in case config uses it)
  ppo_params.num_envs = num_envs
  ppo_params.num_eval_envs = num_envs

  if getattr(ppo_params, "normalize_observations", True):
    preprocess_fn = running_statistics.normalize
  else:
    preprocess_fn = None

  # === 4) Get the per-env obs shape like ppo.train does ===
  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)

  key_envs = jax.random.split(jax.random.PRNGKey(123), num_envs)
  init_state = jit_reset(key_envs)

  # Remove the leading batch dim to get per-env shapes
  obs_shape = jax.tree.map(lambda x: x.shape[1:], init_state.obs)
  print("Obs shape:", obs_shape)

  # === 5) Build the networks exactly as in training ===
  ppo_network = network_factory(
      obs_shape,
      env.action_size,
      preprocess_observations_fn=preprocess_fn,
  )

  # === 6) And build the inference fn just like ppo.train does ===
  make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
  jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

  ## Record video of final policy
  def unvmap(x):
    return jax.tree.map(lambda y: y[0], x)

  # Prepare for evaluation

  rng = jax.random.PRNGKey(123)
  rollout = []
  n_episodes = 2
  to_keep = 256

  def keep_until(state, i):
      return jax.tree.map(lambda x: x[:i], state)

  for episode in range(n_episodes):
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
    frames_wrist_camera = env.render([unvmap(s) for s in rollout][::render_every], camera="mounted")

    video_path = f"./rollout-{episode}.mp4"
    media.write_video(video_path, frames, fps=1.0 / env.dt / render_every)
    video_path = f"./rollout-wrist-camera-{episode}.mp4"
    media.write_video(video_path, frames_wrist_camera, fps=1.0 / env.dt / render_every)
    rollout = []
  print(f"Rollout video saved as '{video_path}'.")
  print("Rollout video saved as 'rollout.mp4'.")





if __name__ == "__main__":
  app.run(main)
