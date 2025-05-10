from datetime import datetime
import functools

from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from flax import linen
from IPython.display import clear_output
import jax
from matplotlib import pyplot as plt
import mediapy as media
import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=100)

from mujoco_playground import manipulation
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.franka_emika_panda import randomize_vision as randomize
from mujoco_playground.config import manipulation_params


import jax.numpy as jnp
from jax.tree_util import tree_map
import mujoco
import mujoco.viewer

env_name = "PandaPickCubeCartesian"
env_cfg = manipulation.get_default_config(env_name)

env_cfg.vision = False
num_envs = 1
episode_length = int(4 / env_cfg.ctrl_dt)

# Rasterizer is less feature-complete than ray-tracing backend but stable
config_overrides = {
    "obs_noise.brightness": [0.75, 2.0],
    "vision_config.use_rasterizer": True,
    "vision_config.render_batch_size": num_envs,
    "vision_config.render_width": 64,
    "vision_config.render_height": 64,
}

env = manipulation.load(env_name, config=env_cfg, 
                        config_overrides=config_overrides)


mj_model = env.mj_model
data = mujoco.MjData(mj_model)
# Set up a live view render
with mujoco.viewer.launch_passive(mj_model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(mj_model, data)
        viewer.sync()