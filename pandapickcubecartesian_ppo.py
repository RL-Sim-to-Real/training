

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




env_name = "PandaPickCubeCartesian"
env_cfg = manipulation.get_default_config(env_name)

env_cfg.vision = False
num_envs = 1
episode_length = int(4 / env_cfg.ctrl_dt)

# Rasterizer is less feature-complete than ray-tracing backend but stable
config_overrides = {
    "obs_noise.brightness": [0.75, 2.0],
    "vision_config.use_rasterizer": False,
    "vision_config.render_batch_size": num_envs,
    "vision_config.render_width": 64,
    "vision_config.render_height": 64,
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




jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

def tile(img, d):
    assert img.shape[0] == d*d
    img = img.reshape((d,d)+img.shape[1:])
    return np.concatenate(np.concatenate(img, axis=1), axis=1)

def unvmap(x):
    return jax.tree.map(lambda y: y[0], x)




rng = jax.random.PRNGKey(0)
state = jit_reset(jax.random.split(rng, num_envs))
rollout = [unvmap(state)]

f = 0.2
for i in range(env_cfg.episode_length):
  action = []
#   for j in range(env.action_size):
#     action.append(
#         jnp.sin(
#             unvmap(state.data.time) * 2 * jnp.pi * f + j * 2 * jnp.pi / env.action_size
#         )
#     )
#   action = jnp.tile(jnp.array(action), (num_envs, 1))
  jit_reset(jax.random.split(rng, num_envs))
  rollout.append(unvmap(state))

frames = env.render(rollout)
media.show_video(frames, fps=1.0 / env.dt)