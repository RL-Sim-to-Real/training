import os
# On your second reading, load the compiled rendering backend to save time!
# os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "<YOUR_PATH>/madrona_mjx/build/cache"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # Ensure that Madrona gets the chance to pre-allocate memory before Jax
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest" # need this to improve reproducability
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# @title Import MuJoCo, MJX, and Brax
from datetime import datetime, time
import functools

from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from flax import linen
from IPython.display import clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
import numpy as np

from mujoco_playground import manipulation
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.franka_emika_panda import randomize_vision_modified as randomize
from mujoco_playground.config import manipulation_params

import pickle # pickle works better
from flax import serialization
import msgpack  # or use `flax.serialization.to_bytes`
np.set_printoptions(precision=3, suppress=True, linewidth=100)


env_name = "PandaPickCubeCartesianModified"
env_cfg = manipulation.get_default_config(env_name)

num_envs = 1024
episode_length = int(4 / env_cfg.ctrl_dt)

# Rasterizer is less feature-complete than ray-tracing backend but stable
config_overrides = {
    "episode_length": episode_length,
    "vision": True,
    "proprioception": False, 
    "obs_noise.brightness": [0.75, 2.0],
    "vision_config.use_rasterizer": False,
    "vision_config.render_batch_size": num_envs,
    "vision_config.render_width": 64,
    "vision_config.render_height": 64,
    "box_init_range": 0.1, # +- 10 cm
    "action_history_length": 5,
    "success_threshold": 0.03,
    "action_scale": 0.02, # 5 cm,
    "actuator": "position",
    "action": "cartesian_increment",
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
ppo_params.num_timesteps = 50_000_000
ppo_params.num_envs = num_envs
ppo_params.num_eval_envs = num_envs
del ppo_params.network_factory
ppo_params.network_factory = network_factory



x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


  # Progress function for logging
def progress(num_steps, metrics):
        
    if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
            )


train_fn = functools.partial(
    ppo.train,
    augment_pixels=True,
    **dict(ppo_params),
    progress_fn=progress,

)

make_inference_fn, params, metrics = train_fn(environment=env)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")