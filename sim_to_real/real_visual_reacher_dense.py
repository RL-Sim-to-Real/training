# @title Import MuJoCo, MJX, and Brax
import os
# On your second reading, load the compiled rendering backend to save time!
# os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "<YOUR_PATH>/madrona_mjx/build/cache"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # Ensure that Madrona gets the chance to pre-allocate memory before Jax


from datetime import datetime
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
from mujoco_playground._src.manipulation.franka_emika_panda import randomize_vision as randomize
from mujoco_playground.config import manipulation_params
from get_policy_network import make_inference_fn #TODO: Figure this out

from flax import serialization
import msgpack  # or use `flax.serialization.to_bytes`

np.set_printoptions(precision=3, suppress=True, linewidth=100)

env_name = "PandaPickCubeCartesian"
env_cfg = manipulation.get_default_config(env_name)

num_envs = 1024
episode_length = int(4 / env_cfg.ctrl_dt)

# Rasterizer is less feature-complete than ray-tracing backend but stable
config_overrides = {
    "episode_length": episode_length,
    "vision": True,
    "obs_noise.brightness": [0.75, 2.0],
    "vision_config.use_rasterizer": False,
    "vision_config.render_batch_size": num_envs,
    "vision_config.render_width": 64,
    "vision_config.render_height": 64,
    "box_init_range": 0.1, # +- 10 cm
    "action_history_length": 5,
    "success_threshold": 0.03
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
    value_hidden_layer_sizes= [256, 256],
    # activation=linen.relu, # only works with default activation right now
    normalise_channels=True
)

ppo_params = manipulation_params.brax_vision_ppo_config(env_name)
ppo_params.num_timesteps = 1
ppo_params.num_envs = num_envs
ppo_params.num_eval_envs = num_envs
del ppo_params.network_factory
ppo_params.network_factory = network_factory



x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
  clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  steps = ppo_params["num_timesteps"]
  plt.xlim([steps * -0.1, steps * 1.25])
  plt.ylim([0, 14])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

#   display(plt.gcf())


train_fn = functools.partial(
    ppo.train,
    augment_pixels=True,
    **dict(ppo_params),
    progress_fn=progress,

)

make_inference_fn, params, metrics = train_fn(environment=env)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")


with open("policy_params.msgpack", "rb") as f:
    param_bytes = f.read()

# Deserialize using empty tree as template
from brax.training import types
dummy_params = params  # or create using network_factory()
loaded_params = serialization.from_bytes(dummy_params, param_bytes)

jit_inference_fn = jax.jit(make_inference_fn(loaded_params, deterministic=True))


from PIL import Image

# Loop through the images
for i in range(1, 5):
    # Load the image
    image_path = f"sim_to_real/test_images/franka-{i}.jpg"
    image = Image.open(image_path).resize((64, 64))
    
    # Convert to numpy array and cast to (64, 64, 3)
    image_array = np.array(image, dtype=np.uint8)
    
    # Ensure the image has 3 channels
    if image_array.shape[-1] != 3:
        raise ValueError(f"Image {image_path} does not have 3 channels.")
    
    # Prepare the observation dictionary
    obs = {'pixels/view_0': image_array}
    
    # Perform inference
    ctrl, _ = jit_inference_fn(obs, jax.random.PRNGKey(0))
    print(f"Control output for franka-{i}.jpg:", ctrl)