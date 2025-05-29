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

# Rasterizer is less feature-complete than ray-tracing backend but stable


network_factory = functools.partial(
    ppo_networks_vision.make_ppo_networks_vision,
    policy_hidden_layer_sizes=[256, 256],
    value_hidden_layer_sizes= [256, 256],
    # activation=linen.relu, # only works with default activation right now
    normalise_channels=True
)

ppo_params = manipulation_params.brax_vision_ppo_config(env_name)

del ppo_params.network_factory
ppo_params.network_factory = network_factory

import pickle

# Load the params object from the pickle file
with open("params.pkl", "rb") as f:
    params = pickle.load(f)




inference_fn = make_inference_fn(network_factory=network_factory)



with open("policy_params.msgpack", "rb") as f:
    param_bytes = f.read()

# Deserialize using empty tree as template
from brax.training import types
dummy_params = params  # or create using network_factory()
loaded_params = serialization.from_bytes(dummy_params, param_bytes)

jit_inference_fn = jax.jit(inference_fn(loaded_params, deterministic=True))


from PIL import Image

# Loop through the images
for i in range(1, 5):
    # Load the image
    image_path = f"test_images/franka-{i}.jpg"
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
    # print(f"Control output for franka-{i}.jpg:", ctrl)
    # Save the control output to a text file
    output_path = "control_outputs.txt"
    with open(output_path, "a") as output_file:
        output_file.write(f"Control output for franka-{i}.jpg: {ctrl}\n")