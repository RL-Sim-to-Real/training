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
import cv2

from flax import serialization
import msgpack  # or use `flax.serialization.to_bytes`
from franka_real.FrankaPickCubeCartesian import FrankaPickCubeCartesian
from PIL import Image
# Deserialize using empty tree as template
from brax.training import types
import time


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



dummy_params = params  # or create using network_factory()
loaded_params = serialization.from_bytes(dummy_params, param_bytes)

jit_inference_fn = jax.jit(inference_fn(loaded_params, deterministic=True))




# Loop through the images
# for i in range(1, 5):
#     # Load the image
#     image_path = f"test_images/franka-{i}.jpg"
#     image = Image.open(image_path).resize((64, 64))
    
#     # Convert to numpy array and cast to (64, 64, 3)
#     image_array = np.array(image, dtype=np.uint8)
    
#     # Ensure the image has 3 channels
#     if image_array.shape[-1] != 3:
#         raise ValueError(f"Image {image_path} does not have 3 channels.")
    
#     # Prepare the observation dictionary
#     obs = {'pixels/view_0': image_array}
    
#     # Perform inference
#     ctrl, _ = jit_inference_fn(obs, jax.random.PRNGKey(0))
#     # print(f"Control output for franka-{i}.jpg:", ctrl)
#     # Save the control output to a text file
#     output_path = "control_outputs.txt"
#     with open(output_path, "a") as output_file:
#         output_file.write(f"Control output for franka-{i}.jpg: {ctrl}\n")


if __name__ == "__main__":
    env = FrankaPickCubeCartesian(camera_index=6)
    img, ee_pos,_ = env.reset()
    key = jax.random.PRNGKey(0)
    success_grasp = False
    input("Press Enter to start the control loop...")
    
    while True:
        # cv2.imshow("Captured Image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(1)  # Use 1 instead of 0 to avoid blocking
        new_key, key = jax.random.split(key)
        # Resize the image using PIL

        # img = np.array(img)
        action, _ = jit_inference_fn({'pixels/view_0': img}, key) # key value added for compatibilitys
        print(action)
        action_y_z = 0.1 * action[:2] # this is the increment
        # # print("Increment YZ:", action_y_z)

        # if (action[2] < -0.6):
        #     success_grasp = env.grasp_object()
        target_y_z = action_y_z + ee_pos[1:3] # this is the target position
        target_y_z = jp.array([jp.clip(target_y_z[0], -0.2, 0.2), jp.clip(target_y_z[1], 0.03, 0.2)]) # for safety
        # # print(f"target YZ: {target_y_z}")
        # # break
        target_x_y_z = jp.concatenate([jp.array([0.57]), target_y_z])
        img, ee_pos = env.step(target_x_y_z)
        print(f"End Effector Position: {ee_pos}")
        # time.sleep(0.02)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # if success_grasp and ee_pos[2] > 0.1:
        #     print("Trial complete")
        #     break
    time.sleep(5)
    env.close()
    cv2.destroyAllWindows()