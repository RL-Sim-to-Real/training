# @title Import MuJoCo, MJX, and Brax
import os
# On your second reading, load the compiled rendering backend to save time!
# os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "<YOUR_PATH>/madrona_mjx/build/cache"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Ensure that Madrona gets the chance to pre-allocate memory before Jax
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure CUDA is initialized correctly


from datetime import datetime
import functools


from flax import linen
from IPython.display import clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
import numpy as np


import cv2

from flax import serialization
import msgpack  # or use `flax.serialization.to_bytes`
from PIL import Image
# Deserialize using empty tree as template
from brax.training import types
import time


from multiprocessing import shared_memory, Process, set_start_method

try:
    set_start_method('spawn')  # Use 'spawn' to avoid issues with JAX and multiprocessing
except RuntimeError:
    pass






def agent_process(action_name, action_shape, action_dtype, 
                  image_name, image_shape, image_dtype):
    from brax.training.agents.ppo import networks_vision as ppo_networks_vision
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import manipulation
    from mujoco_playground import wrapper
    from mujoco_playground._src.manipulation.franka_emika_panda import randomize_vision as randomize
    from mujoco_playground.config import manipulation_params
    from get_policy_network import make_inference_fn #TODO: Figure this out
    import pickle

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


    # Load the params object from the pickle file
    with open("params.pkl", "rb") as f:
        params = pickle.load(f)

    inference_fn = make_inference_fn(network_factory=network_factory)

    jit_inference_fn = jax.jit(inference_fn(params, deterministic=True))
    action_shm = shared_memory.SharedMemory(name=action_name)
    action_array = np.ndarray(action_shape, dtype=action_dtype, buffer=action_shm.buf)
    image_shm = shared_memory.SharedMemory(name=image_name)
    img_array = np.ndarray(image_shape, dtype=image_dtype, buffer=image_shm.buf)
    key = jax.random.PRNGKey(0)
    try:
        while True:
            key, _ = jax.random.split(key)
            # start = time.time()
            action, _ = jit_inference_fn({'pixels/view_0': img_array.copy()}, key) # imperical inference time is 0.016
            # end = time.time()
            # print("Inference time:", end - start)
            time.sleep(0.034) # set the cycle time to 50 ms
            action_array[:] = action
    except KeyboardInterrupt:
        print("Agent process interrupted by user.")

    action_shm.close()
    image_shm.close()




if __name__ == "__main__":
    from franka_real.FrankaPickCubeCartesian import FrankaPickCubeCartesian
    env = FrankaPickCubeCartesian(camera_index=6)
    img, ee_pos,_ = env.reset()
   
    success_grasp = False
    input("Press Enter to start the control loop...")
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    action_shm = shared_memory.SharedMemory(create=True, size=action.nbytes)
    action_array = np.ndarray(buffer=action_shm.buf, dtype=np.float32, shape=action.shape)
    image_shm = shared_memory.SharedMemory(create=True, size=img.nbytes)
    image_array = np.ndarray(buffer=image_shm.buf, dtype=np.uint8, shape=img.shape)
    image_array[:] = img  # Copy the initial image to shared memory
    action_array[:] = action  # Copy the initial action to shared memory
    p = Process(target=agent_process, args=(action_shm.name, action.shape, action.dtype,
                                               image_shm.name, img.shape, img.dtype))
    p.start()
    while True:
        cv2.imshow("Captured Image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)  # Use 1 instead of 0 to avoid blocking
        
        # Resize the image using PIL
        action = action_array.copy()  # Copy the action from shared memory
        action_y_z = 0.02 * action[:2] # this is the increment
        if (action[2] < -0.2 and not success_grasp): # grasp it only once
            success_grasp = env.grasp_object()
        target_y_z = action_y_z + ee_pos[1:3] # this is the target position
        target_y_z = jp.array([jp.clip(target_y_z[0], -0.1, 0.1), jp.clip(target_y_z[1], 0.03, 0.2)]) # for safety
        target_x_y_z = jp.concatenate([jp.array([0.57]), target_y_z])
        start = time.time()
        img, ee_pos = env.step(target_x_y_z)
        end = time.time()
        print(f"Time taken for one step: {end - start:.3f} seconds")
        image_array[:] = img

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if success_grasp and ee_pos[2] > 0.1:
            print("Trial complete")
            break
    # time.sleep(2)
    ## slowly lower z value to place down
    target_x_y_z = jp.array([ee_pos[0], ee_pos[1], 0.035])  # Keep x, y the same and set z to 0.02
    env.step(target_x_y_z)
    env.open_gripper()
    env.close()
    p.join()  # Wait for the agent process to finish
    action_shm.close()
    action_shm.unlink()  # Unlink the shared memory
    image_shm.close()
    image_shm.unlink()  # Unlink the shared memory

    cv2.destroyAllWindows()