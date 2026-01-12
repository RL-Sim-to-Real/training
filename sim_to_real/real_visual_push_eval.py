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
from camera import Camera


from flax import serialization
import msgpack  # or use `flax.serialization.to_bytes`
from PIL import Image
# Deserialize using empty tree as template
from brax.training import types
import time

from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from mujoco_playground import manipulation
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.franka_emika_panda import randomize_vision as randomize
from mujoco_playground.config import manipulation_params
from get_policy_network_push import make_inference_fn #TODO: Figure this out
import pickle

import json
from multiprocessing import shared_memory, Process, set_start_method
import threading
import pyrealsense2 as rs
from pathlib import Path
from typing import Dict, Tuple

from franka_real.FrankaPushCube import FrankaPushCube
from franka_real.FrankaEvalAutomator import FrankaEvalAutomator

try:
    set_start_method('spawn')  # Use 'spawn' to avoid issues with JAX and multiprocessing
except RuntimeError:
    pass




depth_frame, color_frame = None, None
lock = threading.Lock()

class Agent():
    def __init__(self, control_mode, use_prop, action_name, action_shape, action_dtype, image_name, image_shape, image_dtype):
        np.set_printoptions(precision=3, suppress=True, linewidth=100)
        self.control_mode = control_mode
        self.use_prop = use_prop

        env_name = "PandaPushCuboid"

        # Rasterizer is less feature-complete than ray-tracing backend but stable
        layer_size = 256

        network_factory = functools.partial(
            ppo_networks_vision.make_ppo_networks_vision,
            policy_hidden_layer_sizes=[layer_size, layer_size, layer_size],
            value_hidden_layer_sizes= [layer_size, layer_size, layer_size],
            activation=linen.relu, # only works with default activation right now
            normalise_channels=True,
        )

        ppo_params = manipulation_params.brax_vision_ppo_config(env_name)

        del ppo_params.network_factory
        ppo_params.network_factory = network_factory

        policy_fn = {
            'cartesian_position': "thesis_policies/push/params_general_cartesian_increment-position_prop_seed0.pkl",
            'cartesian_velocity': "thesis_policies/push/params_general_cartesian_increment-velocity_prop_seed9.pkl",
            'joint_position': "thesis_policies/push/params_general_joint_increment-position_prop_seed7.pkl",
            'joint_velocity': "thesis_policies/push/params_general_joint-velocity_prop_seed1.pkl",
        }[control_mode]

        with open(policy_fn, "rb") as f:
            params = pickle.load(f)

        self.action_shape = (7,)
        norm_obs_dict = {
            'cartesian_position': True,
            'cartesian_velocity': True,
            'joint_position': True,
            'joint_velocity': True,
        }
        inference_fn = make_inference_fn(
            normalize_observations=norm_obs_dict[control_mode],
            action_size=self.action_shape[0],
            network_factory=network_factory,
            include_prop=True,
        )   

        self.jit_inference_fn = jax.jit(inference_fn(params, deterministic=True))
        # self.action_shm = shared_memory.SharedMemory(name=action_name)
        # self.action_array = np.ndarray(self.action_shape, dtype=action_dtype, buffer=self.action_shm.buf)
        self.image_shm = shared_memory.SharedMemory(name=image_name)
        self.img_array = np.ndarray(image_shape, dtype=image_dtype, buffer=self.image_shm.buf)
        self.key = jax.random.PRNGKey(0)

    def get_action(self, proprioception=None):
        self.key, _ = jax.random.split(self.key)
        obs = {'pixels/view_0': self.img_array.copy() / 255.0}
        if self.use_prop:
            obs['_prop'] = proprioception
        t0 = time.time()
        action, _ = self.jit_inference_fn(obs, self.key) # empirical inference time is 0.016
        print(f"Inference time: {(time.time() - t0) * 1000.:.3f} ms")
        # self.action_array[:] = action
        return action

    def __del__(self):
        # self.action_shm.close()
        # self.action_shm.unlink()
        self.image_shm.close()
        self.image_shm.unlink()



def camera_process(image_name, image_shape, image_dtype):
    camera = Camera(cam_index=4)
    # FIXME: set the frame dimensions to be square to match training
    image_shm = shared_memory.SharedMemory(name=image_name)
    img_array = np.ndarray(image_shape, dtype=image_dtype, buffer=image_shm.buf)
    while True:

        img = camera.capture_img()
        cv2.imshow("Captured Image", img)
        cv2.waitKey(1)  # Use 1 instead of 0 to avoid blocking
        img = cv2.resize(img, (64, 64)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB 
        img_array[:] = img  # Copy the image to shared memory
        # print(f"Time taken to capture and process image: {end_time - start_time:.3f} seconds")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def rs_camera_process(image_name, image_shape, image_dtype, pipeline, align):
    image_shm = shared_memory.SharedMemory(name=image_name)
    img_array = np.ndarray(image_shape, dtype=image_dtype, buffer=image_shm.buf)
    while True:
        # FIXME: put this in a thread in the main process so that frame objects are available in the main process
        # Get frames
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            time.sleep(0.001)
            continue

        img = np.asanyarray(color_frame.get_data())

        cv2.imshow("Captured Image", img)
        cv2.waitKey(1)  # Use 1 instead of 0 to avoid blocking
        img = cv2.resize(img, (64, 64)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB 
        img_array[:] = img  # Copy the image to shared memory
        # print(f"Time taken to capture and process image: {end_time - start_time:.3f} seconds")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def rs_camera_thread(img_array):
    global depth_frame, color_frame, lock

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        cf = frames.get_color_frame().keep()
        df = frames.get_depth_frame().keep()
        if not color_frame or not depth_frame:
            time.sleep(0.001)
            continue

        with lock:
            color_frame = cf
            depth_frame = df

        img = np.asanyarray(color_frame.get_data())

        cv2.imshow("Captured Image", img)
        print("Showing image")
        cv2.waitKey(1)  # Use 1 instead of 0 to avoid blocking
        img = cv2.resize(img, (64, 64)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB 
        img_array[:] = img  # Copy the image to shared memory
        # print(f"Time taken to capture and process image: {end_time - start_time:.3f} seconds")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def make_homog(Rmat, tvec):
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = tvec.flatten()
    return T

def get_birds_eye_view(point_cam_array, env):
    # move the robot above the cube
    # height of 30cm or more works good
    for _ in range(5):
        point_base, point_ee = get_point_base(point_cam_array.copy(), env)
        angle = point_cam_array[4]
        angle_180 = point_cam_array[5]
        is_square = point_cam_array[6]
        print(f"Cube position (x, y, z) in robot base frame: {point_base}")
        pose_ee = point_base[:3].copy()
        pose_ee[2] = 0.3
        env.move_to_pose_ee(pose_ee)
        # time.sleep(1)
    return angle, angle_180, is_square, pose_ee


def put_cube_on_white_strip(env, angle, pose_ee):
    ee_angle = np.array(env.euler_from_quaternion(env.reset_ee_quaternion))
    ee_angle[2] += -(angle if angle < 45 else angle - 90) * np.pi / 180.0
    grasp_height = 0.05
    pose_ee[2] = grasp_height
    pose_ee[0] -= 0.015 # offset to be above the cube center
    env.move_to_pose_ee(pose_ee, ref_ee_angle=ee_angle)
    env.grasp_object()
    # time.sleep(0.25)
    pose_ee[2] = 0.3
    # apply a small random yaw offset to the end-effector before moving away
    env.move_to_pose_ee(pose_ee)
    
    # randomize the cube position8
    cube_pos = np.array([np.random.uniform(0.52, 0.62), np.random.uniform(-0.1, 0.1), grasp_height + 0.01])
    
    print(f"Moving cube to new position: {cube_pos[:2]}")

    env.move_to_pose_ee(cube_pos)
    cube_pos[2] = grasp_height + 0.0005
    yaw_offset_deg = np.random.uniform(-45.0, 45.0)
    yaw_offset = np.deg2rad(yaw_offset_deg)
    ee_angle = np.array(env.euler_from_quaternion(env.reset_ee_quaternion))
    random_ee_angle = ee_angle.copy()
    random_ee_angle[2] += yaw_offset
    env.move_to_pose_ee(cube_pos, ref_ee_angle=random_ee_angle)
    # time.sleep(1)
    env.open_gripper()
    # time.sleep(0.25)
    pose_ee[:2] = cube_pos[:2]
    pose_ee[2] = 0.2
    env.move_to_pose_ee(pose_ee)
    # time.sleep(1) 


def get_point_base(point_cam, env):
    # Get robot pose
    ee_pose = env.robot.endpoint_pose()  # has .translation and .quaternion
    ee_quaternion = [ee_pose['orientation'].w, ee_pose['orientation'].x,
                        ee_pose['orientation'].y, ee_pose['orientation'].z]
    # R_ee = R.from_quat(ee_pose.quaternion).as_matrix()
    R_ee = env.matrix_from_quaternion(ee_quaternion)
    T_base_ee = make_homog(R_ee, ee_pose['position'])
    # print('ee_pose:', ee_pose['position'], env.euler_from_quaternion(ee_quaternion))
    T_ee_camera_file = os.path.expanduser("~/Desktop/ICRA2026/Franka-Real/franka_real/T_ee_camera.json")
    with open(T_ee_camera_file, "r") as f:
        T_ee_camera = np.array(json.load(f))
    T_base_camera = T_base_ee @ T_ee_camera

    point_cam_homog = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])

    point_ee = T_ee_camera @ point_cam_homog
    point_base = T_base_camera @ point_cam_homog
    return point_base, point_ee

def reset_cube_position(point_cam_array, env, target_joints):
    angle, angle_180, is_square, pose_ee = get_birds_eye_view(point_cam_array, env)
    # while is_square < 0.5: # don't need this for this task
    #     print("Cube is knocked over")
    #     make_cube_upright(env, pose_ee, angle, angle_180)
    #     env.move_to_joint_positions(target_joints)
    #     env.apply_joint_vel(np.zeros((7,)))
    #     angle, angle_180, is_square, pose_ee = get_birds_eye_view(point_cam_array, env)

    put_cube_on_white_strip(env, angle, pose_ee)

def has_contact(img: np.ndarray) -> bool:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # red wraps around hue, so use two ranges
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    cube_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # clean up
    kernel = np.ones((3, 3), np.uint8)
    cube_mask = cv2.morphologyEx(cube_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cube_mask = cv2.morphologyEx(cube_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    h, w = img.shape[:2]
    roi = img[int(h*0.6):h, :]  # bottom 40% only

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # broad threshold for “not red, not background wood”
    # tweak these by eye
    lower_gray = np.array([0, 0, 80])
    upper_gray = np.array([180, 60, 255])
    fingers_mask_roi = cv2.inRange(hsv_roi, lower_gray, upper_gray)

    fingers_mask_roi = cv2.morphologyEx(fingers_mask_roi, cv2.MORPH_OPEN, kernel, iterations=2)
    fingers_mask = np.zeros_like(cube_mask)
    fingers_mask[int(h*0.6):h, :] = fingers_mask_roi

    # dilate cube a tiny bit so “almost touching” counts as touching
    dilated_cube = cv2.dilate(cube_mask, kernel, iterations=1)

    overlap = cv2.bitwise_and(dilated_cube, fingers_mask)
    touching = np.any(overlap > 0)
    print(f"Touching: {touching}")
    return touching



def cube_in_bottom_half(img: np.ndarray) -> Dict:
    """
    Detects a red cube and checks if its centroid lies in the bottom half of the image.
    Returns a dict with centroid, image size, and a boolean flag.
    """

    h, w = img.shape[:2]
    
    # HSV threshold for red (two ranges due to hue wrap-around)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # WARNING: MAKE SURE IMAGE IS BGR
    lower1, upper1 = np.array([0, 90, 90]),  np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 90, 90]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # Clean up mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    # Largest red contour → cube
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False  # No cube detected
    cube = max(cnts, key=cv2.contourArea)

    # Centroid
    M = cv2.moments(cube)
    cx = int(M['m10'] / (M['m00'] + 1e-6))
    cy = int(M['m01'] / (M['m00'] + 1e-6))

    # Bottom-half test
    in_bottom = cy >= h // 2
    return in_bottom

def run_trials(max_trials, 
               action_name, 
               action_shape, 
               action_dtype, 
               point_cam_name, 
               point_cam_shape, 
               point_cam_dtype, image_name, image_shape, image_dtype, cube_in_position_name, cube_in_position_shape, cube_in_position_dtype):
    save_logs = True
    control_mode = [
        'cartesian_position',
        'joint_position',
        'joint_velocity',
        'cartesian_velocity',
    ][0]

    use_prop = True

    env = FrankaPushCube(camera_index=0, control_mode=control_mode, seed=0)
    agent = Agent(control_mode, use_prop, action_name, action_shape, action_dtype, image_name, image_shape, image_dtype)

    # reset the cube position
    point_cam_shm = shared_memory.SharedMemory(name=point_cam_name)
    point_cam_array = np.ndarray(point_cam_shape, dtype=point_cam_dtype, buffer=point_cam_shm.buf)

    cube_in_position_shm = shared_memory.SharedMemory(name=cube_in_position_name)
    cube_in_position_array = np.ndarray(cube_in_position_shape, dtype=cube_in_position_dtype, buffer=cube_in_position_shm.buf)

    # reset the robot joints to initial position
    target_joints = np.array([-0.01266706, 0.23113158, 0.01397337, -2.11847885, -0.00837887, 2.33090511, 0.80890272])
    # target_joints = np.array([-0.00002, 0.47804, -0.00055, -1.81309, -0.00161, 2.34597, 0.78501])
    env.move_to_joint_positions(target_joints)
    ee_pos,_ = env.reset()
    trial_length = 12
    skip_to_trial = 0
    for i in range(max_trials):
        cube_in_position_array[0] = 0
        if i < skip_to_trial:
            np.array([np.random.uniform(0.52, 0.62), np.random.uniform(-0.1, 0.1), 0 + 0.01])
            continue
        env.move_to_joint_positions(target_joints)
        env.open_gripper()
        env.apply_joint_vel(np.zeros((7,)))
        print("Resetting cube position...")
        reset_cube_position(point_cam_array, env, target_joints)

        # reset the robot joints to initial position again
        env.move_to_joint_positions(target_joints)
        env.apply_joint_vel(np.zeros((7,)))

        success = False
        env.grasped = False
        ee_pos,_ = env.reset()
        t_start = time.time()


        action = np.zeros(agent.action_shape, dtype=np.float32)
        env.logger.clear()
        init_position_ee = ee_pos.copy()
        aggregate_displacement = 0.0
       
        while True:
            # action = action_array.copy()  # Copy the action from shared memory
            proprioception = None
            if agent.use_prop:
                obs = env.get_state()
                joint_p = obs['joints']
                normalized_jp = 2*(joint_p - jp.array(_jnt_range())[:, 0]) / (
                    jp.array(_jnt_range())[:, 1] - jp.array(_jnt_range())[:, 0]
                ) - 1.0
                joint_v = obs['joint_vels']

                normalized_jv = 2*(joint_v - jp.array(_jnt_vel_range())[:, 0]) / (
                    jp.array(_jnt_vel_range())[:, 1] - jp.array(_jnt_vel_range())[:, 0]
                ) - 1.0

                ee_height = obs['height']
                
                proprioception = np.concatenate([ 
                    normalized_jp, 
                    normalized_jv,  # Include joint velocities in proprioception
                    action, ee_height], axis=-1).astype(np.float32)
                print(f"Proprioception: {proprioception.shape}, {proprioception}")
            action = agent.get_action(proprioception=proprioception)
            
            print(f"Action: {action}")


            ee_pos = env.step(action)
  
            print("cube_in_position:", cube_in_position_array[0])
            print("ee height:", ee_pos[2])
            displacement = np.linalg.norm(ee_pos[:2] - init_position_ee[:2])
            print("displacement:", displacement)
            aggregate_displacement += displacement * cube_in_position_array[0] * (ee_pos[2] < 0.0557) * (displacement > 0.003)
            init_position_ee = ee_pos.copy()
        
            if time.time() - t_start > trial_length:
                print(f"---- Trial {i}: Timeout")
                break
            if ee_pos[0] < 0.27 or ee_pos[0] > 0.8 or ee_pos[1] > 0.35 or ee_pos[1] < -0.35 or ee_pos[2] > 0.5:
                print(f"---- Trial {i}: Robot moved out of workspace")
                break

        print("Aggregate displacement:", aggregate_displacement)
        env.logger.metrics[-1]['aggregate displacement'] = aggregate_displacement


        # save logs
        if save_logs:
            fp = Path(f'real_franka_eval_logs/push/{control_mode}_use_prop_{use_prop}_trial_{i}.pkl.csv')
            fp.parent.mkdir(parents=True, exist_ok=True)
            env.logger.save(fp)

    env.reset()
    env.close()

    return success

def prepare_realsense(fps):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align

def _jnt_range():
    # TODO(siholt): Use joint limits from XML.
    return [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ]


def _jnt_vel_range():
    return [
        [-2.1750, 2.1750],
        [-2.1750, 2.1750],
        [-2.1750, 2.1750],
        [-2.1750, 2.1750],
        [-2.6100, 2.6100],
        [-2.6100, 2.6100],
        [-2.6100, 2.6100],
    ]


def main():
    record_video = False
    if record_video:
        video, video_ts = [], []
        ext_cam = Camera(cam_index=6)
        
    # env = FrankaPickCubeCartesian(camera_index=0)
    eval_automator = FrankaEvalAutomator()

    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8) * 255  # Dummy image for initialization

    action = np.zeros((7,), dtype=np.float32)
    action_shm = shared_memory.SharedMemory(create=True, size=action.nbytes)
    action_array = np.ndarray(buffer=action_shm.buf, dtype=np.float32, shape=action.shape)
    image_shm = shared_memory.SharedMemory(create=True, size=dummy_img.nbytes)
    image_array = np.ndarray(buffer=image_shm.buf, dtype=np.uint8, shape=dummy_img.shape)
    point_cam = np.zeros(7, dtype=np.float32)
    point_cam_shm = shared_memory.SharedMemory(create=True, size=point_cam.nbytes)
    point_cam_array = np.ndarray(buffer=point_cam_shm.buf, dtype=np.float32, shape=point_cam.shape)
    cub_in_position = shared_memory.SharedMemory(create=True, size=1)
    cube_in_position_array = np.ndarray(buffer=cub_in_position.buf, dtype=np.uint8, shape=(1,))
    action_array[:] = action  # Copy the initial action to shared memory

    # main loop
    fps = 30
    pipeline, align = prepare_realsense(fps)
    n_processes, max_trials, trial_process = 0, 1, None
    while True:
        t0 = time.time()
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            time.sleep(0.001)
            continue

        img = np.asanyarray(color_frame.get_data())

        
        h, w = img.shape[:2] # this prserves principle point
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        img = img[y0:y0+side, x0:x0+side]
        original_img = img.copy()
        
        cube_in_position_array[0] = 1 if cube_in_bottom_half(original_img.copy()) else 0 # image is BGR!
        # has_contact_array[0] = 1 if is_cube_close_to_tape(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy())['is_close'] else 0 # expects RGB image
        cv2.imshow("Captured Image", img)
        cv2.waitKey(1)  # Use 1 instead of 0 to avoid blocking
        # continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB 
        cropped_img = img
        img = cv2.resize(img, (64, 64)) 
        image_array[:] = img  # Copy the image to shared memory

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if record_video:
            ext_frame = ext_cam.capture_img()
            ext_frame = cv2.cvtColor(ext_frame, cv2.COLOR_BGR2RGB)
            # put img in the bottom left corner of ext_frame
            ext_frame[:64, :64, :] = img
            video.append(np.concatenate([ext_frame, cropped_img], axis=1))
            video_ts.append(time.time())


        latest_point_cam = eval_automator.get_red_cube_position(depth_frame, color_frame)
        if latest_point_cam is not None:
            point_cam = latest_point_cam
            point_cam_array[:] = point_cam

        if n_processes == 0 and (trial_process is None or not trial_process.is_alive()):
            n_processes += 1
            trial_process = Process(target=run_trials, args=(max_trials, action_shm.name, action.shape, action.dtype,
                                                            point_cam_shm.name, point_cam.shape, point_cam.dtype,
                                                            image_shm.name, dummy_img.shape, dummy_img.dtype, 
                                                            cub_in_position.name, cube_in_position_array.shape, cube_in_position_array.dtype))
            trial_process.start()
        if n_processes >= 1 and not trial_process.is_alive():
            break

        time.sleep(max(1./fps - (time.time() - t0), 0))

    # store the video
    print('----------------------------------------------------------------- storing video...')
    if record_video:
        video_fp = f'real_franka_eval_logs/videos/push/real_franka_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
        video = np.stack(video)
        video_fps = len(video) / (video_ts[-1] - video_ts[0])
        media.write_video(video_fp, np.array(video), fps=video_fps)
        ext_cam.cap.release()

    if trial_process is not None:
        trial_process.join()


    action_shm.close()
    action_shm.unlink()  # Unlink the shared memory
    image_shm.close()
    image_shm.unlink()  # Unlink the shared memory

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()