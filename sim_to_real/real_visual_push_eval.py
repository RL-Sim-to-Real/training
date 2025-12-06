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
            'cartesian_position': "test_policies/params_general_cartesian_increment-position.pkl",
            'joint_position': "test_policies/params_general_joint_increment-position.pkl",
            'joint_velocity': "thesis_policies/push/params_general_joint-velocity.pkl",
            'joint_torque': "test_policies/params_general_joint-torque.pkl",
        }[control_mode]
        if use_prop:
            # policy_fn = policy_fn.replace('test_policies/', 'test_policies/qvel/')
            policy_fn = policy_fn.replace('.pkl', '_prop.pkl')
        with open(policy_fn, "rb") as f:
            params = pickle.load(f)

        self.action_shape = (3,) if 'cartesian' in policy_fn else (7,)
        inference_fn = make_inference_fn(network_factory=network_factory, action_size=self.action_shape[0], include_prop=use_prop)

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
        start_time = time.time()
        img = camera.capture_img()
        end_time = time.time()
        # h, w = img.shape[:2]
        # crop_x = 50 #(w - h) // 2
        # img = img[:, crop_x:(w - crop_x)]  # Crop to square
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
        # img = np.asanyarray(depth_frame.get_data())
        # img = cv2.convertScaleAbs(img, alpha=0.10)
        # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

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
    # image_shm = shared_memory.SharedMemory(name=image_name)
    # img_array = np.ndarray(image_shape, dtype=image_dtype, buffer=image_shm.buf)
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
        # img = np.asanyarray(depth_frame.get_data())
        # img = cv2.convertScaleAbs(img, alpha=0.10)
        # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

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

# def make_cube_upright(env, pose_ee, angle, angle_180):
#     grasp_height = 0.04
#     grasp_offset = 0 # degrees
#     ee_angle = np.array(env.euler_from_quaternion(env.reset_ee_quaternion))
#     cube_ee_angle = ee_angle.copy()
#     # cube_ee_angle[2] += -(angle if angle < 45 else angle - 90) * np.pi / 180.0
#     cube_ee_angle[2] += -(angle_180 - 90) * np.pi / 180.0
#     cube_ee_angle[1] -= grasp_offset * np.pi / 180.
#     env.move_to_pose_ee(pose_ee, ref_ee_angle=cube_ee_angle)
#     pose_ee[2] = grasp_height
#     pose_ee[0] -= 0.025 # offset to be above the cube center
#     env.move_to_pose_ee(pose_ee, ref_ee_angle=cube_ee_angle)
#     env.grasp_object()
#     pose_ee[2] = 0.3
#     ee_angle_flipped = ee_angle.copy()
#     ee_angle_flipped[2] -= np.pi
#     env.move_to_pose_ee(pose_ee)
#     # env.move_to_pose_ee(pose_ee, ref_ee_angle=ee_angle_flipped)
#     # randomize the cube position
#     cube_pos = np.array([0.48, 0.0, grasp_height + 0.04])
#     rotated_ee_angle = ee_angle.copy()

#     rotated_ee_angle[1] += (90 - 10 - grasp_offset) * np.pi / 180.

#     env.move_to_pose_ee(cube_pos, ref_ee_angle=rotated_ee_angle)
#     cube_pos[2] = grasp_height + 0.028
#     env.move_to_pose_ee(cube_pos, ref_ee_angle=rotated_ee_angle)
#     env.open_gripper()
#     pose_ee[:2] = cube_pos[:2]
#     pose_ee[2] = 0.2
#     env.move_to_pose_ee(pose_ee)

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
    env.move_to_pose_ee(pose_ee)
    # time.sleep(1)
    # randomize the cube position8
    cube_pos = np.array([np.random.uniform(0.60, 0.60), np.random.uniform(-0.03, 0.03), grasp_height + 0.005])
    # cube_pos = np.array([0.55, 0.0, 0.053]) # for debugging
    print(f"Moving cube to new position: {cube_pos[:2]}")
    env.move_to_pose_ee(cube_pos)
    cube_pos[2] = grasp_height + 0.0005
    env.move_to_pose_ee(cube_pos)
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
    # print('camera position:', T_base_camera[:3, 3])
    # T_base_camera = T_base_ee @ np.linalg.inv(T_ee_camera)
    # T_base_camera = np.linalg.inv(T_base_ee @ np.linalg.inv(T_ee_camera))
    # T_base_camera = T_ee_camera @ T_base_ee
    # T_base_camera = np.linalg.inv(T_ee_camera @ T_base_ee)
    point_cam_homog = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
    # point_cam_homog = np.array([point_cam[1], -point_cam[0], -point_cam[2], 1.0])
    # point_ee = np.linalg.inv(T_ee_camera) @ point_cam_homog
    point_ee = T_ee_camera @ point_cam_homog
    point_base = T_base_camera @ point_cam_homog
    return point_base, point_ee

def reset_cube_position(point_cam_array, env, target_joints):
    angle, angle_180, is_square, pose_ee = get_birds_eye_view(point_cam_array, env)
    # while is_square < 0.5: # don't need this
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


def is_cube_close_to_tape(
    img_rgb: np.ndarray,
    close_frac: float = 0.25,   # fraction of tape thickness used as threshold
    min_thr_px: int = 6,        # minimum pixel threshold
    annotate_path: str = None   # if set, save an annotated PNG here
) -> Dict:
    """
    Returns:
      {
        'cube_center': (x, y),
        'tape_center': (x, y),
        'vertical_distance_pixels': int,
        'euclidean_distance_pixels': int,
        'tape_height_pixels': int,
        'threshold_pixels': int,
        'is_close': bool,
        'annotated_image_path': str or None
      }
    """


    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # --- Detect red cube (handle hue wrap-around) ---
    # Tune S/V floors if lighting changes
    lower_red1 = np.array([0,   90,  90], np.uint8)
    upper_red1 = np.array([10, 255, 255], np.uint8)
    lower_red2 = np.array([170, 90,  90], np.uint8)
    upper_red2 = np.array([180, 255, 255], np.uint8)

    mask_r = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    cnts_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_r:
        return {'is_close': False}  # assume not close if no cube found
    cube_cnt = max(cnts_r, key=cv2.contourArea)
    M = cv2.moments(cube_cnt)
    if M['m00'] == 0:
        raise RuntimeError("Degenerate red contour (zero area).")
    cx_cube = int(M['m10'] / M['m00'])
    cy_cube = int(M['m01'] / M['m00'])

    # --- Detect white tape (low saturation, high value) ---
    lower_white = np.array([0,   0, 180], np.uint8)
    upper_white = np.array([179, 40, 255], np.uint8)
    mask_w = cv2.inRange(hsv, lower_white, upper_white)
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=2)

    cnts_w, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_w:
        return {'is_close': False}
    tape_cnt = max(cnts_w, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(tape_cnt)
    cx_tape = x + w // 2
    cy_tape = y + h // 2

    # --- Distances & decision ---
    dy = abs(cy_cube - cy_tape)
    euclid = int(np.hypot(cx_cube - cx_tape, cy_cube - cy_tape))
    thr = int(max(min_thr_px, close_frac * h))
    is_close = dy <= thr

    return {
        "cube_center": (cx_cube, cy_cube),
        "tape_center": (cx_tape, cy_tape),
        "vertical_distance_pixels": int(dy),
        "euclidean_distance_pixels": int(euclid),
        "tape_height_pixels": int(h),
        "threshold_pixels": int(thr),
        "is_close": bool(is_close),
    }

def run_trials(max_trials, 
               action_name, 
               action_shape, 
               action_dtype, 
               point_cam_name, 
               point_cam_shape, 
               point_cam_dtype, image_name, image_shape, image_dtype, has_contact_name, has_contact_shape, has_contact_dtype):
    save_logs = True
    control_mode = [
        'cartesian_position',
        'joint_position',
        'joint_velocity',
        'joint_torque',
    ][2]

    use_prop = True
    # action_shm = shared_memory.SharedMemory(name=action_name)
    # action_array = np.ndarray(action_shape, dtype=action_dtype, buffer=action_shm.buf)
    env = FrankaPushCube(camera_index=0, control_mode=control_mode)
    agent = Agent(control_mode, use_prop, action_name, action_shape, action_dtype, image_name, image_shape, image_dtype)

    # reset the cube position
    point_cam_shm = shared_memory.SharedMemory(name=point_cam_name)
    point_cam_array = np.ndarray(point_cam_shape, dtype=point_cam_dtype, buffer=point_cam_shm.buf)

    # image_shm = shared_memory.SharedMemory(name=image_name)
    # img_array = np.ndarray(image_shape, dtype=image_dtype, buffer=image_shm.buf)
    has_contact_shm = shared_memory.SharedMemory(name=has_contact_name)
    has_contact_array = np.ndarray(has_contact_shape, dtype=has_contact_dtype, buffer=has_contact_shm.buf)

    # reset the robot joints to initial position
    target_joints = np.array([-0.01266706, 0.23113158, 0.01397337, -2.11847885, -0.00837887, 2.33090511, 0.80890272])
    # target_joints = np.array([-0.00002, 0.47804, -0.00055, -1.81309, -0.00161, 2.34597, 0.78501])
    env.move_to_joint_positions(target_joints)

    trial_length = 60
    displacement = 0
    for i in range(max_trials):
        has_contact_array[0] = 0
        
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
        env.close_gripper()
        ee_pos,_ = env.reset()
        t_start = time.time()


        action = np.zeros(agent.action_shape, dtype=np.float32)
        env.logger.clear()
        init_pose_ee = ee_pos.copy()
        print(ee_pos)
        env.close_gripper()
        # env.gripper.home_joints()
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

                ee_height_n = obs['height']
                # h_min, h_max = 0.0, 0.2
                # ee_height_n = jp.clip(2.0 * (ee_height_n - h_min) / (h_max - h_min) - 1.0, -1.0, 1.0)
                
                proprioception = np.concatenate([ 
                    normalized_jp, 
                    normalized_jv,  # Include joint velocities in proprioception
                    action, ee_height_n], axis=-1).astype(np.float32)
                # proprioception = np.concatenate([obs['height'], action, np.array([float(env.grasped)])], axis=0).astype(np.float32)
                # proprioception = np.zeros((2,), dtype=np.float32)
                print(f"Proprioception: {proprioception.shape}, {proprioception}")
            action = agent.get_action(proprioception=proprioception)
            # action_y_z = 0.05 * action[:2] # this is the increment
            print(f"Action: {action}")

            # reopen gripper if grasp was unsuccessful

            start = time.time()
            ee_pos = env.step(action)
            end = time.time()
            print(f"Time taken for one step: {end - start:.3f} seconds")
            print(has_contact_array[0])
            if has_contact_array[0] > 0:
                success = True 
                print(f"---- Trial {i}: Success! Displacement: {displacement:.3f} m")
                time.sleep(1)
                break
            
            # fingertip_width = env.get_fingertip_width()
            # if env.grasped and fingertip_width > 0.035 and ee_pos[2] > 0.1:
            #     print(f"---- Trial {i}: Complete")
            #     success = True
            #     break

            if time.time() - t_start > trial_length:
                print(f"---- Trial {i}: Timeout")
                break
            if ee_pos[0] < 0.3:
                print(f"---- Trial {i}: Robot moved out of workspace")
                break
        # print(env.logger.metrics)
        env.logger.metrics[-1]['success'] = success
        env.logger.metrics[-1]['trial time'] = time.time() - t_start


        # save logs
        if save_logs:
            fp = Path(f'real_franka_eval_logs/{control_mode}_use_prop_{use_prop}_trial_{i}.pkl.csv')
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
    has_contact_shm = shared_memory.SharedMemory(create=True, size=1)
    has_contact_array = np.ndarray(buffer=has_contact_shm.buf, dtype=np.uint8, shape=(1,))
    action_array[:] = action  # Copy the initial action to shared memory
    # p = Process(target=agent_process, args=(action_shm.name, action.shape, action.dtype,
    #                                            image_shm.name, dummy_img.shape, dummy_img.dtype))
    # c = Process(target=camera_process, args=(image_shm.name, dummy_img.shape, dummy_img.dtype))
    # c = Process(target=rs_camera_process, args=(image_shm.name, dummy_img.shape, dummy_img.dtype, pipeline, align))
    # c = threading.Thread(target=rs_camera_thread, args=(image_array,), daemon=True)
    # c.start()  # Start the camera process
    # p.start()

    # main loop
    fps = 30
    pipeline, align = prepare_realsense(fps)
    n_processes, max_trials, trial_process = 0, 10, None
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
        # img = np.asanyarray(depth_frame.get_data())
        # img = cv2.convertScaleAbs(img, alpha=0.10)
        # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        

        
        img = img[:, 120:]
        original_img = img.copy()
        has_contact_array[0] = 1 if is_cube_close_to_tape(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy())['is_close'] else 0 # expects RGB image
        cv2.imshow("Captured Image", img)
        cv2.waitKey(1)  # Use 1 instead of 0 to avoid blocking
        # H, W = img.shape[:2]


        # print(f"Captured image size: {W}x{H}")
        img = cv2.resize(img, (64, 64)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB 
        image_array[:] = img  # Copy the image to shared memory

        # print(f"Time taken to capture and process image: {end_time - start_time:.3f} seconds")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if record_video:
            ext_frame = ext_cam.capture_img()
            ext_frame = cv2.cvtColor(ext_frame, cv2.COLOR_BGR2RGB)
            # put img in the bottom left corner of ext_frame
            ext_frame[:64, :64, :] = img
            video.append(np.concatenate([ext_frame, original_img], axis=1))
            video_ts.append(time.time())
            # cv2.imshow("External Camera", ext_frame)
            # cv2.waitKey(1)

        latest_point_cam = eval_automator.get_red_cube_position(depth_frame, color_frame)
        if latest_point_cam is not None:
            point_cam = latest_point_cam
            point_cam_array[:] = point_cam
        # print(f"Cube position (x, y, z) in camera frame: {point_cam}")
        # if n_processes < max_trials and (trial_process is None or not trial_process.is_alive()):
        if n_processes == 0 and (trial_process is None or not trial_process.is_alive()):
            n_processes += 1
            trial_process = Process(target=run_trials, args=(max_trials, action_shm.name, action.shape, action.dtype,
                                                            point_cam_shm.name, point_cam.shape, point_cam.dtype,
                                                            image_shm.name, dummy_img.shape, dummy_img.dtype, 
                                                            has_contact_shm.name, has_contact_array.shape, has_contact_array.dtype))
            trial_process.start()
        if n_processes >= 1 and not trial_process.is_alive():
            break

        time.sleep(max(1./fps - (time.time() - t0), 0))

    # store the video
    print('----------------------------------------------------------------- storing video...')
    if record_video:
        video_fp = f'real_franka_eval_logs/videos/real_franka_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
        video = np.stack(video)
        video_fps = len(video) / (video_ts[-1] - video_ts[0])
        media.write_video(video_fp, np.array(video), fps=video_fps)
        ext_cam.cap.release()

    if trial_process is not None:
        trial_process.join()

    # results = []
    # for _ in range(2):
    #     # eval_automator.reset_cube()
    #     result = run_trial(action_array, env)
    #     results.append(result)

    # env.reset()
    # env.close()
    # p.join()  # Wait for the agent process to finish
    # c.join()  
    action_shm.close()
    action_shm.unlink()  # Unlink the shared memory
    image_shm.close()
    image_shm.unlink()  # Unlink the shared memory

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()