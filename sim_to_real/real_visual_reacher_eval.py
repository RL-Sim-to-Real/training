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
from get_policy_network import make_inference_fn #TODO: Figure this out
import pickle

import json
from multiprocessing import shared_memory, Process, set_start_method
import threading
import pyrealsense2 as rs
from pathlib import Path

from franka_real.FrankaPickCubeCartesian import FrankaPickCubeCartesian
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

        env_name = "PandaPickCubeCartesianModified"

        # Rasterizer is less feature-complete than ray-tracing backend but stable
        layer_size = 256

        network_factory = functools.partial(
            ppo_networks_vision.make_ppo_networks_vision,
            policy_hidden_layer_sizes=[layer_size, layer_size],
            value_hidden_layer_sizes= [layer_size, layer_size],
            # activation=linen.relu, # only works with default activation right now
            normalise_channels=True,
        )

        ppo_params = manipulation_params.brax_vision_ppo_config(env_name)

        del ppo_params.network_factory
        ppo_params.network_factory = network_factory

        policy_fn = {
            'cartesian_position': "test_policies/params_general_cartesian_increment-position.pkl",
            'joint_position': "test_policies/params_general_joint_increment-position.pkl",
            'joint_velocity': "test_policies/params_general_joint-velocity.pkl",
            'joint_torque': "test_policies/params_general_joint-torque.pkl",
        }[control_mode]
        if use_prop:
            policy_fn = policy_fn.replace('test_policies/', 'test_policies/qvel/')
            policy_fn = policy_fn.replace('.pkl', '_prop.pkl')
        with open(policy_fn, "rb") as f:
            params = pickle.load(f)

        self.action_shape = (4,) if 'cartesian' in policy_fn else (8,)
        inference_fn = make_inference_fn(network_factory=network_factory, action_size=self.action_shape[0], include_prop=use_prop)

        self.jit_inference_fn = jax.jit(inference_fn(params, deterministic=True))
        # self.action_shm = shared_memory.SharedMemory(name=action_name)
        # self.action_array = np.ndarray(self.action_shape, dtype=action_dtype, buffer=self.action_shm.buf)
        self.image_shm = shared_memory.SharedMemory(name=image_name)
        self.img_array = np.ndarray(image_shape, dtype=image_dtype, buffer=self.image_shm.buf)
        self.key = jax.random.PRNGKey(0)

    def get_action(self, proprioception=None):
        self.key, _ = jax.random.split(self.key)
        obs = {'pixels/view_0': self.img_array.copy()}
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

def agent_process(action_name, action_shape, action_dtype, 
                  image_name, image_shape, image_dtype):
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    env_name = "PandaPickCubeCartesianModified"

    # Rasterizer is less feature-complete than ray-tracing backend but stable
    layer_size = 256


    network_factory = functools.partial(
        ppo_networks_vision.make_ppo_networks_vision,
        policy_hidden_layer_sizes=[layer_size, layer_size],
        value_hidden_layer_sizes= [layer_size, layer_size],
        # activation=linen.relu, # only works with default activation right now
        normalise_channels=True,
    )

    ppo_params = manipulation_params.brax_vision_ppo_config(env_name)

    del ppo_params.network_factory
    ppo_params.network_factory = network_factory


    # Load the params object from the pickle file
    # policy_fn = "policies/policy_params_general_3d_256_image_aug_black_white_strip.pkl"
    # policy_fn = "test_policies/params_general_cartesian_increment-position.pkl"
    policy_fn = "test_policies/params_general_cartesian_increment-position_prop.pkl"
    with open(policy_fn, "rb") as f:
        params = pickle.load(f)

    inference_fn = make_inference_fn(network_factory=network_factory, include_prop='prop' in policy_fn)

    jit_inference_fn = jax.jit(inference_fn(params, deterministic=True))
    action_shm = shared_memory.SharedMemory(name=action_name)
    action_array = np.ndarray(action_shape, dtype=action_dtype, buffer=action_shm.buf)
    image_shm = shared_memory.SharedMemory(name=image_name)
    img_array = np.ndarray(image_shape, dtype=image_dtype, buffer=image_shm.buf)
    key = jax.random.PRNGKey(0)
    try:
        while True:
            key, _ = jax.random.split(key)
            obs = {'pixels/view_0': img_array.copy()}
            if 'prop' in policy_fn:
                obs['_prop'] = np.array([0.0]*19, dtype=np.float32)
            t0 = time.time()
            action, _ = jit_inference_fn(obs, key) # imperical inference time is 0.016
            print(f"Inference time: {(time.time() - t0) * 1000.:.3f} ms")
            # print(f"Action: {action}")
            time.sleep(0.25) # set the cycle time to 40 ms
            action_array[:] = action
    except KeyboardInterrupt:
        print("Agent process interrupted by user.")

    action_shm.close()
    image_shm.close()

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

def make_cube_upright(env, pose_ee, angle, angle_180):
    grasp_height = 0.04
    grasp_offset = 0 # degrees
    ee_angle = np.array(env.euler_from_quaternion(env.reset_ee_quaternion))
    cube_ee_angle = ee_angle.copy()
    # cube_ee_angle[2] += -(angle if angle < 45 else angle - 90) * np.pi / 180.0
    cube_ee_angle[2] += -(angle_180 - 90) * np.pi / 180.0
    cube_ee_angle[1] -= grasp_offset * np.pi / 180.
    env.move_to_pose_ee(pose_ee, ref_ee_angle=cube_ee_angle)
    pose_ee[2] = grasp_height
    # pose_ee[0] += 0.015 # offset to be above the cube center
    env.move_to_pose_ee(pose_ee, ref_ee_angle=cube_ee_angle)
    env.grasp_object()
    pose_ee[2] = 0.3
    env.move_to_pose_ee(pose_ee)
    # randomize the cube position
    cube_pos = np.array([0.48, 0.0, grasp_height + 0.04])
    rotated_ee_angle = ee_angle.copy()
    rotated_ee_angle[1] += (90 - 10 - grasp_offset) * np.pi / 180.
    env.move_to_pose_ee(cube_pos, ref_ee_angle=rotated_ee_angle)
    cube_pos[2] = grasp_height + 0.01
    env.move_to_pose_ee(cube_pos, ref_ee_angle=rotated_ee_angle)
    env.open_gripper()
    pose_ee[:2] = cube_pos[:2]
    pose_ee[2] = 0.2
    env.move_to_pose_ee(pose_ee)

def put_cube_on_white_strip(env, angle, pose_ee):
    ee_angle = np.array(env.euler_from_quaternion(env.reset_ee_quaternion))
    ee_angle[2] += -(angle if angle < 45 else angle - 90) * np.pi / 180.0
    grasp_height = 0.065
    pose_ee[2] = grasp_height
    pose_ee[0] += 0.015 # offset to be above the cube center
    env.move_to_pose_ee(pose_ee, ref_ee_angle=ee_angle)
    env.grasp_object()
    # time.sleep(0.25)
    pose_ee[2] = 0.3
    env.move_to_pose_ee(pose_ee)
    # time.sleep(1)
    # randomize the cube position
    cube_pos = np.array([np.random.uniform(0.565, 0.595), np.random.uniform(-0.095, 0.095), grasp_height + 0.01])
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
    while is_square < 0.5:
        print("Cube is knocked over")
        make_cube_upright(env, pose_ee, angle, angle_180)
        env.move_to_joint_positions(target_joints)
        env.apply_joint_vel(np.zeros((7,)))
        angle, angle_180, is_square, pose_ee = get_birds_eye_view(point_cam_array, env)

    put_cube_on_white_strip(env, angle, pose_ee)

def run_trials(max_trials, action_name, action_shape, action_dtype, point_cam_name, point_cam_shape, point_cam_dtype, image_name, image_shape, image_dtype):
    control_mode = [
        'cartesian_position',
        'joint_position',
        'joint_velocity',
        'joint_torque',
    ][2]
    use_prop = True
    # action_shm = shared_memory.SharedMemory(name=action_name)
    # action_array = np.ndarray(action_shape, dtype=action_dtype, buffer=action_shm.buf)
    env = FrankaPickCubeCartesian(camera_index=0, control_mode=control_mode)
    agent = Agent(control_mode, use_prop, action_name, action_shape, action_dtype, image_name, image_shape, image_dtype)

    # reset the cube position
    point_cam_shm = shared_memory.SharedMemory(name=point_cam_name)
    point_cam_array = np.ndarray(point_cam_shape, dtype=point_cam_dtype, buffer=point_cam_shm.buf)

    # reset the robot joints to initial position
    target_joints = np.array([-0.01266706, 0.23113158, 0.01397337, -2.11847885, -0.00837887, 2.33090511, 0.80890272])
    env.move_to_joint_positions(target_joints)

    trial_length = 60
    for i in range(max_trials):
        env.open_gripper()
        env.move_to_joint_positions(target_joints)
        env.apply_joint_vel(np.zeros((7,)))
        reset_cube_position(point_cam_array, env, target_joints)

        # reset the robot joints to initial position again
        env.move_to_joint_positions(target_joints)
        env.apply_joint_vel(np.zeros((7,)))

        success = False
        env.grasped = False
        ee_pos,_ = env.reset()
        t_start = time.time()
        print("Resetting cube position...")
        env.open_gripper()
        action = np.zeros(agent.action_shape, dtype=np.float32)
        env.logger.clear()
        while True:
            # action = action_array.copy()  # Copy the action from shared memory
            proprioception = None
            if agent.use_prop:
                obs = env.get_state()
                # proprioception = np.concatenate([obs['joints'], obs['joint_vels'], action, np.array([float(env.grasped)])], axis=0).astype(np.float32)
                proprioception = np.concatenate([obs['joint_vels'], action, np.array([float(env.grasped)])], axis=0).astype(np.float32)
                # proprioception = np.zeros((2,), dtype=np.float32)
                print(f"Proprioception: {proprioception.shape}, {proprioception}")
            action = agent.get_action(proprioception=proprioception)
            # action_y_z = 0.05 * action[:2] # this is the increment
            print(f"Action: {action}")
            if (action[-1] < -0.7 and not env.grasped): # grasp it only once
                print("attempting grasp")
                env.grasped = env.grasp_object()
                time.sleep(0.25)  # Wait for the gripper to close
            if env.grasped and action[-1] >= -0.0:
                env.open_gripper()
                env.grasped = False
                time.sleep(0.25)
            
            # reopen gripper if grasp was unsuccessful
            fingertip_width = env.get_fingertip_width()
            if env.grasped and fingertip_width < 0.035:
                print('unsuccessful grasp, opening gripper')
                env.gripper.stop_action()
                env.gripper.open()
                time.sleep(0.25)  # Wait for the gripper to open
                env.grasped = False

            start = time.time()
            ee_pos = env.step(action)
            end = time.time()
            print(f"Time taken for one step: {end - start:.3f} seconds")

            fingertip_width = env.get_fingertip_width()
            if env.grasped and fingertip_width > 0.035 and ee_pos[2] > 0.1:
                print(f"---- Trial {i}: Complete")
                success = True
                break

            if time.time() - t_start > trial_length:
                print(f"---- Trial {i}: Timeout")
                break

        env.logger.metrics[-1]['success'] = success

        # put the cube back down
        if env.grasped:
            target_x_y_z = jp.array([ee_pos[0], ee_pos[1], 0.08])  # Keep x, y the same and set z to 0.02
            env.move_to_pose_ee(target_x_y_z)
        env.open_gripper()

        # save logs
        fp = Path(f'real_franka_eval_logs/{control_mode}_use_prop_{use_prop}_trial_{i}.pkl')
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

def main():
    record_video = False
    if record_video:
        video, video_ts = [], []
        ext_cam = Camera(cam_index=6)
        
    # env = FrankaPickCubeCartesian(camera_index=0)
    eval_automator = FrankaEvalAutomator()

    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8) * 255  # Dummy image for initialization

    action = np.zeros((8,), dtype=np.float32)
    action_shm = shared_memory.SharedMemory(create=True, size=action.nbytes)
    action_array = np.ndarray(buffer=action_shm.buf, dtype=np.float32, shape=action.shape)
    image_shm = shared_memory.SharedMemory(create=True, size=dummy_img.nbytes)
    image_array = np.ndarray(buffer=image_shm.buf, dtype=np.uint8, shape=dummy_img.shape)
    point_cam = np.zeros(7, dtype=np.float32)
    point_cam_shm = shared_memory.SharedMemory(create=True, size=point_cam.nbytes)
    point_cam_array = np.ndarray(buffer=point_cam_shm.buf, dtype=np.float32, shape=point_cam.shape)
    action_array[:] = action  # Copy the initial action to shared memory
    # p = Process(target=agent_process, args=(action_shm.name, action.shape, action.dtype,
    #                                            image_shm.name, dummy_img.shape, dummy_img.dtype))
    # c = Process(target=camera_process, args=(image_shm.name, dummy_img.shape, dummy_img.dtype))
    # c = Process(target=rs_camera_process, args=(image_shm.name, dummy_img.shape, dummy_img.dtype, pipeline, align))
    # c = threading.Thread(target=rs_camera_thread, args=(image_array,), daemon=True)
    # c.start()  # Start the camera process
    # p.start()
    # input("Press Enter to start the control loop...")

    # main loop
    fps = 60
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
        original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.asanyarray(depth_frame.get_data())
        # img = cv2.convertScaleAbs(img, alpha=0.10)
        # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        cv2.imshow("Captured Image", img)
        cv2.waitKey(1)  # Use 1 instead of 0 to avoid blocking
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
                                                            image_shm.name, dummy_img.shape, dummy_img.dtype))
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