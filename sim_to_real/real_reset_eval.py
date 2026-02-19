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


def camera_process(image_name, image_shape, image_dtype):
    camera = Camera(cam_index=4)
    # FIXME: set the frame dimensions to be square to match training
    image_shm = shared_memory.SharedMemory(name=image_name)
    img_array = np.ndarray(image_shape, dtype=image_dtype, buffer=image_shm.buf)
    while True:
        start_time = time.time()
        img = camera.capture_img()
        end_time = time.time()
        cv2.imshow("Captured Image", img)
        cv2.waitKey(1)  # Use 1 instead of 0 to avoid blocking
        img = cv2.resize(img, (64, 64)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB 
        img_array[:] = img  # Copy the image to shared memory

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
    pose_ee[0] -= 0.025 # offset to be above the cube center
    env.move_to_pose_ee(pose_ee, ref_ee_angle=cube_ee_angle)
    env.grasp_object()
    pose_ee[2] = 0.3
    ee_angle_flipped = ee_angle.copy()
    ee_angle_flipped[2] -= np.pi
    env.move_to_pose_ee(pose_ee)
    # env.move_to_pose_ee(pose_ee, ref_ee_angle=ee_angle_flipped)
    # randomize the cube position
    cube_pos = np.array([0.48, 0.0, grasp_height + 0.04])
    rotated_ee_angle = ee_angle.copy()

    rotated_ee_angle[1] += (90 - 10 - grasp_offset) * np.pi / 180.

    env.move_to_pose_ee(cube_pos, ref_ee_angle=rotated_ee_angle)
    cube_pos[2] = grasp_height + 0.028
    env.move_to_pose_ee(cube_pos, ref_ee_angle=rotated_ee_angle)
    env.open_gripper()
    pose_ee[:2] = cube_pos[:2]
    pose_ee[2] = 0.2
    env.move_to_pose_ee(pose_ee)

def put_cube_on_white_strip(env, angle, pose_ee, begin_time):
    ee_angle = np.array(env.euler_from_quaternion(env.reset_ee_quaternion))
    ee_angle[2] += -(angle if angle < 45 else angle - 90) * np.pi / 180.0
    grasp_height = 0.065
    pose_ee[2] = grasp_height
    pose_ee[0] -= 0.0 # offset to be above the cube center
    env.move_to_pose_ee(pose_ee, ref_ee_angle=ee_angle)
    env.grasp_object()
    # time.sleep(0.25)
    pose_ee[2] = 0.2
    env.move_to_pose_ee(pose_ee)
    env.logger.metrics[-1]['success'] = True
    env.logger.metrics[-1]['Trial'] = time.time() - begin_time
    # Don't log after this point since we are placing the cube
    # randomize the cube position
    cube_pos = np.array([np.random.uniform(0.52, 0.62), np.random.uniform(-0.1, 0.1), grasp_height + 0.01])
    # cube_pos = np.array([0.55, 0.0, 0.053]) # for debugging
    print(f"Moving cube to new position: {cube_pos[:2]}")
    env.move_to_pose_ee(cube_pos, log=False)
    cube_pos[2] = grasp_height + 0.0005
    env.move_to_pose_ee(cube_pos, log=False)
    # time.sleep(1)
    env.open_gripper()
    # time.sleep(0.25)
    pose_ee[:2] = cube_pos[:2]
    pose_ee[2] = 0.2
    env.move_to_pose_ee(pose_ee, log=False)
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
    # point_cam_homog = np.array([point_cam[1], -point_cam[0], -point_cam[2], 1.0])
    # point_ee = np.linalg.inv(T_ee_camera) @ point_cam_homog
    point_ee = T_ee_camera @ point_cam_homog
    point_base = T_base_camera @ point_cam_homog
    return point_base, point_ee

def reset_cube_position(point_cam_array, env, target_joints, begin_time):
    angle, angle_180, is_square, pose_ee = get_birds_eye_view(point_cam_array, env)
    while is_square < 0.5:
        print("Cube is knocked over")
        make_cube_upright(env, pose_ee, angle, angle_180)
        env.move_to_joint_positions(target_joints)
        env.apply_joint_vel(np.zeros((7,)))
        angle, angle_180, is_square, pose_ee = get_birds_eye_view(point_cam_array, env)

    put_cube_on_white_strip(env, angle, pose_ee, begin_time)

def run_trials(max_trials, action_name, action_shape, action_dtype, point_cam_name, point_cam_shape, point_cam_dtype, image_name, image_shape, image_dtype):
    save_logs = True
    control_mode = "reset"


    use_prop = True
    # action_shm = shared_memory.SharedMemory(name=action_name)
    # action_array = np.ndarray(action_shape, dtype=action_dtype, buffer=action_shm.buf)
    env = FrankaPickCubeCartesian(camera_index=0, control_mode=control_mode)
    

    # reset the cube position
    point_cam_shm = shared_memory.SharedMemory(name=point_cam_name)
    point_cam_array = np.ndarray(point_cam_shape, dtype=point_cam_dtype, buffer=point_cam_shm.buf)

    # reset the robot joints to initial position
    target_joints = np.array([-0.01266706, 0.23113158, 0.01397337, -2.11847885, -0.00837887, 2.33090511, 0.80890272])
    # target_joints = np.array([-0.00002, 0.47804, -0.00055, -1.81309, -0.00161, 2.34597, 0.78501])
    env.move_to_joint_positions(target_joints)
    max_grasp_attempts = 10

    trial_length = 30
    skip_to_trial = 0
    for i in range(max_trials):
        if i < skip_to_trial:
            np.array([np.random.uniform(0.52, 0.62), np.random.uniform(-0.095, 0.095), 0 + 0.01])
            continue
    
        env.logger.clear()
        env.open_gripper()
        env.move_to_joint_positions(target_joints)
        env.apply_joint_vel(np.zeros((7,)))
        begin_time = time.time()
        reset_cube_position(point_cam_array, env, target_joints, begin_time) # make sure to uncomment

        # reset the robot joints to initial position again
        env.move_to_joint_positions(target_joints)
        env.apply_joint_vel(np.zeros((7,)))


        # save logs
        if save_logs:
            fp = Path(f'real_franka_eval_logs/pick/{control_mode}_use_prop_{use_prop}_trial_{i}.pkl.csv')
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

    # main loop
    fps = 30
    pipeline, align = prepare_realsense(fps)
    n_processes, max_trials, trial_process = 0, 12, None
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

        cv2.imshow("Captured Image", img)
        cv2.waitKey(1)  # Use 1 instead of 0 to avoid blocking


        # print(f"Captured image size: {W}x{H}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB 
        cropped_img = img.copy()
        img = cv2.resize(img, (64, 64)) 
        
        image_array[:] = img  # Copy the image to shared memory

        # print(f"Time taken to capture and process image: {end_time - start_time:.3f} seconds")
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
                                                            image_shm.name, dummy_img.shape, dummy_img.dtype))
            trial_process.start()
        if n_processes >= 1 and not trial_process.is_alive():
            break

        time.sleep(max(1./fps - (time.time() - t0), 0))

    # store the video
    print('----------------------------------------------------------------- storing video...')
    if record_video:
        video_fp = f'real_franka_eval_logs/videos/pick/real_franka_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
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