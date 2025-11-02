import os
import json
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import pyrealsense2 as rs

from franka_real.FrankaPickCubeCartesian import FrankaPickCubeCartesian
from franka_real.FrankaEvalAutomator import FrankaEvalAutomator

# ---- Shared-memory layout ----
# shm buffer float64[8]:
#   [0:7] -> latest_point_cam (x, y, z, ?, angle, angle_180, is_square)
#   [7]   -> valid flag (0.0 = no data, 1.0 = data ready)


def producer_realsense(shm_name: str, stop_event: mp.Event, fps: int = 30):
    """Continuously writes latest cube detection to shared memory."""
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray((8,), dtype=np.float64, buffer=shm.buf)

    pipeline = None
    try:
        # RealSense setup
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        pipeline.start(config)
        align = rs.align(rs.stream.color)

        evaluator = FrankaEvalAutomator()

        # Clear flag
        buf[7] = 0.0

        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                time.sleep(0.005)
                continue

            point = evaluator.get_red_cube_position(depth_frame, color_frame)
            if point is None:
                time.sleep(0.005)
                continue

            arr = np.asarray(point, dtype=np.float64).ravel()
            if arr.size < 7:
                tmp = np.zeros((7,), dtype=np.float64)
                tmp[:arr.size] = arr
                arr = tmp

            # Write data first, then mark valid
            buf[:7] = arr[:7]
            buf[7] = 1.0  # valid

            time.sleep(0.01)

    finally:
        try:
            if pipeline is not None:
                pipeline.stop()
        except Exception:
            pass
        shm.close()


# --- Geometry helpers ---
def make_homog(Rmat, tvec):
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = np.asarray(tvec).flatten()
    return T


def get_point_base(point_cam, env):
    ee_pose = env.robot.endpoint_pose()
    ee_quaternion = [
        ee_pose['orientation'].w,
        ee_pose['orientation'].x,
        ee_pose['orientation'].y,
        ee_pose['orientation'].z,
    ]
    R_ee = env.matrix_from_quaternion(ee_quaternion)
    T_base_ee = make_homog(R_ee, ee_pose['position'])

    T_ee_camera_file = os.path.expanduser(
        "~/Desktop/ICRA2026/Franka-Real/franka_real/T_ee_camera.json"
    )
    with open(T_ee_camera_file, "r") as f:
        T_ee_camera = np.array(json.load(f))

    T_base_camera = T_base_ee @ T_ee_camera
    point_cam_h = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
    point_base = T_base_camera @ point_cam_h
    return point_base


# --- SHM read helpers and motion primitives ---
def read_point_from_shm(buf, wait=True, timeout=2.0):
    t0 = time.time()
    while wait and buf[7] == 0.0 and (time.time() - t0) < timeout:
        time.sleep(0.005)
    if buf[7] == 0.0:
        return None
    return np.array(buf[:7], dtype=np.float64)


def get_birds_eye_view_from_shm(buf, env, iters=5):
    """Move above the detected cube using live SHM updates."""
    pose_ee = None
    angle = angle_180 = is_square = 0.0
    for _ in range(iters):
        arr = read_point_from_shm(buf, wait=True)
        if arr is None:
            continue
        angle, angle_180, is_square = float(arr[4]), float(arr[5]), float(arr[6])
        point_base = get_point_base(arr, env)
        print(f"Cube (base frame): {point_base[:3]}")
        pose_ee = point_base[:3].copy()
        pose_ee[2] = 0.30
        env.move_to_pose_ee(pose_ee)
    return angle, angle_180, is_square, pose_ee


def recenter_on_cube(buf, env, max_iters=10, tol=0.005):
    """Iteratively refine EE XY to center above cube using live SHM."""
    for _ in range(max_iters):
        arr = read_point_from_shm(buf, wait=True)
        if arr is None:
            continue
        point_base = get_point_base(arr, env)
        target_xy = point_base[:2]
        ee_pose = env.robot.endpoint_pose()
        ee_xy = np.array(ee_pose['position'][:2])
        err = target_xy - ee_xy
        if np.linalg.norm(err) < tol:
            break
        pose_ee = np.array(ee_pose['position'])
        pose_ee[:2] = target_xy
        env.move_to_pose_ee(pose_ee)


def make_cube_upright(env, pose_ee, angle, angle_180):
    grasp_height = 0.04
    grasp_offset = 0.0
    ee_angle = np.array(env.euler_from_quaternion(env.reset_ee_quaternion))
    cube_ee_angle = ee_angle.copy()
    cube_ee_angle[2] += -(angle_180 - 90) * np.pi / 180.0
    cube_ee_angle[1] -= grasp_offset * np.pi / 180.0

    env.move_to_pose_ee(pose_ee, ref_ee_angle=cube_ee_angle)
    pose_ee = pose_ee.copy()
    pose_ee[2] = grasp_height
    pose_ee[0] -= 0.025
    env.move_to_pose_ee(pose_ee, ref_ee_angle=cube_ee_angle)
    env.grasp_object()
    pose_ee[2] = 0.3
    env.move_to_pose_ee(pose_ee)

    cube_pos = np.array([0.48, 0.0, grasp_height + 0.04])
    rotated_ee_angle = ee_angle.copy()
    rotated_ee_angle[1] += (90 - 10 - grasp_offset) * np.pi / 180.0
    env.move_to_pose_ee(cube_pos, ref_ee_angle=rotated_ee_angle)
    cube_pos[2] = grasp_height + 0.028
    env.move_to_pose_ee(cube_pos, ref_ee_angle=rotated_ee_angle)
    env.open_gripper()
    pose_ee[:2] = cube_pos[:2]
    pose_ee[2] = 0.2
    env.move_to_pose_ee(pose_ee)


def put_cube_on_white_strip(env, angle, pose_ee):
    ee_angle = np.array(env.euler_from_quaternion(env.reset_ee_quaternion))
    ee_angle[2] += -(angle if angle < 45 else angle - 90) * np.pi / 180.0
    grasp_height = 0.065

    pose_ee = pose_ee.copy()
    pose_ee[2] = grasp_height
    pose_ee[0] -= 0.015
    env.move_to_pose_ee(pose_ee, ref_ee_angle=ee_angle)
    env.grasp_object()

    pose_ee[2] = 0.3
    env.move_to_pose_ee(pose_ee)

    cube_pos = np.array([
        np.random.uniform(0.52, 0.62),
        np.random.uniform(-0.095, 0.095),
        grasp_height + 0.01,
    ])
    print(f"Placing cube at: {cube_pos[:2]}")
    env.move_to_pose_ee(cube_pos)
    cube_pos[2] = grasp_height + 0.0005
    env.move_to_pose_ee(cube_pos)
    env.open_gripper()

    pose_ee[:2] = cube_pos[:2]
    pose_ee[2] = 0.2
    env.move_to_pose_ee(pose_ee)


def reset_cube_position(buf, env, target_joints):
    angle, angle_180, is_square, pose_ee = get_birds_eye_view_from_shm(buf, env)
    recenter_on_cube(buf, env)
    while is_square < 0.5:
        print("Cube detected as knocked over. Uprighting...")
        make_cube_upright(env, pose_ee, angle, angle_180)
        env.move_to_joint_positions(target_joints)
        env.apply_joint_vel(np.zeros((7,)))
        angle, angle_180, is_square, pose_ee = get_birds_eye_view_from_shm(buf, env)
        recenter_on_cube(buf, env)
    put_cube_on_white_strip(env, angle, pose_ee)


def consumer_reset(shm_name: str):
    """Reads cube detections from shared memory and performs the reset."""
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray((8,), dtype=np.float64, buffer=shm.buf)

    env = None
    try:
        env = FrankaPickCubeCartesian(camera_index=0, control_mode="joint_position")
        target_joints = np.array([
            -0.01266706, 0.23113158, 0.01397337, -2.11847885, -0.00837887, 2.33090511, 0.80890272
        ])
        env.open_gripper()
        env.move_to_joint_positions(target_joints)
        env.apply_joint_vel(np.zeros((7,)))

        # Wait for producer to populate once
        print("Waiting for latest_point_cam...")
        while buf[7] == 0.0:
            time.sleep(0.01)

        print("Initial latest_point_cam:", np.array(buf[:7], dtype=np.float64))
        reset_cube_position(buf, env, target_joints)
        print("Reset complete.")

    finally:
        if env is not None:
            env.apply_joint_vel(np.zeros((7,)))
            env.open_gripper()
            env.close()
        shm.close()


if __name__ == "__main__":
    # Create shared memory (8 float64 values)
    shm = shared_memory.SharedMemory(create=True, size=8 * 8)
    shm_name = shm.name
    np.ndarray((8,), dtype=np.float64, buffer=shm.buf)[:] = 0.0

    stop_event = mp.Event()

    try:
        p_prod = mp.Process(target=producer_realsense, args=(shm_name, stop_event), daemon=True)
        p_cons = mp.Process(target=consumer_reset, args=(shm_name,), daemon=False)

        p_prod.start()
        p_cons.start()

        p_cons.join()  # Wait for one reset run

    finally:
        stop_event.set()
        if p_prod.is_alive():
            p_prod.join(timeout=2.0)
        shm.close()
        shm.unlink()