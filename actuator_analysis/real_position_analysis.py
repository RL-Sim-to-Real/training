


from franka_real.FrankaPickCubeCartesian import FrankaPickCubeCartesian
import pandas as pd
import time
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="displacement value",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=200.0,
        help="proportional gain",
    )
    parser.add_argument(
        "--d",
        type=float,
        default=50.0,
        help="derivative gain",
    )
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    positions = []

    args = parse_args()
    k = [args.k, 200.0, 200.0, 200.0, 300.0, 200.0, 50.0] 
    d = [args.d, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0]
    env = FrankaPickCubeCartesian(camera_index=0, control_mode="joint_position", k=k, d=d) 
    env.robot_status.enable()

    print("Environment reset complete.")
    # k,d = env.controller_param_config_client.get_controller_gains()

    delta = args.delta


    obs = env.get_state()
    q_pos = obs["joints"].copy()
    joint = 2

    # ensure first joint is at zero
    q_pos[joint] = 0.0
    print(q_pos)
    for _ in range(100):
        env.robot.set_joint_positions(dict(zip(env.joint_names, q_pos)))
        env.rate.sleep()


    curr_contr = env.cmi.current_controller
    # print(curr_contr, self.cmi.is_running(curr_contr))
    print('------------------------------------------------------ restarting ', curr_contr)
    env.cmi.stop_controller(curr_contr)
    while env.cmi.is_running(curr_contr):
        print('waiting for controller to stop')
        time.sleep(1)
    env.cmi.start_controller(curr_contr)
    env.robot_status.enable()
    # print("joint position controller:", env.cmi.joint_position_controller)
    # k[0] = 0.0
    # print(k)
    # env.controller_param_config_client.set_controller_gains(k,d)
    print(q_pos)
    q_pos[joint] += delta
    q_d = np.zeros_like(q_pos)
    for _ in range(100):
        env.robot.set_joint_positions_velocities(q_pos,  q_d)
        # env.robot.set_joint_positions(dict(zip(env.joint_names, q_pos)))
        # env.apply_joint_vel(q_d)
        env.rate.sleep()
        obs = env.get_state()
        positions.append(obs["joints"][joint])
        # delta = -1 * delta
        # q_pos[0] += delta
    


    df = pd.DataFrame(positions, columns=[f"joint{joint+1}_pos"])
    df.to_csv(f"joint{joint+1}_positions_real.csv", index=False)
    env.close()