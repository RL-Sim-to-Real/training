


from franka_real.FrankaPickCubeCartesian import FrankaPickCubeCartesian
import pandas as pd
import time

if __name__ == "__main__":

    positions = []
    env = FrankaPickCubeCartesian(camera_index=0, control_mode="joint_position") 
    env.reset()
    print("Environment reset complete.")

    obs = env.get_state()
    q_pos = obs["joints"]
    # ensure first joint is at zero
    q_pos[0] = 0.0
    print(q_pos)
    for _ in range(100):
        env.robot.set_joint_positions(dict(zip(env.joint_names, q_pos)))
        env.rate.sleep()
    q_pos[0] += 0.5
    for _ in range(100):
        env.robot.set_joint_positions(dict(zip(env.joint_names, q_pos)))
        env.rate.sleep()
        obs = env.get_state()
        positions.append(obs["joints"][0])
    df = pd.DataFrame(positions, columns=["joint1_pos"])
    df.to_csv("joint1_pos_data.csv", index=False)
    env.close()