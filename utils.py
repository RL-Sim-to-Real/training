import cv2
import os

def visualize_policy(agent, env, output_dir):
    images = []
    obs, _ = env.reset()
    for _ in range(200):
        img = env.render()
        images.append(img)
        action = agent.sample_actions(obs)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        if done:
            break

    height, width, layers = images[0].shape
    video_name = os.path.join(output_dir + "/videos", f'checkpoint_{env.total_steps}.mp4')
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    video.release()