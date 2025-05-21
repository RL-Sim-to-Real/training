import cv2
import os

def visualize_policy(agent, env, output_dir):
    images = []
    state, _ = env.reset()
    for _ in range(200):
        action = agent.sample_actions(state)
        next_state, reward, done, info = env.step(action)
        if isinstance(next_state, tuple) and len(next_state) == 2:
            img, _ = next_state
        else:
            img = env.sim.cam.render(rgb=True)[0]
        images.append(img)
        state = next_state
        if done:
            break

    height, width, layers = images[0].shape
    video_name = os.path.join(output_dir + "/videos", f'checkpoint_{env.total_steps}.mp4')
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    video.release()