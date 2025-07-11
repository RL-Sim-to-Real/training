"""
Visualize TensorBoard logs for episodic returns.
@author: Alireza Azimi
"""

import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Path to extracted logs directory
logdir = "../logs"  # or provide full path

event_data = []

# Walk through all subdirectories to find event files
for root, _, files in os.walk(logdir):
    for file in files:
        if file.startswith("events.out.tfevents"):
            event_path = os.path.join(root, file)
            try:
                ea = EventAccumulator(event_path)
                ea.Reload()
                if "episode/sum_reward" in ea.Tags()["scalars"]:
                    events = ea.Scalars("episode/sum_reward")
                    steps = [e.step for e in events]
                    rewards = [e.value for e in events]
                    event_data.append((steps, rewards))
            except Exception as e:
                print(f"Failed to process {event_path}: {e}")

# Plotting
plt.figure(figsize=(10, 6))
for i, (steps, rewards) in enumerate(event_data):
    plt.plot(steps, rewards, label=f'Run {i+1}')
plt.xlabel("Timesteps")
plt.ylabel("Episodic Return")
plt.title("PandaPickCube")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pandapickcube_episodic_returns.png")