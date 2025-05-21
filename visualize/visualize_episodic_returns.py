import json
import re



log_file_paths = [
"/home/chemist/Desktop/CoRL-2025/training/results/reacher_non_visual_dense_torque/seed_0/train.log",
"/home/chemist/Desktop/CoRL-2025/training/results/reacher_non_visual_dense_velocity/seed_0/train.log",
"/home/chemist/Desktop/CoRL-2025/training/results/reacher_non_visual_dense_position/seed_0/train.log",
]

import matplotlib.pyplot as plt

# Initialize a figure
plt.figure(figsize=(10, 6))

# Iterate through each log file path
for log_file_path in log_file_paths:
    steps = []
    returns = []
    
    # Read and parse the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            log_entry = json.loads(line)
            steps.append(log_entry["step"])
            returns.append(log_entry["return"])
    
    # Plot the data
    match = re.search(r'_(torque|position|velocity)/', log_file_path)
    label = match.group(1) if match else "unknown"  # Extract torque, position, or velocity
    plt.plot(steps, returns, label=f"{label}-{label}")

# Add title, labels, and legend
plt.title("Action-Actuation Episodic Performance")
plt.xlabel("Time Steps")
plt.ylabel("Episodic Return")
plt.legend()
plt.grid(True)

# Save the plot to a PDF file
plt.savefig("non_visual_reach_tasks.pdf")
plt.close()