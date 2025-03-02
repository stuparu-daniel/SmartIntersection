import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from constants import TOTAL_UNITS_OF_SIMULATION

logs_dir = os.path.join("..", "logs")
log_files = glob.glob(os.path.join(logs_dir, "training_rewards_*.csv"))

plt.figure(figsize=(10, 5))

for log_file in log_files:
    hidden_dim = log_file.split("_")[-1].split(".")[0]  # Extract hidden layer size
    episodes = []
    total_waiting_times = []

    with open(log_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            try:
                _, ep, waiting_time = line.strip().split(",")
                episodes.append(int(ep))
                total_waiting_times.append(float(waiting_time) / TOTAL_UNITS_OF_SIMULATION)
            except ValueError:
                continue

    if episodes and total_waiting_times:
        plt.plot(episodes, total_waiting_times, label=f"Hidden {hidden_dim}", linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Total Waiting Time (seconds)")
plt.title("Training Progress - Waiting Time Optimization")
plt.legend()
plt.grid(True)
plt.show()
