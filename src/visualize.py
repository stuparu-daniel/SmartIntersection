import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from reference import run_reference_simulation

logs_dir = os.path.join("..", "logs")
graphs_dir = os.path.join("..", "graphs")

os.makedirs(graphs_dir, exist_ok=True)

# Moving average window size
SMOOTHING_WINDOW = 10  # Adjust this to control the level of smoothing

def smooth_data(data, window_size=10):
    """Applies a moving average to smooth the data."""
    if len(data) < window_size:
        return data  # Return raw data if it's too short for smoothing
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

SUMO_CONFIG = os.path.join("..", "sumo_simulation", "Square.sumocfg")
ref_avg_wait_time, ref_avg_speed = run_reference_simulation(SUMO_CONFIG)

log_files = glob.glob(os.path.join(logs_dir, "training_rewards_square_128_throughput.csv"))

for log_file in log_files:
    # file_parts = log_file.split("_")
    # agent_name = file_parts[-2]  # Extract agent name (e.g., DQN, DQNPER, etc.)
    # hidden_dim = file_parts[-1].split(".")[0]  # Extract hidden layer size

    agent_name = "DQNPER_with_throughput"
    hidden_dim = "28x128x128x128x3"

    episodes = []
    total_rewards = []
    avg_speeds = []
    avg_waiting_times = []

    # Read data from CSV files
    with open(log_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            try:
                ep, _, avg_wait_time, avg_speed, reward = line.strip().split(",")
                episodes.append(int(ep))
                total_rewards.append(float(reward))
                avg_speeds.append(float(avg_speed))
                avg_waiting_times.append(float(avg_wait_time))
            except ValueError:
                continue

    # Apply smoothing
    smoothed_rewards = smooth_data(total_rewards, SMOOTHING_WINDOW)
    smoothed_speeds = smooth_data(avg_speeds, SMOOTHING_WINDOW)
    smoothed_waiting_times = smooth_data(avg_waiting_times, SMOOTHING_WINDOW)
    smoothed_episodes = episodes[:len(smoothed_rewards)]

    # Generate and save Total Rewards graph
    if episodes and total_rewards:
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, total_rewards, label="Raw Data", linewidth=1, alpha=0.5)
        plt.plot(smoothed_episodes, smoothed_rewards, label="Smoothed Data", linewidth=2, color='red')
        plt.xlabel("Episode")
        plt.ylabel("Total Rewards")
        plt.title(f"Training Progress - {agent_name} - Square Intersection - Total Rewards (Hidden {hidden_dim})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(graphs_dir, f"square_intersection_total_rewards_{agent_name}_{hidden_dim}.png"))
        plt.close()

    # Generate and save Average Speed graph
    if episodes and avg_speeds:
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, avg_speeds, label="Raw Data", linewidth=1, alpha=0.5, color='green')
        plt.plot(smoothed_episodes, smoothed_speeds, label="Smoothed Data", linewidth=2, color='blue')
        plt.axhline(y=ref_avg_speed, label="Reference Data", linewidth=3, color='red')
        plt.xlabel("Episode")
        plt.ylabel("Average Speed")
        plt.title(f"Training Progress - {agent_name} - Square Intersection - Average Speed (Hidden {hidden_dim})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(graphs_dir, f"square_intersection_average_speed_{agent_name}_{hidden_dim}.png"))
        plt.close()

    # Generate and save Average Waiting Time graph
    if episodes and avg_waiting_times:
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, avg_waiting_times, label="Raw Data", linewidth=1, alpha=0.5, color='purple')
        plt.plot(smoothed_episodes, smoothed_waiting_times, label="Smoothed Data", linewidth=2, color='orange')
        plt.axhline(y=ref_avg_wait_time, label="Reference Data", linewidth=3, color='brown')
        plt.xlabel("Episode")
        plt.ylabel("Average Waiting Time")
        plt.title(f"Training Progress - {agent_name} - Square Intersection - Avg Waiting Time (Hidden {hidden_dim})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(graphs_dir, f"square_intersection_avg_waiting_time_{agent_name}_{hidden_dim}.png"))
        plt.close()

print(f"Smoothed graphs saved in {graphs_dir}")
