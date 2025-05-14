import os
import torch
import csv
import random
import time

import traci

from sumo_wrapper import SumoWrapper
from constants import TOTAL_UNITS_OF_SIMULATION, EPISODES

# Set up directories for logs and intersections
logs_dir = os.path.join("..", "logs")
intersections_dir = os.path.join("..", "intersections_128_throughput")  # Folder to store best RL models
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(intersections_dir, exist_ok=True)

random.seed(0)
torch.manual_seed(0)

SUMO_CONFIG = os.path.join("..", "sumo_simulation", "Square.sumocfg")
MODEL_DIR = os.path.join("..", "use_models")
sumo_wrapper = SumoWrapper(SUMO_CONFIG, MODEL_DIR)

log_file_path = os.path.join(logs_dir, "training_rewards_square_128_throughput.csv")
with open(log_file_path, mode='w', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Episode", "Total_Waiting_Time", "Total_Avg_Waiting_Time", "Total_Avg_Speed", "Total_Reward"])

    best_total_reward: float = float('-inf')  # Track best overall reward

    for episode in range(EPISODES):
        start_time: float = time.time()

        total_waiting_time = 0
        total_speed = 0
        total_rewards = 0
        steps = 0
        intersection_rewards = {}  # Track individual intersection rewards

        for t in range(TOTAL_UNITS_OF_SIMULATION):
            avg_waiting_time, avg_speed, reward, intersection_rewards_dict = sumo_wrapper.step()
            if avg_waiting_time is None:  # Simulation ended early
                break
            total_waiting_time += sum(traci.lane.getWaitingTime(lane) for lane in traci.lane.getIDList())
            total_speed += sum(traci.lane.getLastStepMeanSpeed(lane) for lane in traci.lane.getIDList()) / max(1, len(traci.lane.getIDList()))
            total_rewards += reward
            steps += 1

            # Store individual intersection rewards
            for intersection_id, int_reward in intersection_rewards_dict.items():
                if intersection_id not in intersection_rewards:
                    intersection_rewards[intersection_id] = 0
                intersection_rewards[intersection_id] += int_reward

        avg_waiting_time = total_waiting_time / max(1, steps)  # Avoid division by zero
        avg_speed = total_speed / max(1, steps)
        sumo_wrapper.train_agents()

        episode_time = time.time() - start_time
        print(f"Episode {episode + 1}:\n"
              f"Total Waiting Time = {total_waiting_time}\n"
              f"Avg Waiting Time = {avg_waiting_time:.2f}\n"
              f"Avg Speed = {avg_speed:.2f} m/s\n"
              f"Total Reward = {total_rewards}\n"
              f"Episode execution Time = {episode_time:.2f} seconds")
        log_writer.writerow([episode + 1, total_waiting_time, avg_waiting_time, avg_speed, total_rewards])

        # Save the best-performing models
        if total_rewards > best_total_reward:  # Compare absolute reward value
            best_total_reward = total_rewards
            episode_folder = os.path.join(intersections_dir, f"episode_{episode + 1}_{total_rewards}")
            os.makedirs(episode_folder, exist_ok=True)

            for intersection_id, intersection in sumo_wrapper.intersections.items():
                intersection_path = os.path.join(episode_folder, f"square_{intersection_id}_{intersection_rewards[intersection_id]}.pth")
                torch.save(intersection.agent.model.state_dict(), intersection_path)
                print(f"Saved best model for intersection {intersection_id} in {intersection_path}")

        sumo_wrapper.reset()

sumo_wrapper.close()
print("Training complete. Best models saved in:", intersections_dir)
