import os
import torch
import csv
import random
from env import SumoTrafficEnv
from dqn import DQNAgent
from constants import TOTAL_UNITS_OF_SIMULATION, HIDDEN_LAYER_SIZES
from flow_modifier import modify_vehicle_flow

models_dir = os.path.join("..", "models")
logs_dir = os.path.join("..", "logs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

random.seed(0)

SUMO_CONFIG = os.path.join("..", "sumo_simulation", "Test3.sumocfg")
ROUTE_FILE = os.path.join("..", "sumo_simulation", "Test3.rou.xml")

episodes = 200

env = SumoTrafficEnv(SUMO_CONFIG)

for hidden_dim in HIDDEN_LAYER_SIZES:
    log_file_path = os.path.join(logs_dir, f"training_rewards_{hidden_dim}.csv")

    with open(log_file_path, mode='w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Hidden_Layer_Size", "Episode", "Total_Rewards", "Average_Speed"])

        agent = DQNAgent(state_dim=len(env.lane_ids) * 2 + 2, action_dim=5, hidden_dim=hidden_dim)
        best_reward = float('-inf')

        for episode in range(episodes):
            modify_vehicle_flow(ROUTE_FILE, 50, 250)

            state = env.reset()
            total_rewards = 0
            total_speed = 0
            total_waiting_time = 0

            for t in range(TOTAL_UNITS_OF_SIMULATION):
                action = agent.act(state)
                next_state, reward, done, waiting_time = env.step(action)
                agent.remember(state, action, reward, next_state)
                state = next_state
                total_rewards += reward
                total_waiting_time += waiting_time
                total_speed += sum(state[len(env.lane_ids):len(env.lane_ids) * 2]) / len(
                    env.lane_ids)  # Extract avg speed

            avg_speed = total_speed / TOTAL_UNITS_OF_SIMULATION
            agent.train()
            print(
                f"Hidden Dim {hidden_dim} - Episode {episode + 1}: Total Rewards = {abs(total_rewards)}, Total "
                f"Waiting Time = {total_waiting_time}, Avg Speed = {avg_speed}")
            log_writer.writerow([hidden_dim, episode + 1, total_rewards, avg_speed])

            if total_rewards > best_reward:
                best_reward = total_rewards
                best_model_path = os.path.join(models_dir, f"best_model_{hidden_dim}_{abs(best_reward)}.pth")
                torch.save(agent.model.state_dict(), best_model_path)
                print(f"New best model for hidden_dim {hidden_dim} saved with total rewards {abs(best_reward)}")

env.close()
print("Training complete. Best models saved in:", models_dir)
