import os
import torch
import csv
from env import SumoTrafficEnv
from dqn import DQNAgent
from constants import TOTAL_UNITS_OF_SIMULATION, HIDDEN_LAYER_SIZES

models_dir = os.path.join("..", "models")
logs_dir = os.path.join("..", "logs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

SUMO_CONFIG = os.path.join("..", "sumo_simulation", "Test3.sumocfg")

episodes = 200

env = SumoTrafficEnv(SUMO_CONFIG)

for hidden_dim in HIDDEN_LAYER_SIZES:
    log_file_path = os.path.join(logs_dir, f"training_rewards_{hidden_dim}.csv")

    with open(log_file_path, mode='w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Hidden_Layer_Size", "Episode", "Total_Reward"])

        agent = DQNAgent(state_dim=10, action_dim=5, hidden_dim=hidden_dim)
        best_reward = float('-inf')

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0

            for t in range(TOTAL_UNITS_OF_SIMULATION):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            agent.train()
            print(f"Hidden Dim {hidden_dim} - Episode {episode + 1}: Total Reward = {total_reward}")
            log_writer.writerow([hidden_dim, episode + 1, total_reward])

            # Save the best model for each hidden_dim
            if total_reward > best_reward:
                best_reward = total_reward
                best_model_path = os.path.join(models_dir, f"best_model_{hidden_dim}_{abs(best_reward)}.pth")
                torch.save(agent.model.state_dict(), best_model_path)
                print(f"New best model for hidden_dim {hidden_dim} saved with reward {best_reward}")

env.close()
print("Training complete. Best models saved in:", models_dir)