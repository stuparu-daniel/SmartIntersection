import os
import torch
import csv
import random
from env import SumoTrafficEnv
from dqn import DQNAgent
from dqn_dropout import DQNDropoutAgent
from dqn_batchnorm import DQNBatchNormAgent
from dqn_dueling import DuelingDQNAgent
from dqn_double import DoubleDQNAgent
from dqn_per import DQNPERAgent
from constants import TOTAL_UNITS_OF_SIMULATION, HIDDEN_LAYER_SIZES, EPISODES
from flow_modifier import modify_vehicle_flow, generate_vehicle_flow_list, get_num_flows
import traci

# Set up directories for models and logs
models_dir = os.path.join("..", "models")
logs_dir = os.path.join("..", "logs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

random.seed(0)
torch.manual_seed(0)

SUMO_CONFIG = os.path.join("..", "sumo_simulation", "Test3.sumocfg")
ROUTE_FILE = os.path.join("..", "sumo_simulation", "Test3.rou.xml")

# Initialize SUMO Environment
env = SumoTrafficEnv(SUMO_CONFIG)

# Dictionary mapping agent names to their respective classes
agent_variants = {
    "DQNPER": DQNPERAgent,
    # "DuelingDQN": DuelingDQNAgent,
    # "DoubleDQN": DoubleDQNAgent,
    # "DQN": DQNAgent
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training {agent_variants}\non {device} for {EPISODES} episodes of length {TOTAL_UNITS_OF_SIMULATION}")

# vehicle_flow_values = generate_vehicle_flow_list(num_of_episodes=EPISODES, num_of_flows=get_num_flows(ROUTE_FILE))

# Train each agent variant with different hidden layer sizes
for agent_name, AgentClass in agent_variants.items():
    for hidden_dim in HIDDEN_LAYER_SIZES:
        log_file_path = os.path.join(logs_dir, f"training_rewards_{agent_name}_{hidden_dim}.csv")

        with open(log_file_path, mode='w', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(["Agent", "Hidden_Layer_Size", "Episode", "Total_Rewards", "Average_Speed", "Average_Waiting_Time"])

            agent = AgentClass(state_dim=len(env.lane_ids) * 2 + 2, action_dim=5, hidden_dim=hidden_dim)
            best_reward = float('-inf')
            best_model_path = os.path.join(models_dir, f"best_model_{agent_name}_{hidden_dim}.pth")

            for episode in range(EPISODES):
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
                    total_speed += sum(state[len(env.lane_ids):len(env.lane_ids) * 2]) / len(env.lane_ids)

                avg_speed = total_speed / TOTAL_UNITS_OF_SIMULATION
                avg_waiting_time = total_waiting_time / TOTAL_UNITS_OF_SIMULATION
                agent.train()

                print(
                    f"{agent_name} - Hidden Dim {hidden_dim} - Episode {episode + 1}: Total Rewards = {abs(total_rewards)}, "
                    f"Total Waiting Time = {total_waiting_time}, Avg Waiting Time = {avg_waiting_time:.2f}, Avg Speed = {avg_speed}")
                log_writer.writerow([agent_name, hidden_dim, episode + 1, total_rewards, avg_speed, avg_waiting_time])

                # Save the best model for this agent and hidden layer size
                if total_rewards > best_reward:
                    best_reward = total_rewards
                    torch.save(agent.model.state_dict(), best_model_path)
                    print(f"New best model for {agent_name} with hidden_dim {hidden_dim} saved.")

env.close()
print("Training complete. Best models saved in:", models_dir)
