import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DoubleDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DoubleDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, tau=0.005, target_update_freq=10):
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online and Target Networks
        self.model = DoubleDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_model = DoubleDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # Sync networks
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.training_steps = 0

    def act(self, state):
        """Epsilon-greedy policy for exploration and exploitation."""
        if random.random() < self.epsilon:
            return random.choice(range(5))
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return torch.argmax(self.model(state_tensor)).item()

    def remember(self, state, action, reward, next_state):
        """Stores experience in replay memory."""
        self.memory.append((state, action, reward, next_state))

    def update_target_network(self):
        """Soft update for the target network (Polyak averaging)."""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self):
        """Trains the Double DQN agent using experience replay."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states = zip(*batch)

        state_tensor = torch.FloatTensor(np.array(batch_states)).to(self.device)
        next_state_tensor = torch.FloatTensor(np.array(batch_next_states)).to(self.device)
        action_tensor = torch.tensor(batch_actions, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_tensor = torch.tensor(batch_rewards, dtype=torch.float32).to(self.device)

        # **Double DQN Update:**
        # Step 1: Select actions using the online model
        best_actions = torch.argmax(self.model(next_state_tensor), dim=1, keepdim=True)

        # Step 2: Evaluate Q-values using the target model
        target_q_values = self.target_model(next_state_tensor).gather(1, best_actions).squeeze()

        # Step 3: Compute the target values
        target_values = reward_tensor + self.gamma * target_q_values

        # Get predicted Q-values for actions taken
        predicted_q_values = self.model(state_tensor).gather(1, action_tensor).squeeze()

        # Compute loss and update online network
        loss = self.criterion(predicted_q_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
