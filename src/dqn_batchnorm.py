import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN_BatchNorm(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DQN_BatchNorm, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 1:  # Ensure input is 2D (batch_size, features)
            x = x.unsqueeze(0)

        if self.training and x.shape[0] > 1:  # Only apply BatchNorm if batch size > 1
            x = torch.relu(self.bn1(self.fc1(x)))
            x = torch.relu(self.bn2(self.fc2(x)))
        else:  # Skip BatchNorm in single-sample inference
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))

        return self.fc3(x)


class DQNBatchNormAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN_BatchNorm(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(5))
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Ensure batch dimension
        return torch.argmax(self.model(state_tensor)).item()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states = zip(*batch)

        # Convert list of NumPy arrays to a single NumPy array first, then to a PyTorch tensor
        state_tensor = torch.FloatTensor(np.array(batch_states)).to(self.device)
        next_state_tensor = torch.FloatTensor(np.array(batch_next_states)).to(self.device)

        target = torch.tensor(batch_rewards, dtype=torch.float32).to(self.device) + self.gamma * torch.max(self.model(next_state_tensor), dim=1)[0]

        predicted = self.model(state_tensor).gather(1, torch.tensor(batch_actions, dtype=torch.long).unsqueeze(1).to(
            self.device)).squeeze()

        loss = self.criterion(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
