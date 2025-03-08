import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import heapq


class PrioritizedReplayBuffer:
    """A prioritized experience replay buffer using a max heap to prioritize experiences with higher TD error."""

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization is used (0 = uniform, 1 = full prioritization)
        self.buffer = []
        self.position = 0

    def add(self, td_error, experience):
        """Adds an experience with its TD error to the priority queue."""
        priority = (abs(td_error) + 1e-5) ** self.alpha  # Ensure nonzero priority
        heapq.heappush(self.buffer, (-priority, self.position, experience))  # Negative priority for max heap
        self.position += 1

        if len(self.buffer) > self.capacity:
            heapq.heappop(self.buffer)  # Remove lowest priority experience

    def sample(self, batch_size, beta=0.4):
        """Samples experiences based on priority and returns weighted importance-sampling values."""
        experiences = heapq.nlargest(batch_size, self.buffer)  # Get top-priority experiences
        weights = np.array([(-p[0]) for p in experiences])  # Convert negative priority back to positive

        # Normalize weights
        weights = weights / weights.max()
        weights = np.power(weights, -beta)  # Importance sampling correction
        weights = torch.FloatTensor(weights)

        batch = [exp[2] for exp in experiences]  # Extract actual experiences
        return batch, weights

    def __len__(self):
        return len(self.buffer)


class DQNPER(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DQNPER, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNPERAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, beta=0.4):
        self.memory = PrioritizedReplayBuffer(10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQNPER(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.beta = beta  # Importance sampling correction factor

    def act(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(range(5))
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return torch.argmax(self.model(state_tensor)).item()

    def remember(self, state, action, reward, next_state):
        """Stores experience with computed TD error as priority."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)

            current_q = self.model(state_tensor)[action]
            target_q = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            td_error = abs(target_q - current_q).item()

        self.memory.add(td_error, (state, action, reward, next_state))

    def train(self):
        """Samples prioritized experiences and trains the network."""
        if len(self.memory) < self.batch_size:
            return

        batch, weights = self.memory.sample(self.batch_size, self.beta)
        batch_states, batch_actions, batch_rewards, batch_next_states = zip(*batch)

        state_tensor = torch.FloatTensor(np.array(batch_states)).to(self.device)
        next_state_tensor = torch.FloatTensor(np.array(batch_next_states)).to(self.device)
        action_tensor = torch.tensor(batch_actions, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_tensor = torch.tensor(batch_rewards, dtype=torch.float32).to(self.device)

        # Compute Q-learning target using max action from the next state
        target_values = reward_tensor + self.gamma * torch.max(self.model(next_state_tensor), dim=1)[0]

        # Get predicted Q-values for the selected actions
        predicted_values = self.model(state_tensor).gather(1, action_tensor).squeeze()

        # Compute loss with importance-sampling weights
        loss = (weights.to(self.device) * self.criterion(predicted_values, target_values)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
