import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN_Dropout(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DQN_Dropout, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

class DQNDropoutAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN_Dropout(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(5))
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return torch.argmax(self.model(state_tensor)).item()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in batch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)

            target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            predicted = self.model(state_tensor)[action]

            loss = self.criterion(predicted, torch.tensor(target, dtype=torch.float32).to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
