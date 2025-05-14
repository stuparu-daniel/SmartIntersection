import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import heapq


class PrioritizedReplayBuffer:
    """A prioritized experience replay buffer using a max heap to prioritize experiences with higher TD error."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        """
            Initializes a prioritized replay buffer using a max-heap for storing high-priority transitions.

            Parameters:
                capacity (int): Maximum number of experiences the buffer can store.
                alpha (float): Degree of prioritization (0 = no prioritization, 1 = full prioritization).
        """

        self.capacity: int = capacity
        self.alpha: float = alpha  # How much prioritization is used (0 = uniform, 1 = full prioritization)
        self.buffer = []
        self.position: int = 0

    def add(self, td_error: float, experience: tuple):
        """
            Adds a new experience to the buffer, assigning it a priority based on its TD error.

            Parameters:
                td_error (float): Temporal difference error used to compute priority.
                experience (tuple): A tuple of (state, action, reward, next_state) representing the transition.
        """

        priority: float = (abs(td_error) + 1e-5) ** self.alpha  # Ensure nonzero priority
        heapq.heappush(self.buffer, (-priority, self.position, experience))  # Negative priority for max heap
        self.position += 1

        if len(self.buffer) > self.capacity:
            heapq.heappop(self.buffer)  # Remove the lowest priority experience

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple[list[tuple], torch.FloatTensor]:
        """
            Samples a batch of prioritized experiences from the buffer using importance sampling.

            Parameters:
                batch_size (int): Number of experiences to sample.
                beta (float): Correction factor for importance sampling to reduce bias.

            Returns:
                tuple:
                    - list[tuple]: A batch of experience tuples.
                    - torch.FloatTensor: Corresponding importance sampling weights for each experience.
        """

        experiences = heapq.nlargest(batch_size, self.buffer)  # Get top-priority experiences
        weights = np.array([(-p[0]) for p in experiences])  # Convert negative priority back to positive

        # Normalize weights
        weights = weights / weights.max()
        weights = np.power(weights, -beta)  # Importance sampling correction
        weights = torch.FloatTensor(weights)

        batch = [exp[2] for exp in experiences]  # Extract actual experiences
        return batch, weights

    def __len__(self) -> int:
        """
            Returns the number of experiences currently stored in the buffer.

            Returns:
                int: Number of stored transitions.
        """

        return len(self.buffer)


class DQNPER(nn.Module):
    """
    A feedforward, fully connected neural network with 3 hidden layers.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        """
            Initializes a feedforward DQN network with three hidden layers.

            Parameters:
                input_dim (int): Size of the input state vector.
                output_dim (int): Number of possible actions.
                hidden_dim (int): Number of neurons per hidden layer.
        """

        super(DQNPER, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Performs a forward pass through the network.

            Parameters:
                x (torch.Tensor): Input tensor representing the state.

            Returns:
                torch.Tensor: Output Q-values for each possible action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class DQNPERAgent:
    """
    An agent that uses Deep Q-Learning with Prioritized Experience Replay.
    Trains on transitions selected based on their TD error.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, beta: float = 0.4):
        """
           Initializes the agent that uses a DQN with Prioritized Experience Replay.

           Parameters:
               state_dim (int): Dimension of the state input.
               action_dim (int): Number of discrete actions available.
               hidden_dim (int): Number of hidden units per layer in the neural network.
               beta (float): Initial importance sampling correction factor for PER.
        """

        self.memory = PrioritizedReplayBuffer(10000)  # Replay memory with priorities
        self.gamma: float = 0.95
        self.epsilon: float = 1.0  # Initial exploration probability
        self.epsilon_min: float = 0.01  # Minimum epsilon
        self.epsilon_decay: float = 0.995  # Epsilon decay rate
        self.learning_rate: float = 0.0005  # Learning rate for optimizer
        self.batch_size: int = 64  # Mini-batch size for training
        self.beta: float = beta  # Importance sampling correction factor
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQNPER(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()


    def act(self, state: np.ndarray) -> int:
        """
            Epsilon-greedy action selection.

            Parameters:
                state (np.ndarray): The current environment state.

            Returns:
                int: Selected action index.
        """

        if random.random() < self.epsilon:
            return random.choice(range(self.action_dim))
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return torch.argmax(self.model(state_tensor)).item()

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """
            Stores a transition in the replay buffer and computes its TD error for prioritization.

            Parameters:
                state (np.ndarray): Current state before action.
                action (int): Action taken.
                reward (float): Reward received.
                next_state (np.ndarray): State resulting from the action.
        """

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)

            current_q = self.model(state_tensor)[action]
            target_q = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            td_error = abs(target_q - current_q).item()

        self.memory.add(td_error, (state, action, reward, next_state))

    def train(self):
        """
        Samples prioritized experiences and trains the network using mini-batches.
        """
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
