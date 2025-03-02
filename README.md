# 🚦 Traffic Light Optimization using Deep Reinforcement Learning

## 📌 Overview
This project optimizes traffic light phases using **Deep Reinforcement Learning (DQN)** to minimize **waiting time** at red lights and **maximize vehicle speeds**. The system dynamically adjusts **green and yellow phase durations** in real-time using **SUMO (Simulation of Urban Mobility)**.

## 🏗️ Architecture
### **1️⃣ Environment (`env.py`)**
- Uses **SUMO** to simulate traffic.
- Defines **state representation** (number of vehicles, speeds, current phase, etc.).
- Implements **step function** to interact with SUMO and update the environment.
- Computes **reward function** based on **waiting time reduction** and **speed optimization**.

### **2️⃣ Reinforcement Learning Agent (`dqn.py`)**
The RL model is a **Deep Q-Network (DQN)** with the following architecture (may and sholud be modified as development proceeds):
```plaintext
Input Layer  ➝  Fully Connected (Hidden Layer 1, X neurons)  ➝  ReLU Activation
             ➝  Fully Connected (Hidden Layer 2, X neurons)  ➝  ReLU Activation
             ➝  Fully Connected (Output Layer, 5 Actions)   ➝  Q-Value Predictions
```
- **Input:** Traffic state (number of cars + average speeds per lane + phase info).
- **Output:** Action (adjust green/yellow phase duration or keep it unchanged).
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam

### **3️⃣ Training Loop (`train.py`)**
- Iterates over **different hidden layer sizes** (`64, 128, 256, 512, 1024`) (not definitive as of right now).
- Dynamically **modifies traffic flow (`flow_modifier.py`)** before each episode.
- Saves the best-performing model for each configuration.
- Logs total waiting time and average speed for visualization.

## 📊 Visualization
- **Graphs training results** using `visualize.py`.
- Plots total waiting time for different network configurations.
- Each model’s performance is compared in a single graph.

## 🚀 How to Run
1. Install SUMO and dependencies.
2. Run the training script:
   ```sh
   python train.py
   ```
3. Visualize results:
   ```sh
   python visualize.py
   ```

## 📁 Project Structure
```plaintext
📂 project_root/
 ├── 📜 README.md              # This file
 ├── 📝 train.py               # Training loop for RL agent
 ├── 🤖 dqn.py                 # Deep Q-Network model
 ├── 🚦 env.py                 # SUMO Traffic Simulation Environment
 ├── 📊 visualize.py           # Graphs training results
 ├── 🛠️ util.py                 # Helper functions
 ├── 🏙️ flow_modifier.py        # Modifies vehicle density per episode
 ├── 📜 constants.py           # Stores global constants
 ├── 📂 sumo_simulation/       # SUMO network files
 │   ├──  Test3.sumocfg      # SUMO simulation configuration
 │   ├──  Test3.net.xml      # Road network file
 │   ├──  Test3.rou.xml      # Vehicle routes (dynamically modified)
 ├── 📂 models/                # Saved RL models
 ├── 📂 logs/                  # Training logs
```

## ⚙️ Dependencies
- **Python 3.10+**
- **SUMO** (Simulation of Urban Mobility)
- `traci` (SUMO Python API)
- `numpy`
- `torch`
- `matplotlib`
