# 🚦 Traffic Light Optimization using Deep Reinforcement Learning

## 📌 Overview  
This project leverages **Deep Q-Networks with Prioritized Experience Replay (DQNPER)** to dynamically optimize **traffic light phases** in a SUMO-based simulation. The goal is to **reduce average waiting time** and **increase vehicle throughput and speed** across intersections by learning intelligent timing adjustments.

## 🏗️ Architecture  

### **1️⃣ Simulation Environment (`sumo_wrapper.py`, `intersection.py`)**
- **`sumo_wrapper.py`** manages the SUMO simulation lifecycle and aggregates data across intersections.
- **`intersection.py`** defines each intersection as a reinforcement learning environment:
  - Constructs the **state vector** using:
    - Vehicle counts,
    - Waiting vehicles,
    - Phase info,
    - Neighboring intersections’ states,
    - Throughput (vehicles passed),
  - Executes **actions** to adjust green phase durations: increase, decrease, or keep constant.
  - Computes **reward** as:  
    ```
    Reward = +throughput - waiting_penalty
    ```

### **2️⃣ Reinforcement Learning Agent (`dqn_per.py`)**
The RL model is a **Deep Q-Network with Prioritized Experience Replay**, structured as:
```plaintext
Input Layer  ➝  FC (Hidden Layer 1, 128 neurons)  ➝  ReLU
             ➝  FC (Hidden Layer 2, 128 neurons)  ➝  ReLU
             ➝  FC (Hidden Layer 3, 128 neurons)  ➝  ReLU
             ➝  FC (Output Layer, 3 Actions)      ➝  Q-Value Predictions
```
- **Input:** Full traffic state from the intersection (vehicle counts, phase info, throughput, neighbors)
- **Output:** One of three discrete actions:
  - Keep current green duration
  - Increase current green duration
  - Decrease current green duration
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- Uses **importance sampling** and **TD error-based prioritization** to improve learning efficiency.

### **3️⃣ Training Loop (`train.py`)**
- Iterates for a number of episodes (EPISODES in constants.py)
- Each episode:
  - Runs a full SUMO simulation 
  - Collects and logs: average waiting time, speed, reward 
  - Trains all intersection agents 
  - Saves the **best-performing models** based on total reward 
- Output is stored in the logs/ and intersections_* folders.

## 📊 Visualization
- **Graphs training results** using `visualize.py`.
- Uses visualize.py to plot training results:
  - Total rewards 
  - Average vehicle speed 
  - Average waiting time 
- Compares learned models with fixed-time control from reference.py 
- Applies smoothing (moving average) to make trends clearer

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
 ├── 🧠 train.py               # Training loop for RL agent
 ├── 🤖 dqn_per.py            # DQNPER model with prioritized replay
 ├── 🚦 sumo_wrapper.py       # Manages simulation and intersection updates
 ├── 🧩 intersection.py       # Encodes intersection state/action/reward logic
 ├── 📊 visualize.py          # Graphs training results
 ├── 🧪 reference.py          # Baseline simulation (fixed timing)
 ├── 🛠️ util.py               # SUMO command helper
 ├── 📜 constants.py          # Simulation config constants
 ├── 📂 logs/                 # Training logs (CSV)
 ├── 📂 graphs/               # Output plots
 ├── 📂 use_models/           # Pretrained models (optional)
 ├── 📂 intersections_*/     # Best models per episode
 ├── 📂 sumo_simulation/     # SUMO network and config files
 │   ├── Square.sumocfg      # SUMO simulation configuration
 │   ├── Square.net.xml      # Road network file
 │   ├── Square.rou.xml      # Vehicle routes (fixed)
```

## ⚙️ Dependencies
- **Python 3.10+**
- **SUMO** (Simulation of Urban Mobility)
- `traci` (SUMO Python API)
- `numpy`
- `torch`
- `matplotlib`
