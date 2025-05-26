import os

import torch
import traci
import sumolib
from dqn_per import DQNPERAgent
import numpy as np


def find_neighbors(network_file, intersection_id):
    """
    Finds the neighboring traffic lights for a given intersection.

    Parameters:
        network_file (str): Path to the SUMO network file (.net.xml).
        intersection_id (str): The ID of the intersection.

    Returns:
        list: A list of neighboring traffic light IDs.
    """
    net = sumolib.net.readNet(network_file)
    neighbors = set()

    # Find the target intersection in the network
    target_node = net.getNode(intersection_id)

    if not target_node:
        print(f"Warning: Intersection {intersection_id} not found in network.")
        return []

    # Iterate over outgoing and incoming edges of the intersection
    for edge in target_node.getOutgoing() + target_node.getIncoming():
        # Get the node at the other end of the edge
        neighbor_node = edge.getToNode() if edge.getFromNode().getID() == intersection_id else edge.getFromNode()

        # Ensure that the neighbor is also a traffic light
        if neighbor_node.getType() == "traffic_light":
            neighbors.add(neighbor_node.getID())

    return list(neighbors)


class Intersection:
    """
        Initializes a traffic light-controlled intersection object with:
            - Traci-based state access,
            - Traffic light phase duration control,
            - Internal lane detection,
            - RL agent initialization,
            - Model loading if applicable.

        Parameters:
            intersection_id (str): The unique ID of the intersection from the SUMO network.
            model_dir (str): Optional directory path for loading a pretrained model.
    """
    def __init__(self, intersection_id, model_dir=""):
        self.id = intersection_id
        self.lane_ids: tuple = tuple(set(traci.trafficlight.getControlledLanes(self.id)))
        self.num_phases: int = len(traci.trafficlight.getAllProgramLogics(self.id)[0].phases)

        self.phase_durations: list[int] = []
        program_logic = traci.trafficlight.getAllProgramLogics(self.id)[0]  # Get traffic light program
        for phase in program_logic.phases:
            self.phase_durations.append(phase.duration)

        NETWORK_FILE = os.path.join("..", "sumo_simulation", "Square.net.xml")
        self.neighbors = find_neighbors(NETWORK_FILE, self.id)
        print(f"Intersection id: {self.id} has neighbors: {self.neighbors}")

        self.phase_time: int = 0
        self.min_green_time: int = 10

        self.internal_lanes = set()
        for link_group in traci.trafficlight.getControlledLinks(self.id):
            for link in link_group:
                if link and len(link) > 1:
                    via_lane = link[2]
                    if via_lane.startswith(":"):  # internal lanes typically start with ':'
                        self.internal_lanes.add(via_lane)
        self.previous_internal_vehicles = set()

        # Initialize a separate RL agent for this intersection
        state_dim = len(self.lane_ids) * 2 + 4
        action_dim = 3  # Actions: Modify green/yellow durations
        self.agent = DQNPERAgent(state_dim=state_dim, action_dim=action_dim)

        print(f"Lane id: {self.id}")
        print(f"Connected lane ids: {self.lane_ids}")
        print(f"State dim: {state_dim}")
        print(f"Initial phase durations: {self.phase_durations}")

        self.model_path = os.path.join(model_dir, "best_model_DQNPER_64.pth")
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path)
                self.agent.model.load_state_dict(checkpoint)
                print(f"Loaded pre-trained model for intersection {self.id} from {self.model_path}")
            except Exception as e:
                print(f"ERROR loading model {self.model_path} for {self.id}: {e}, skipping model load.")

    def get_state(self) -> np.ndarray:
        """
            Constructs and returns the current normalized state vector of the intersection.

            The state includes:
                - Number of cars per lane,
                - New vehicles passed through the intersection (throughput),
                - Waiting vehicle count,
                - Current traffic light phase and simulation time,
                - Neighbor traffic light states,
                - Number of halted cars per lane.

            Returns:
                np.ndarray: A flattened and normalized feature vector representing the full current state.
        """

        num_cars: list[int] = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lane_ids]
        phase = traci.trafficlight.getPhase(self.id)
        waiting_cars: list[int] = [traci.lane.getLastStepHaltingNumber(lane) for lane in self.lane_ids]

        current_internal_vehicles = set()
        for lane in self.internal_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            current_internal_vehicles.update(vehicles)

        # Throughput = how many new vehicles entered the intersection since last step
        new_vehicles_passed = len(current_internal_vehicles - self.previous_internal_vehicles)

        waiting_score = 0
        for lane in self.lane_ids:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for v_id in vehicles:
                if traci.vehicle.getSpeed(v_id) < 0.1:
                    waiting_score += 1

        return np.array(num_cars + [new_vehicles_passed, waiting_score] + [phase, traci.simulation.getTime()] + waiting_cars, dtype=np.float32)

    def step(self):
        """
            Advances the simulation logic for the intersection by one decision step according to its RL agent's decision.
            Performs the following:
                - Retrieves current state.
                - Selects action via the agent.
                - Optionally adjusts green phase duration.
                - Switches phases when thresholds are met.
                - Calculates reward as a combination of throughput and waiting time.
                - Feeds transition to the agent for learning.

            Returns:
                tuple:
                    float: Average waiting time on incoming lanes.
                    float: Average vehicle speed on incoming lanes.
                    float: Calculated reward for this step.
        """

        current_phase = traci.trafficlight.getPhase(self.id)
        state: np.ndarray = self.get_state()
        action: int = self.agent.act(state)

        # Increase phase time counter
        self.phase_time += 1
        if self.phase_time >= self.phase_durations[current_phase]:
            if current_phase in [0, 2]:  # Green phases
                if action == 1 and self.phase_time >= self.min_green_time:
                    self.phase_durations[current_phase] = min(self.phase_durations[current_phase] + 1, 60)
                elif action == 2 and self.phase_time >= self.min_green_time:
                    self.phase_durations[current_phase] = max(self.phase_durations[current_phase] - 1, 10)
            # Check if phase has changed
            new_phase: int = (current_phase + 1) % self.num_phases

            # Apply phase change with enforced durations
            traci.trafficlight.setPhase(self.id, new_phase)
            traci.trafficlight.setPhaseDuration(self.id, self.phase_durations[new_phase])
            self.phase_time = 0

        new_state: np.ndarray = self.get_state()
        waiting_times = [traci.lane.getWaitingTime(lane) for lane in self.lane_ids]
        total_waiting_time = sum(waiting_times)
        avg_waiting_time = total_waiting_time / len(self.lane_ids) if self.lane_ids else 0
        avg_speed = sum(traci.lane.getLastStepMeanSpeed(lane) for lane in self.lane_ids) / len(self.lane_ids)

        current_internal_vehicles = set()
        for lane in self.internal_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            current_internal_vehicles.update(vehicles)

        # Throughput = how many new vehicles entered the intersection since last step
        new_vehicles_passed = len(current_internal_vehicles - self.previous_internal_vehicles)
        self.previous_internal_vehicles = current_internal_vehicles

        waiting_score = 0
        for lane in self.lane_ids:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for v_id in vehicles:
                if traci.vehicle.getSpeed(v_id) < 0.1:
                    waiting_score += 1

        # Final reward = encourage high throughput, penalize high waiting
        reward = -waiting_score + new_vehicles_passed

        self.agent.remember(state, action, reward, new_state)
        return avg_waiting_time, avg_speed, reward

    def train_agent(self):
        """
            Triggers training of the RL agent associated with this intersection,
            using transitions stored in the prioritized replay buffer.
        """
        self.agent.train()
