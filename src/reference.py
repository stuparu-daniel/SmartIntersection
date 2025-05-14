import os
import traci
import numpy as np
from constants import TOTAL_UNITS_OF_SIMULATION
from util import create_sumo_command

def run_reference_simulation(sumo_config_path):
    """
    Runs the SUMO simulation with fixed-time traffic light phases (no ML adjustments).
    Records the average waiting time and speed of vehicles.

    Parameters:
        sumo_config_path (str): Path to the SUMO configuration file.

    Returns:
        dict: Contains 'avg_waiting_time' and 'avg_speed' as keys.
    """
    # Start SUMO without ML-based modifications

    sumo_cmd = create_sumo_command(sumo_config_path)
    traci.start(sumo_cmd)

    # Identify all traffic light-controlled intersections
    intersection_ids = traci.trafficlight.getIDList()

    total_waiting_time = 0
    total_speed = 0
    num_steps = 0

    for step in range(TOTAL_UNITS_OF_SIMULATION):
        traci.simulationStep()
        num_steps += 1

        # Compute waiting time and speed
        waiting_times = sum(traci.lane.getWaitingTime(lane) for lane in traci.lane.getIDList())
        avg_speed = sum(traci.lane.getLastStepMeanSpeed(lane) for lane in traci.lane.getIDList()) / max(1, len(traci.lane.getIDList()))

        total_waiting_time += waiting_times
        total_speed += avg_speed

    traci.close()

    avg_waiting_time = total_waiting_time / max(1, num_steps)  # Prevent division by zero
    avg_speed = total_speed / max(1, num_steps)

    print(f"Reference Simulation Results (Fixed-Time Traffic Lights):\n"
          f"Total number of steps: {num_steps}\n"
          f"Total waiting time: {total_waiting_time:.2f} seconds\n"
          f"Average Waiting Time: {avg_waiting_time:.2f} seconds\n"
          f"Average Speed: {avg_speed:.2f} m/s")

    return avg_waiting_time, avg_speed


# Run the reference simulation
SUMO_CONFIG = os.path.join("..", "sumo_simulation", "Square.sumocfg")
run_reference_simulation(SUMO_CONFIG)
