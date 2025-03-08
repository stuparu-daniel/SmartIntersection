import os
import traci
import numpy as np
from env import SumoTrafficEnv
from constants import TOTAL_UNITS_OF_SIMULATION


def run_reference_simulation(sumo_config_path):
    """
    Runs the SUMO simulation without reinforcement learning and records
    the average waiting time and speed of vehicles.

    Parameters:
        sumo_config_path (str): Path to the SUMO configuration file.

    Returns:
        dict: Contains 'avg_waiting_time' and 'avg_speed' as keys.
    """
    env = SumoTrafficEnv(sumo_config_path)
    total_waiting_time = 0
    total_speed = 0
    num_steps = 0

    env.reset()

    for step in range(TOTAL_UNITS_OF_SIMULATION):
        traci.simulationStep()
        num_steps += 1

        # Calculate waiting time and speed for all lanes
        waiting_times = sum(traci.lane.getWaitingTime(lane) for lane in env.lane_ids)
        avg_speed = sum(traci.lane.getLastStepMeanSpeed(lane) for lane in env.lane_ids) / len(env.lane_ids)

        total_waiting_time += waiting_times
        total_speed += avg_speed

    env.close()

    avg_waiting_time = total_waiting_time / num_steps
    avg_speed = total_speed / num_steps

    print(f"Reference Simulation Results:\n"
          f"Total number of steps: {num_steps}\n"
          f"Total waiting time: {total_waiting_time}\n"
          f"Average Waiting Time: {avg_waiting_time:.2f} seconds\n"
          f"Average Speed: {avg_speed:.2f} m/s")

    return {"avg_waiting_time": avg_waiting_time, "avg_speed": avg_speed}


SUMO_CONFIG = os.path.join("..", "sumo_simulation", "Test3.sumocfg")
run_reference_simulation(SUMO_CONFIG)

