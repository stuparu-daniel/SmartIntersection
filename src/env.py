import os
import sys
import traci
import numpy as np
import spdlog as spd
import constants
from util import create_sumo_command
from constants import STEP_LENGTH, TOTAL_UNITS_OF_SIMULATION

class SumoTrafficEnv:
    def __init__(self, sumo_config_path):
        self.sumo_cmd = create_sumo_command(sumo_config_path)
        self.lane_ids = ["-E0_0", "-E0_1", "-E1_0", "-E1_1", "E0_0", "E0_1", "E1_0", "E1_1"]
        self.action_space = 5  # Increase actions for adjusting phase durations
        self.observation_space = len(self.lane_ids) * 2 + 2  # Cars + Speeds + Phase + Step Count
        self.num_phases = 4
        self.phase_durations = [42, 3, 42, 3]  # Initial durations for phases
        self.step_count = 0
        self.step_length = STEP_LENGTH
        self.total_time = TOTAL_UNITS_OF_SIMULATION
        self.steps_to_change_phase = 50  # Empirical value

    def reset(self):
        try:
            traci.close()
        except Exception:
            pass

        traci.start(self.sumo_cmd)
        traci.trafficlight.setPhase("clusterJ4_J5_J6_J7", 0)
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        num_cars = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lane_ids]
        avg_speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in self.lane_ids]
        phase = traci.trafficlight.getPhase("clusterJ4_J5_J6_J7")
        return np.array(num_cars + avg_speeds + [phase, self.step_count], dtype=np.float32)

    def step(self, action):
        current_phase = traci.trafficlight.getPhase("clusterJ4_J5_J6_J7")

        if self.step_count % self.steps_to_change_phase == 0:
            if current_phase in [0, 2]:  # Green phases
                if action == 1:  # Increase green duration
                    self.phase_durations[current_phase] = min(self.phase_durations[current_phase] + 5, 60)
                elif action == 2:  # Decrease green duration
                    self.phase_durations[current_phase] = max(self.phase_durations[current_phase] - 5, 10)
            elif current_phase in [1, 3]:  # Yellow phases
                if action == 3:  # Increase yellow duration
                    self.phase_durations[current_phase] = min(self.phase_durations[current_phase] + 1,
                                                              7)  # Prevents too short yellows
                elif action == 4:  # Decrease yellow duration
                    self.phase_durations[current_phase] = max(self.phase_durations[current_phase] - 1,
                                                              3)  # Minimum yellow duration

            # Determine if we need to switch to the next phase
            if action in [1, 2, 3, 4]:
                new_phase = (current_phase + 1) % self.num_phases

                traci.trafficlight.setPhase("clusterJ4_J5_J6_J7", new_phase)
                traci.trafficlight.setPhaseDuration("clusterJ4_J5_J6_J7", self.phase_durations[new_phase])

        traci.simulationStep()
        self.step_count += 1

        state = self.get_state()
        waiting_times = sum(traci.lane.getWaitingTime(lane) for lane in self.lane_ids)
        avg_speed = sum(traci.lane.getLastStepMeanSpeed(lane) for lane in self.lane_ids) / len(self.lane_ids)
        avg_waiting_time = waiting_times / len(self.lane_ids) if self.lane_ids else 0


        # reward = -(waiting_times - avg_speed * 10)  # Reward higher speeds, penalize waiting
        # !!!!!!!!!!!!! EXPERIMENT !!!!!!!!!!!!!
        reward = -(avg_waiting_time - avg_speed * 10)
        done = False

        return state, reward, done, waiting_times

    def close(self):
        try:
            traci.close()
        except Exception:
            pass