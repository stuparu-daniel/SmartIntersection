import os
import sys
import traci
import numpy as np
import spdlog as spd
import constants
from util import create_sumo_command

class SumoTrafficEnv:
    def __init__(self, sumo_config_path):
        # self.sumo_cmd = ["sumo-gui", "-c", sumo_config_path, "--no-warnings", "--step-length", "0.1", "--delay", "1000"]'
        self.sumo_cmd = create_sumo_command(sumo_config_path)
        self.lane_ids = ["-E0_0", "-E0_1", "-E1_0", "-E1_1", "E0_0", "E0_1", "E1_0", "E1_1"]
        self.action_space = 5  # Increase actions for adjusting phase durations
        self.observation_space = 10
        self.num_phases = 4
        self.phase_durations = [42, 3, 42, 3]  # Initial durations for phases
        self.step_count = 0
        self.step_length = 0.1
        self.total_time = 3600
        self.steps_to_change_phase = 50  # Empirical value

    def reset(self):
        try:
            traci.close()
        except Exception:
            pass

        # print(f"sumo_cmd {self.sumo_cmd}")
        traci.start(self.sumo_cmd)
        traci.trafficlight.setPhase("clusterJ4_J5_J6_J7", 0)
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        num_cars = [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lane_ids]
        phase = traci.trafficlight.getPhase("clusterJ4_J5_J6_J7")
        return np.array(num_cars + [phase, self.step_count], dtype=np.float32)

    def step(self, action):
        current_phase = traci.trafficlight.getPhase("clusterJ4_J5_J6_J7")

        if self.step_count % self.steps_to_change_phase == 0:
            if action == 1:  # Increase green duration
                self.phase_durations[current_phase] = min(self.phase_durations[current_phase] + 5, 60)
            elif action == 2:  # Decrease green duration
                self.phase_durations[current_phase] = max(self.phase_durations[current_phase] - 5, 15)
            elif action == 3:  # Increase yellow duration
                self.phase_durations[1] = min(self.phase_durations[1] + 1, 5)
                self.phase_durations[3] = min(self.phase_durations[3] + 1, 5)
            elif action == 4:  # Decrease yellow duration
                self.phase_durations[1] = max(self.phase_durations[1] - 1, 3)
                self.phase_durations[3] = max(self.phase_durations[3] - 1, 3)

            # 🟢 **Explicitly cycle through phases**
            new_phase = (current_phase + 1) % self.num_phases
            traci.trafficlight.setPhase("clusterJ4_J5_J6_J7", new_phase)
            traci.trafficlight.setPhaseDuration("clusterJ4_J5_J6_J7", self.phase_durations[new_phase])

        traci.simulationStep()
        self.step_count += 1

        state = self.get_state()
        reward = -sum(state[:8])  # Reward fewer cars waiting
        done = False

        return state, reward, done, {}

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

