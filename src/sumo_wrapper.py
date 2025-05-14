import traci
from intersection import Intersection
from util import create_sumo_command


class SumoWrapper:
    def __init__(self, sumo_config, model_dir=""):
        self.sumo_cmd: list[str] = create_sumo_command(sumo_config)
        traci.start(self.sumo_cmd)

        # Identify all intersections with traffic lights
        self.intersections = {tl: Intersection(tl, model_dir) for tl in traci.trafficlight.getIDList()}

    def reset(self):
        """Resets the simulation and all intersections."""
        try:
            traci.close()
        except Exception:
            pass

        traci.start(self.sumo_cmd)

        return {id_: intersection.get_state() for id_, intersection in self.intersections.items()}

    def step(self):
        """Steps through the SUMO simulation by updating each intersection."""
        if traci.simulation.getMinExpectedNumber() == 0:  # No vehicles left
            print("No vehicles remaining. Ending simulation early.")
            return None, None, None  # Indicate termination

        waiting_times, speeds, rewards = [], [], []
        intersection_rewards = {}
        for intersection in self.intersections.values():
            last_phase = traci.trafficlight.getPhase(intersection.id)
            avg_waiting_time, avg_speed, reward = intersection.step(last_phase)
            waiting_times.append(avg_waiting_time)
            speeds.append(avg_speed)
            rewards.append(reward)

            intersection_rewards[intersection.id] = reward

        traci.simulationStep()
        return sum(waiting_times) / len(waiting_times), sum(speeds) / len(speeds), sum(rewards), intersection_rewards

    def train_agents(self):
        """Trains all intersection agents."""
        for intersection in self.intersections.values():
            intersection.train_agent()

    def close(self):
        """Closes the SUMO simulation."""
        traci.close()
