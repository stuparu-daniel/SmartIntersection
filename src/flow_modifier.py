import xml.etree.ElementTree as ET
import random
from constants import HIDDEN_LAYER_SIZES


def get_num_flows(xml_file: str) -> int:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    flows = root.findall(".//flow")

    return len(flows)
def generate_vehicle_flow_list(num_of_episodes: int, num_of_flows: int, min_traffic: int = 50, max_traffic: int = 250) -> list[list[int]]:
    """
    Generates a fixed list of vehsPerHour values for a given number of episodes and flows.

    Parameters:
        num_of_episodes (int): Number of training episodes.
        num_of_flows (int): Number of flow entries in the XML file.
        min_traffic (int, = 50) : Minimum traffic value for the vehicle flow.
        max_traffic (int, = 250) : Maximum traffic value for the vehicle flow.

    Returns:
        list[list[int]]: A list containing 'num_of_episodes' lists,
                         where each sublist contains 'num_of_flows' random traffic values
                         in the range [min_traffic, max_traffic].
    """

    return [[random.randint(min_traffic, max_traffic) for _ in range(num_of_flows)] for _ in range(num_of_episodes)]

def modify_vehicle_flow(xml_file: str, traffic_flow_list: list[list[int]], episode: int):
    """
    Modifies the vehsPerHour values in the SUMO route file with random values
    in the range [min_value, max_value] before each training episode.
    """
    if episode >= len(traffic_flow_list):
        raise ValueError('Episode is out of range')

    flow_values = traffic_flow_list[episode]

    tree = ET.parse(xml_file)
    root = tree.getroot()

    flows = root.findall(".//flow")
    if len(flows) != len(flow_values):
        raise ValueError('Number of flows does not match')

    for flow, vehs_per_hour in zip(flows, flow_values):
        flow.set("vehsPerHour", str(vehs_per_hour))

    tree.write(xml_file, encoding="UTF-8", xml_declaration=True)
    print(f"Updated {xml_file} with new vehsPerHour values.")
