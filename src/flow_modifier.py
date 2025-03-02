import xml.etree.ElementTree as ET
import random


def modify_vehicle_flow(xml_file: str, min_value: int, max_value: int):
    """
    Modifies the vehsPerHour values in the SUMO route file with random values
    in the range [min_value, max_value] before each training episode.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for flow in root.findall(".//flow"):
        new_vehs_per_hour = random.randint(min_value, max_value)
        flow.set("vehsPerHour", str(new_vehs_per_hour))

    tree.write(xml_file, encoding="UTF-8", xml_declaration=True)
    print(f"Updated {xml_file} with new vehsPerHour values.")
