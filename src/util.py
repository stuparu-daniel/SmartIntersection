import constants


def create_sumo_command(config_path: str) -> list[str]:
    result = []
    if constants.GUI:
        result.append("sumo-gui")
    else:
        result.append("sumo")
    result.append("-c")
    result.append(config_path)
    result.append("--no-warnings")
    if constants.DEV_MODE:
        result.append("--step-length")
        result.append(str(constants.STEP_LENGTH))
        result.append("--delay")
        result.append(str(constants.DELAY))
    else:
        result.append("--no-step-log")

    return result
