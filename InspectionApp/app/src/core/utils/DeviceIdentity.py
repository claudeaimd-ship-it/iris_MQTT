import socket


def get_device_identity(
    default_plant: str = "unknown_plant",
    default_line: str = "unknown_line",
    default_machine: str = "unknown_machine",
) -> tuple[str, str, str]:
    """
    Derive (plant, line, machine) from the Pi's hostname.

    Expects the hostname to follow the convention ``{plant}-{line}-{machine}``,
    set once at OS level when the Pi is provisioned (e.g. via
    ``sudo hostnamectl set-hostname plant2-mda-haspa01``). This is the same
    hostname Iris already reports in ``/api/info`` for Abigail's fleet
    identification, so no separate config file needs to be maintained per Pi.

    If the machine segment itself contains dashes (e.g. 'haspa-01'), it is
    preserved as-is — only the first two dashes are used as separators.

    Args:
        default_plant (str): Fallback if hostname doesn't match the convention.
        default_line (str): Fallback if hostname doesn't match the convention.
        default_machine (str): Fallback if hostname doesn't match the convention.

    Returns:
        tuple[str, str, str]: (plant, line, machine).
    """
    hostname = socket.gethostname()
    parts = hostname.split("-", 2)

    if len(parts) == 3:
        return parts[0], parts[1], parts[2]

    print(
        f"[WARN] DeviceIdentity: hostname '{hostname}' does not follow the "
        f"'<plant>-<line>-<machine>' convention. Falling back to "
        f"'{default_plant}/{default_line}/{default_machine}'. Rename the Pi "
        f"with 'sudo hostnamectl set-hostname <plant>-<line>-<machine>' to fix this."
    )
    return default_plant, default_line, default_machine