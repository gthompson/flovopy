import os
import platform
import socket
from pathlib import Path

def get_system_id():
    """Return a unique ID for the current system (hostname or similar)."""
    return socket.gethostname().lower()

def get_config():
    sysname = platform.system().lower()
    hostname = get_system_id()

    config = {
        "os": sysname,
        "hostname": hostname,
        "home": str(Path.home()),
        "dropbox": None,
        "seisan_db": None,
        "b18_waveform_processing": None,
    }

    # --- Mac-specific config ---
    if sysname == "darwin":
        config.update({
            "dropbox": os.path.join(config["home"], "Library", "CloudStorage", "Dropbox"),
            "seisan_db": os.path.join(config["home"], "SEISAN", "DATABASE"),
            "b18_waveform_processing": os.path.join(config["home"], "Developer", "flovopy", "wrappers", "MVOE_conversion_pipeline"),
        })

    # --- Ubuntu-specific config ---
    elif sysname == "linux":
        config.update({
            "dropbox": os.path.join(config["home"], "Dropbox"),
            "seisan_db": os.path.join(config["home"], "SEISAN", "DATABASE"),
            "b18_waveform_processing": "/data/b18_waveform_processing",
        })

    # Add other host-specific overrides here if needed

    return config