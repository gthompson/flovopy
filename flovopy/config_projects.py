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
        "seisan_top": None,
        "box": None,
        "enhanced_results": None,
        "repo": None,
        "mvo_seiscomp_db": None,
        "mvo_seisan_index_db": None,  
        "inventory": None,
        "miniseed_top": None,
        "json_top": None,
        "sds_top": None,
        "event_db": 'MVOE_',
        "continuous_db": 'DSNC_',
    }

    # --- Mac-specific config ---
    if sysname == "darwin":
        tachyon = os.path.join("/Volumes", "tachyon", "from_hal")
        config.update({
            "box": os.path.join(config["home"], 'Library', 'CloudStorage', 'Box-Box'),
            "seisan_top": os.path.join(tachyon, "SEISAN_DB"),
            "enhanced_results": os.path.join(tachyon, "b18_waveform_processing"),
            "sds_top": os.path.join(tachyon, "SDS_RAW"),
            "sds_vel": os.path.join(tachyon, "SDS_VEL"),

        })

    # --- Ubuntu-specific config ---
    elif sysname == "linux":
        config.update({
            "seisan_top": os.path.join("/data", "SEISAN_DB"),
            "enhanced_results": "/data/b18_waveform_processing",
            "sds_top": os.path.join("/data", "SDS_RAW"),
            "sds_vel": os.path.join("/data", "SDS_VEL"),
        })
    
    config["dropbox"] = os.path.join(config["home"], "Dropbox")
    config["repo"] = os.path.join(config["home"], "Developer", "flovopy-test")
    config["mvo_seiscomp_db"] = os.path.join(config["seisan_top"], "mvo_seiscomp_db.sqlite")
    #config["mvo_seisan_index_db"] = os.path.join(config["seisan_top"], "mvo_seisan_index_db.sqlite")
    config["mvo_seisan_index_db"] = os.path.join(config["seisan_top"], "index_mvoe4.sqlite")
    config["inventory"] = os.path.join(config['seisan_top'], 'CAL', 'MV.xml')
    config["miniseed_top"] = os.path.join(config['seisan_top'], 'miniseed')
    config["json_top"] = os.path.join(config['seisan_top'], 'json')

    config["mvo_seiscomp_db"] = os.path.join(config["seisan_top"], "seiscomp_db.sqlite")
    #config["mvo_seisan_index_db"] = os.path.join(config["seisan_top"], "mvo_seisan_index_db.sqlite")
    config["mvo_seisan_index_db"] = os.path.join(config["seisan_top"], "seisan_db.sqlite")
    config["inventory"] = os.path.join(config['seisan_top'], 'CAL', 'MV.xml')
    config["miniseed_top"] = os.path.join(config['seisan_top'], 'miniseed2')
    config["json_top"] = os.path.join(config['seisan_top'], 'json2')
    # Add other host-specific overrides here if needed

    return config