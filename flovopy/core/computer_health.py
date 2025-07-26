import os
import time
import threading
import pandas as pd
from obspy import UTCDateTime
import psutil


def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except Exception as e:
        print(f"Could not read CPU temperature: {e}", flush=True)
        return None
    
def log_memory_usage(prefix=''):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024**2)
    print(f"{prefix} 🧠 Process {os.getpid()} using {mem_mb:.1f} MB RAM")

def pause_if_too_hot(threshold=72.0, max_cooldown_seconds=900):
    temp = get_cpu_temperature()
    if temp is not None and temp >= threshold:
        cooldown_seconds = (temp - threshold) * 30
        cooldown_seconds = min((cooldown_seconds, max_cooldown_seconds))  # Clamp between 30s and 10min
        print(f"🔥 CPU temperature is {temp:.1f}°C — pausing for {cooldown_seconds} seconds to cool down...", flush=True)

        time.sleep(cooldown_seconds)   

def log_cpu_temperature_to_csv(log_path="cpu_temperature_log.csv"):
    temp = get_cpu_temperature()
    if temp is None:
        print("⚠️ Could not read CPU temperature", flush=True)
        return

    timestamp = UTCDateTime().isoformat()
    new_row = pd.DataFrame([{"timestamp": timestamp, "temperature_C": temp}])

    # Append to CSV using pandas
    if os.path.exists(log_path):
        new_row.to_csv(log_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(log_path, mode='w', header=True, index=False)

    #print(f"🌡️ Logged CPU temperature: {temp:.1f}°C at {timestamp}", flush=True)




def start_cpu_logger(interval_sec=30, log_path="cpu_temperature_log.csv"):
    def logger():
        while True:
            log_cpu_temperature_to_csv(log_path)
            time.sleep(interval_sec)
    
    thread = threading.Thread(target=logger, daemon=True)
    thread.start()
