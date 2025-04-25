import os
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import subprocess

SMS_ALERT_ENABLED = False  # Set to False by default
SMS_ALERT_THRESH = 85  # temperature threshold (°C)
SMS_GATEWAY = os.environ.get("SYSMON_SMS_GATEWAY")  # e.g., 5551234567@tmomail.net
SMS_SENDER = os.environ.get("SYSMON_SMS_SENDER", "SysMon")
SMS_SENT_ALREADY = False


def send_sms_alert(message):
    if not SMS_GATEWAY:
        print("[WARN] No SMS_GATEWAY set in environment; skipping alert.")
        return

    try:
        subprocess.run(
            ["mail", "-s", f"SysMon Alert from {SMS_SENDER}", SMS_GATEWAY],
            input=message.encode("utf-8"),
            check=True
        )
        print(f"[SMS SENT] to {SMS_GATEWAY}")
    except Exception as e:
        print(f"[SMS ERROR] Failed to send alert: {e}")


def send_test_sms():
    message = f"[TEST] SysMon test alert at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    send_sms_alert(message)


def log_system_status_csv(logfile, rownum=None, cooldown=True):
    global SMS_SENT_ALREADY

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    cpu_perc = psutil.cpu_percent(percpu=True)
    mem_perc = [p.memory_percent() for p in psutil.process_iter(attrs=['memory_percent'])]

    temp = None
    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            temp = temps["coretemp"][0].current
        elif "cpu-thermal" in temps:
            temp = temps["cpu-thermal"][0].current
    except Exception:
        pass

    cpu_core = psutil.Process(os.getpid()).cpu_num()

    warnings = []
    critical = False
    if cpu > 90:
        warnings.append("HIGH_CPU")
    if mem.percent > 90:
        warnings.append("HIGH_RAM")
    if temp:
        if temp > 85:
            warnings.append("CRITICAL_TEMP")
            critical = True
        elif temp > 75:
            warnings.append("HIGH_TEMP")

    # SMS alert
    if SMS_ALERT_ENABLED and not SMS_SENT_ALREADY and temp and temp >= SMS_ALERT_THRESH:
        try:
            msg = f"SysMon ALERT: CPU Temp = {temp:.1f}°C at {now} UTC"
            send_sms_alert(msg)
            SMS_SENT_ALREADY = True
        except Exception as e:
            print(f"[SMS ERROR] Failed to send alert: {e}")

    if cooldown and critical:
        print(f"[WARN] CPU temperature is {temp:.1f}°C — sleeping for 2 minutes to cool.")
        time.sleep(120)

    row = pd.DataFrame([{
        "timestamp": now,
        "rownum": rownum if rownum is not None else -1,
        "cpu_percent": round(cpu, 1),
        "ram_percent": round(mem.percent, 1),
        "ram_mb_used": mem.used // (1024**2),
        "temp_c": round(temp, 1) if temp is not None else None,
        "cpu_core": cpu_core,
        "warnings": ",".join(warnings),
        "cpu_per_core": ",".join([f"{c:.1f}" for c in cpu_perc]),
        "proc_mem_perc": ",".join([f"{m:.2f}" for m in mem_perc[:10]])
    }])

    row.to_csv(logfile, mode='a', index=False, header=not os.path.exists(logfile))


def plot_system_log(csv_path, out_file=None):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    plt.figure(figsize=(12, 6))
    ax = df.plot(x="timestamp", y=["cpu_percent", "ram_percent", "temp_c"], marker='o', ax=plt.gca())
    plt.title("System Resource Monitoring")
    plt.ylabel("Percent / °C")
    plt.grid(True)
    for i, row in df.iterrows():
        if isinstance(row['warnings'], str) and any(w in row['warnings'] for w in ["HIGH_CPU", "HIGH_RAM", "HIGH_TEMP", "CRITICAL_TEMP"]):
            plt.axvline(row['timestamp'], color='red', alpha=0.3)
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System monitor log and plot utility.")
    parser.add_argument("--log", type=str, help="Path to system monitor CSV log file", required=False)
    parser.add_argument("--plot", action="store_true", help="Plot system metrics from log")
    parser.add_argument("--rownum", type=int, help="Row number to log (optional)")
    parser.add_argument("--lognow", action="store_true", help="Log system status now")
    parser.add_argument("--sms-test", action="store_true", help="Send test SMS alert")
    args = parser.parse_args()

    if args.sms_test:
        send_test_sms()

    if args.lognow and args.log:
        log_system_status_csv(args.log, rownum=args.rownum)

    if args.plot and args.log:
        plot_system_log(args.log)

    '''
    python sysmon.py --log /data/system_monitor.csv --lognow --rownum 1200
    python sysmon.py --log /data/system_monitor.csv --plot
    '''