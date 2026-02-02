# IoT_Device/universal_agent.py
import time
import random
import psutil
from prometheus_client import start_http_server, Gauge, Counter

# --- Configuration ---
DEVICE_ID = "ubi-secure-node-01"
UPDATE_INTERVAL = 5

# --- 1. Universal System Metrics (The "Product" Part) ---
# These mimic standard node_exporter metrics but run inside your Python container
REAL_CPU_USAGE = Gauge('system_cpu_usage_percent', 'Real CPU usage of the container', ['device_id'])
REAL_RAM_USAGE = Gauge('system_ram_usage_bytes', 'Real RAM usage', ['device_id'])
REAL_RAM_TOTAL = Gauge('system_ram_total_bytes', 'Total system RAM', ['device_id'])
NET_BYTES_SENT = Gauge('system_network_transmit_bytes', 'Network bytes sent', ['device_id'])

# --- 2. Thesis Simulation Metrics (The "Research" Part) ---
# Safety Metric: CO2 Level (Simulating physical hazard)
SIM_CO2_LEVEL = Gauge('iot_sensor_co2_ppm', 'Simulated CO2 levels', ['device_id', 'room'])
# Security Metric: Attack Simulation (Simulating Cryptojacking/DDoS)
SIM_ATTACK_MODE = Gauge('iot_security_attack_indicator', '1 if under simulated attack, 0 normal', ['device_id'])

# Counter for login attempts (Brute force detection)
LOGIN_ATTEMPTS = Counter('iot_security_login_attempts_total', 'Total login attempts', ['device_id', 'status'])

def collect_metrics():
    # --- A. Collect REAL Metrics ---
    # This proves your container works as a real monitoring tool
    # Get memory object once
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent()
    net = psutil.net_io_counters().bytes_sent
    
    # Used RAM
    REAL_RAM_USAGE.labels(device_id=DEVICE_ID).set(mem.used)
    # Total RAM
    REAL_RAM_TOTAL.labels(device_id=DEVICE_ID).set(mem.total)
    REAL_CPU_USAGE.labels(device_id=DEVICE_ID).set(cpu)
    NET_BYTES_SENT.labels(device_id=DEVICE_ID).set(net)

    # --- B. Generate SIMULATED Thesis Data ---
    # Normal baselines
    co2 = random.uniform(400, 800)
    is_attack = 0
    
    # 10% Chance to trigger "Anomaly Mode" (The Thesis Simulation)
    if random.random() > 0.90:
        print(f"[{DEVICE_ID}] ⚠️  SIMULATING SECURITY INCIDENT...")
        
        # Scenario 1: Cryptojacking Spike (Fake high CPU load)
        # We report 99% usage to Grafana, even if real CPU is low
        REAL_CPU_USAGE.labels(device_id=DEVICE_ID).set(random.uniform(90, 100))
        is_attack = 1
        
        # Scenario 2: Brute Force Attack
        # Rapidly increment login failures
        for _ in range(5): 
            LOGIN_ATTEMPTS.labels(device_id=DEVICE_ID, status='failed').inc()
            
    # 5% Chance to trigger "Safety Incident" (Physical Hazard)
    elif random.random() > 0.95:
         print(f"[{DEVICE_ID}] ⚠️  SIMULATING SAFETY HAZARD...")
         co2 = random.uniform(2500, 5000) # Dangerous CO2 levels

    SIM_CO2_LEVEL.labels(device_id=DEVICE_ID, room='kitchen').set(co2)
    SIM_ATTACK_MODE.labels(device_id=DEVICE_ID).set(is_attack)

    return cpu, co2

def main():
    print(f"Starting Universal IoT Security Agent on port 8000...")
    start_http_server(8000)
    
    while True:
        cpu, co2 = collect_metrics()
        # print(f"[{DEVICE_ID}] Real CPU: {cpu}% | Sim CO2: {co2:.0f}ppm")
        time.sleep(UPDATE_INTERVAL)

if __name__ == '__main__':
    main()
