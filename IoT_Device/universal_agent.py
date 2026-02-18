import time
import random
import psutil
from prometheus_client import start_http_server, Gauge, Counter

# --- Configuration ---
DEVICE_ID = "ubi-secure-node-01"
UPDATE_INTERVAL = 5

# --- Metrics Definition ---
# 1. Real System Metrics (The "Product")
REAL_CPU_USAGE = Gauge('system_cpu_usage_percent', 'Real CPU usage', ['device_id'])
REAL_RAM_USAGE = Gauge('system_ram_usage_bytes', 'Real RAM usage', ['device_id'])
REAL_RAM_TOTAL = Gauge('system_ram_total_bytes', 'Total system RAM', ['device_id'])
NET_BYTES_SENT = Gauge('system_network_transmit_bytes', 'Network bytes sent', ['device_id'])

# 2. Scientific Simulation Metrics (The "CicIoT2023" Data)
# We override the real network/cpu data with simulated values when "Attack Mode" is on
# to prove we can detect the *patterns* defined in the dataset.
SIM_ATTACK_TYPE = Gauge('iot_attack_type_code', '0=Benign, 1=Mirai/DDoS, 2=BruteForce', ['device_id'])

# 3. Safety Metrics
SIM_CO2_LEVEL = Gauge('iot_sensor_co2_ppm', 'Simulated CO2 levels', ['device_id', 'room'])

# 4. Security Counters
LOGIN_ATTEMPTS = Counter('iot_security_login_attempts_total', 'Total login attempts', ['device_id', 'status'])

# --- CICIoT2023 ATTACK PROFILES ---
# Reference: High-level behavior extracted from CicIoT2023 CSV features
# (Packet rates, CPU load correlation, and Connection attempts)

def get_benign_behavior():
    """Profile: Normal Smart Home Traffic"""
    # Low, intermittent traffic (Smart bulb sending keep-alive)
    sim_cpu = random.uniform(1, 5)          # Idle CPU
    sim_net = random.uniform(100, 5000)     # Low Bytes/sec
    sim_logins = 0                          # No failed logins
    code = 0
    return sim_cpu, sim_net, sim_logins, code

def get_mirai_behavior():
    """Profile: Mirai Botnet / UDP Flood"""
    # Matches CicIoT2023 'DDoS-UDP_Flood' & 'Mirai-UDPPlain'
    # Characteristic: Extremely high 'flow_packets_s'

    sim_cpu = random.uniform(80, 100)       # CPU Spikes due to packet generation load
    sim_net = random.uniform(5000000, 15000000) # 5MB - 15MB/sec (Massive spike)
    sim_logins = 0
    code = 1
    return sim_cpu, sim_net, sim_logins, code

def get_bruteforce_behavior():
    """Profile: Dictionary Attack / Brute Force"""
    # Matches CicIoT2023 'BruteForce-Web'
    # Characteristic: High 'syn_flag_count' (connection attempts), Low 'flow_bytes'

    sim_cpu = random.uniform(10, 20)        # Slight CPU increase (processing auth requests)
    sim_net = random.uniform(20000, 50000)  # Moderate traffic (Header data only)
    sim_logins = random.randint(5, 20)      # 5-20 Fails PER SECOND
    code = 2
    return sim_cpu, sim_net, sim_logins, code

def collect_metrics():
    # 1. Decide on the Scenario (Time-based for demo purposes)
    # 0-30s: Benign | 30-45s: Mirai | 45-60s: Brute Force
    current_second = int(time.time()) % 60

    if current_second < 30:
        cpu, net, logins, code = get_benign_behavior()
        # print(f"[{DEVICE_ID}] Status: BENIGN (Normal Operation)")
    elif current_second < 45:
        cpu, net, logins, code = get_mirai_behavior()
        print(f"[{DEVICE_ID}] ⚠️  Status: MIRAI BOTNET ATTACK DETECTED")
    else:
        cpu, net, logins, code = get_bruteforce_behavior()
        print(f"[{DEVICE_ID}] ⚠️  Status: BRUTE FORCE ATTACK DETECTED")

    # 2. Push Metrics to Prometheus
    # Note: We overwrite "Real" CPU with "Simulated" CPU during attacks
    # because we need to prove Grafana ALERTS work.

    # Get Real RAM (We can keep this real)
    mem = psutil.virtual_memory()
    REAL_RAM_USAGE.labels(device_id=DEVICE_ID).set(mem.used)
    REAL_RAM_TOTAL.labels(device_id=DEVICE_ID).set(mem.total)

    # Set the Profile Data
    REAL_CPU_USAGE.labels(device_id=DEVICE_ID).set(cpu)
    NET_BYTES_SENT.labels(device_id=DEVICE_ID).set(net)
    SIM_ATTACK_TYPE.labels(device_id=DEVICE_ID).set(code)

    # Handle Logins (Counter needs to increment)
    if logins > 0:
        for _ in range(logins):
            LOGIN_ATTEMPTS.labels(device_id=DEVICE_ID, status='failed').inc()

    # Safety Simulation (Independent of Cyber Attack)
    # 5% chance of CO2 hazard
    co2 = random.uniform(400, 800)
    if random.random() > 0.95:
        co2 = random.uniform(2000, 5000)
    SIM_CO2_LEVEL.labels(device_id=DEVICE_ID, room='kitchen').set(co2)

def main():
    print(f"Starting CicIoT2023-Based Simulator on port 8000...")
    start_http_server(8000)

    while True:
        collect_metrics()
        time.sleep(UPDATE_INTERVAL)

if __name__ == '__main__':
    main()
