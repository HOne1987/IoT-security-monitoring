import time
import csv
import os
from prometheus_client import start_http_server, Gauge, Counter

# --- Configuration ---
DEVICE_ID = "ubi-secure-node-01"
PHYSICAL_DATASET = 'SensorData500Rows.csv'
CYBER_DATASET = 'temp_00001_20221019140448.pcap_500_rows.csv'

# We inject the headers dynamically since the dataset stripped them out!
PHYSICAL_HEADERS = [
    "Timestamp", "DevID", "DevIP", "Temp", "Humidity", "PIR", "NoiseL", "NoiseM",
    "NoiseH", "NoiseA", "AirQuality", "Accelero", "Gyro", "Pressure", "Lux", "Prox",
    "Motion", "MassPM1.0", "MassPM2.5", "MassPM4.0", "MassPM10", "NumPM0.5",
    "NumPM1.0", "NumPM2.5", "NumPM4.0", "NumPM10", "TPM"
]

# --- 1. Physical Metrics (From Master SensorData) ---
REAL_TEMP = Gauge('iot_physical_temperature_c', 'Real dataset temperature', ['device_ip'])
REAL_HUMIDITY = Gauge('iot_physical_humidity_percent', 'Real dataset humidity', ['device_ip'])
REAL_PRESSURE = Gauge('iot_physical_pressure_hpa', 'Real dataset pressure', ['device_ip'])

# --- 2. Cyber Metrics (From PCAP CSV) ---
CYBER_PACKET_BYTES = Gauge('iot_cyber_packet_size_bytes', 'Size of the network packet', ['protocol', 'source', 'destination'])
CYBER_PACKET_COUNTER = Counter('iot_cyber_packets_total', 'Total packets processed', ['protocol'])

def replay_datasets():
    """Reads both the Headerless Physical CSV and the Cyber CSV simultaneously."""
    if not os.path.exists(PHYSICAL_DATASET) or not os.path.exists(CYBER_DATASET):
        print("Error: Dataset CSV files not found.")
        time.sleep(5)
        return

    print(f"▶️ Starting Dual-Dataset Replay (Logical Sync)...")

    with open(PHYSICAL_DATASET, 'r') as phys_file, open(CYBER_DATASET, 'r') as cyber_file:
        # phys_reader is a standard reader because there are no headers in the file
        phys_reader = csv.reader(phys_file)
        # cyber_reader is a DictReader because it has headers
        cyber_reader = csv.DictReader(cyber_file)

        # zip() pairs them line-by-line logically
        for phys_row_raw, cyber_row in zip(phys_reader, cyber_reader):

            # --- PROCESS PHYSICAL ROW ---
            # Map the raw list to our 27 headers
            phys_row = dict(zip(PHYSICAL_HEADERS, phys_row_raw))

            dev_ip = phys_row.get('DevIP', 'Unknown')

            try:
                # Extract and push the physical safety metrics
                REAL_TEMP.labels(device_ip=dev_ip).set(float(phys_row.get('Temp', 0)))
                REAL_HUMIDITY.labels(device_ip=dev_ip).set(float(phys_row.get('Humidity', 0)))
                REAL_PRESSURE.labels(device_ip=dev_ip).set(float(phys_row.get('Pressure', 0)))
            except ValueError:
                pass

            # --- PROCESS CYBER ROW ---
            protocol = cyber_row.get('_ws.col.Protocol', 'UNKNOWN')
            length_str = cyber_row.get('_ws.col.Length', '0')
            src_ip = cyber_row.get('_ws.col.Source', '0.0.0.0')
            dst_ip = cyber_row.get('_ws.col.Destination', '0.0.0.0')

            try:
                packet_size = float(length_str)
                CYBER_PACKET_BYTES.labels(protocol=protocol, source=src_ip, destination=dst_ip).set(packet_size)

                # --- NEW VOLUMETRIC FLOOD LOGIC ---
                if src_ip == '198.51.100.44':
                    # If it's our injected attacker, simulate a massive flood (5000 packets per second)
                    CYBER_PACKET_COUNTER.labels(protocol=protocol).inc(5000)
                else:
                    # Normal background traffic
                    CYBER_PACKET_COUNTER.labels(protocol=protocol).inc(1)

            except ValueError:
                packet_size = 0

            try:
                packet_size = float(length_str)
                CYBER_PACKET_BYTES.labels(protocol=protocol, source=src_ip, destination=dst_ip).set(packet_size)
                CYBER_PACKET_COUNTER.labels(protocol=protocol).inc()
            except ValueError:
                packet_size = 0

            # Print to terminal
            print(f"[PHYSICAL - {dev_ip}] Temp: {phys_row.get('Temp')}°C  |  [CYBER] {protocol} Packet: {packet_size} bytes")

            time.sleep(1) # Replay at 1 row per second

    print("🔁 Replay finished. Restarting tape...")

def main():
    print(f"Starting 100% Dataset-Driven Agent on port 8000...")
    start_http_server(8000)
    while True:
        replay_datasets()

if __name__ == '__main__':
    main()
