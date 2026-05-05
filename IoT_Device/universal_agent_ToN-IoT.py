import pandas as pd
import time
import traceback
from prometheus_client import start_http_server, Gauge, Counter

# --- Configuration ---
PHYSICAL_CSV = '/app/data/IoT_Thermostat.csv'
CYBER_CSV = '/app/data/Network_dataset_1.csv'
PROMETHEUS_PORT = 8000

# --- Metrics ---
REAL_TEMP = Gauge('iot_physical_temperature_c', 'Physical Temp', ['device'])
CYBER_PACKET_COUNTER = Counter('iot_cyber_packets_total', 'Packet Count', ['protocol'])
CYBER_PACKET_BYTES = Gauge('iot_cyber_packet_size_bytes', 'Latest Packet Size', ['protocol', 'source', 'destination'])

print(f"Starting ToN-IoT Edge Agent on port {PROMETHEUS_PORT}...")
start_http_server(PROMETHEUS_PORT)

def load_and_normalize_data():
    print(f"Loading Physical Dataset: {PHYSICAL_CSV}...")
    df_phys = pd.read_csv(PHYSICAL_CSV, low_memory=False)

    print(f"Loading Cyber Dataset: {CYBER_CSV}...")
    df_cyber = pd.read_csv(CYBER_CSV, low_memory=False)

    print("Sanitizing Data (Removing corrupt rows and '-' artifacts)...")
    # 1. Clean Cyber '-' artifacts
    metric_cols = ['src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes']
    for col in metric_cols:
        df_cyber[col] = pd.to_numeric(df_cyber[col].astype(str).str.replace('-', '0'), errors='coerce').fillna(0)

    # 2. Clean Physical Date artifacts and drop NaNs
    df_phys['datetime'] = pd.to_datetime(df_phys['date'] + ' ' + df_phys['time'], errors='coerce')
    df_phys = df_phys.dropna(subset=['datetime', 'current_temperature']) # Destroy broken rows
    df_phys['ts'] = df_phys['datetime'].astype('int64') // 10**9

    print("Performing Temporal Normalization (Anchoring to T=0)...")
    df_cyber['ts'] = df_cyber['ts'] - df_cyber['ts'].min()
    df_phys['ts'] = df_phys['ts'] - df_phys['ts'].min()

    print("Cropping datasets to prevent asymmetric burnout...")
    max_shared_time = min(df_cyber['ts'].max(), df_phys['ts'].max())
    df_cyber = df_cyber[df_cyber['ts'] <= max_shared_time]
    df_phys = df_phys[df_phys['ts'] <= max_shared_time]

    df_cyber = df_cyber.sort_values('ts')
    df_phys = df_phys.sort_values('ts')

    phys_records = df_phys[['ts', 'current_temperature']].to_dict('records')
    cyber_records = df_cyber[['ts', 'proto', 'src_ip', 'dst_ip', 'src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes']].to_dict('records')

    return phys_records, cyber_records

def run_synchronized_emulation(phys_records, cyber_records):
    phys_iter = iter(phys_records)
    cyber_iter = iter(cyber_records)

    try:
        first_phys_row = next(phys_iter)
        first_cyber_row = next(cyber_iter)
        time_zero_epoch = first_phys_row['ts']
        print(f"Anchor Established at Normalized EPOCH: {time_zero_epoch}")
    except StopIteration:
        print("Error: Empty datasets.")
        return

    current_cyber_row = first_cyber_row
    current_epoch_window = time_zero_epoch

    REAL_TEMP.labels(device="Thermostat_1").set(float(first_phys_row['current_temperature']))

    # --- The Core Synchronization Loop ---
    for phys_row in phys_iter:
        try:
            row_epoch = phys_row['ts']
            temp = float(phys_row['current_temperature'])

            if row_epoch > current_epoch_window:
                packets_this_second = 0
                while current_cyber_row:
                    translated_cyber_epoch = current_cyber_row['ts']

                    if translated_cyber_epoch <= current_epoch_window:
                        protocol = current_cyber_row['proto']
                        src_ip = current_cyber_row['src_ip']
                        dst_ip = current_cyber_row['dst_ip']

                        flow_packets = int(current_cyber_row['src_pkts']) + int(current_cyber_row['dst_pkts'])
                        flow_bytes = int(current_cyber_row['src_bytes']) + int(current_cyber_row['dst_bytes'])

                        try:
                            CYBER_PACKET_BYTES.labels(protocol=protocol, source=src_ip, destination=dst_ip).set(flow_bytes)
                            CYBER_PACKET_COUNTER.labels(protocol=protocol).inc(flow_packets)
                            packets_this_second += flow_packets
                        except ValueError:
                            pass

                        try:
                            current_cyber_row = next(cyber_iter)
                        except StopIteration:
                            current_cyber_row = None
                            break
                    else:
                        break

                # LOG SILENCER: Only print every 60 seconds so Docker doesn't choke!
                if int(current_epoch_window) % 60 == 0:
                    print(f"[Emulation T+{int(current_epoch_window)}s] Temp: {temp}°C | Pushed {packets_this_second} Cyber Pkts")

                time.sleep(1)
                current_epoch_window = row_epoch

            REAL_TEMP.labels(device="Thermostat_1").set(temp)

        except (ValueError, KeyError):
            continue

if __name__ == '__main__':
    print("--- Beginning ToN-IoT Trace-Driven Emulation ---")
    try:
        phys_data, cyber_data = load_and_normalize_data()
        run_synchronized_emulation(phys_data, cyber_data)
        print("--- Emulation Complete. ---")
    except Exception as e:
        print("\n!!! FATAL CRASH IN AGENT !!!")
        print(traceback.format_exc())

    print("Standing by in idle mode.")
    while True:
        time.sleep(60)
