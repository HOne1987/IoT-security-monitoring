import pandas as pd
import time
from prometheus_client import start_http_server, Gauge, Counter

# --- Configuration ---
# Update these paths to where your ToN-IoT CSVs are located
PHYSICAL_CSV = '/app/data/IoT_Thermostat.csv'
CYBER_CSV = '/app/data/Network_dataset_1.csv'
PROMETHEUS_PORT = 8000

# --- Metrics ---
# Note: ToN-IoT Thermostat data doesn't have an IP column, so we just use a generic 'device' label
REAL_TEMP = Gauge('iot_physical_temperature_c', 'Physical Temp', ['device'])
CYBER_PACKET_COUNTER = Counter('iot_cyber_packets_total', 'Packet Count', ['protocol'])
CYBER_PACKET_BYTES = Gauge('iot_cyber_packet_size_bytes', 'Latest Packet Size', ['protocol', 'source', 'destination'])

print(f"Starting ToN-IoT Edge Agent on port {PROMETHEUS_PORT}...")
start_http_server(PROMETHEUS_PORT)

def load_and_normalize_data():
    print(f"Loading Physical Dataset: {PHYSICAL_CSV}...")
    df_phys = pd.read_csv(PHYSICAL_CSV)

    print(f"Loading Cyber Dataset: {CYBER_CSV}...")
    df_cyber = pd.read_csv(CYBER_CSV)

    print("Translating Thermostat string time to UNIX Epoch...")
    # Combine 'date' and 'time' columns into a standard UNIX integer
    df_phys['datetime'] = pd.to_datetime(df_phys['date'] + ' ' + df_phys['time'])
    df_phys['ts'] = df_phys['datetime'].astype('int64') // 10**9

    print("Performing Temporal Normalization (Anchoring to T=0)...")
    # Shift both datasets so they perfectly overlap starting at Time 0
    df_cyber['ts'] = df_cyber['ts'] - df_cyber['ts'].min()
    df_phys['ts'] = df_phys['ts'] - df_phys['ts'].min()

    # Sort them just to be mathematically safe
    df_cyber = df_cyber.sort_values('ts')
    df_phys = df_phys.sort_values('ts')

    # Convert the required columns to lists of dictionaries so the existing while loop runs super fast
    phys_records = df_phys[['ts', 'current_temperature']].to_dict('records')
    cyber_records = df_cyber[['ts', 'proto', 'src_ip', 'dst_ip', 'src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes']].to_dict('records')

    return phys_records, cyber_records

def run_synchronized_emulation(phys_records, cyber_records):
    # Create iterators just like the old csv.reader
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

    # Push the very first physical row (Hardcoded device name since ToN-IoT lacks IPs for sensors)
    REAL_TEMP.labels(device="Thermostat_1").set(float(first_phys_row['current_temperature']))

    # --- The Core Synchronization Loop ---
    for phys_row in phys_iter:
        try:
            row_epoch = phys_row['ts']
            temp = float(phys_row['current_temperature'])

            # If we have moved to the NEXT physical second in the CSV
            if row_epoch > current_epoch_window:

                # 1. Catch up the Cyber packets for the second we just finished
                packets_this_second = 0
                while current_cyber_row:
                    translated_cyber_epoch = current_cyber_row['ts']

                    if translated_cyber_epoch <= current_epoch_window:
                        protocol = current_cyber_row['proto']
                        src_ip = current_cyber_row['src_ip']
                        dst_ip = current_cyber_row['dst_ip']

                        # CRITICAL FIX: ToN-IoT uses "Flows". We must add src_pkts + dst_pkts
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
                        break # Packet belongs to the future, stop aggregating

                print(f"[Normalized Epoch: {current_epoch_window}] Pushed Physical Data | Aggregated {packets_this_second} Cyber Packets")

                # 2. Emulate the 1-second passing ONCE per epoch window
                time.sleep(1)

                # 3. Advance our time window to the new second
                current_epoch_window = row_epoch

            # Push the physical metrics instantly
            REAL_TEMP.labels(device="Thermostat_1").set(temp)

        except (ValueError, KeyError) as e:
            # Catching KeyError just in case a row is malformed
            continue

if __name__ == '__main__':
    print("--- Beginning ToN-IoT Trace-Driven Emulation ---")
    phys_data, cyber_data = load_and_normalize_data()
    run_synchronized_emulation(phys_data, cyber_data)
    print("--- Emulation Complete. Standing by. ---")
    # Keep the script alive so the container doesn't crash
    while True:
        time.sleep(60)
