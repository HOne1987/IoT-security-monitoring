import csv
import time
from prometheus_client import start_http_server, Gauge, Counter

# --- Configuration ---
PHYSICAL_CSV = '/app/data/SensorData500Rows.csv'
CYBER_CSV = '/app/data/temp_00001_20221019140448.pcap_35000_rows.csv'
PROMETHEUS_PORT = 8000

# --- Metrics ---
REAL_TEMP = Gauge('iot_physical_temperature_c', 'Physical Temp', ['device_ip'])
CYBER_PACKET_COUNTER = Counter('iot_cyber_packets_total', 'Packet Count', ['protocol'])
CYBER_PACKET_BYTES = Gauge('iot_cyber_packet_size_bytes', 'Latest Packet Size', ['protocol', 'source', 'destination'])

print(f"Using The Following Physical Dataset {PHYSICAL_CSV}...")
print(f"Using The Following Cyber Dataset {CYBER_CSV}...")
print(f"Starting Highly Synchronized IoT Edge Agent on port {PROMETHEUS_PORT}...")
start_http_server(PROMETHEUS_PORT)

def run_synchronized_emulation():
    with open(PHYSICAL_CSV, 'r') as phys_f, open(CYBER_CSV, 'r') as cyber_f:
        phys_reader = csv.reader(phys_f)
        cyber_reader = csv.reader(cyber_f)

        # Skip headers
        next(phys_reader, None)
        next(cyber_reader, None)

        try:
            first_phys_row = next(phys_reader)
            first_cyber_row = next(cyber_reader)
            time_zero_epoch = float(first_phys_row[0])
            print(f"Anchor Established at EPOCH: {time_zero_epoch}")
        except StopIteration:
            print("Error: Empty datasets.")
            return

        current_cyber_row = first_cyber_row
        current_epoch_window = time_zero_epoch

        # Push the very first physical row
        try:
            REAL_TEMP.labels(device_ip=first_phys_row[2]).set(float(first_phys_row[3]))
        except (ValueError, IndexError):
            pass

        # 2. The Corrected Synchronization Loop
        for phys_row in phys_reader:
            try:
                row_epoch = float(phys_row[0])
                temp = float(phys_row[3])
                ip = phys_row[2]

                # If we have moved to the NEXT physical second in the CSV
                if row_epoch > current_epoch_window:

                    # 1. Catch up the Cyber packets for the second we just finished
                    packets_this_second = 0
                    while current_cyber_row:
                        relative_cyber_time = float(current_cyber_row[1])
                        translated_cyber_epoch = time_zero_epoch + relative_cyber_time

                        if translated_cyber_epoch <= current_epoch_window:
                            protocol = current_cyber_row[4]
                            try:
                                CYBER_PACKET_BYTES.labels(protocol=protocol, source=current_cyber_row[2], destination=current_cyber_row[3]).set(float(current_cyber_row[5]))
                                CYBER_PACKET_COUNTER.labels(protocol=protocol).inc(1)
                                packets_this_second += 1
                            except ValueError:
                                pass

                            try:
                                current_cyber_row = next(cyber_reader)
                            except StopIteration:
                                current_cyber_row = None
                                break
                        else:
                            break # Packet belongs to the future, stop aggregating

                    print(f"[Epoch: {current_epoch_window}] Pushed Physical Data | Aggregated {packets_this_second} Cyber Packets")

                    # 2. Emulate the 1-second passing ONCE per epoch window
                    time.sleep(1)

                    # 3. Advance our time window to the new second
                    current_epoch_window = row_epoch

                # Push the physical metrics (This happens for EVERY row instantly, grouping multiple IPs)
                REAL_TEMP.labels(device_ip=ip).set(temp)

            except (ValueError, IndexError):
                continue

if __name__ == '__main__':
    print("--- Beginning Trace-Driven Emulation ---")
    run_synchronized_emulation()
    print("--- Emulation Complete. Standing by. ---")
    # Keep the script alive so the container doesn't crash,
    # but DON'T loop the data!
    while True:
        time.sleep(60)
