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

        # Read the headers (skip them)
        next(phys_reader, None)
        cyber_headers = next(cyber_reader, None)

        # 1. Establish the Anchor (Time Zero)
        try:
            first_phys_row = next(phys_reader)
            first_cyber_row = next(cyber_reader)

            # The physical dataset has the absolute EPOCH time
            time_zero_epoch = float(first_phys_row[0])

            print(f"Anchor Established at EPOCH: {time_zero_epoch}")

        except StopIteration:
            print("Error: Empty datasets.")
            return

        # Prepare to loop
        current_cyber_row = first_cyber_row

        # Process the very first physical row
        try:
            temp = float(first_phys_row[3])
            ip = first_phys_row[2]
            REAL_TEMP.labels(device_ip=ip).set(temp)
        except (ValueError, IndexError):
            pass

        # 2. The Main Synchronization Loop
        for phys_row in phys_reader:
            try:
                # The "current" second we are emulating
                current_epoch_window = float(phys_row[0])

                # Push the physical metrics for this second
                temp = float(phys_row[3])
                ip = phys_row[2]
                REAL_TEMP.labels(device_ip=ip).set(temp)

                print(f"[Time: {current_epoch_window}] Emulating physical state (Temp: {temp})...")

                # 3. Aggregate all cyber packets that happened in this exact second
                packets_this_second = 0
                while current_cyber_row:
                    # Translate the relative cyber time to absolute EPOCH time
                    relative_cyber_time = float(current_cyber_row[1])
                    translated_cyber_epoch = time_zero_epoch + relative_cyber_time

                    # If this packet happened BEFORE the next physical reading, count it
                    if translated_cyber_epoch <= current_epoch_window:
                        protocol = current_cyber_row[4]
                        length = current_cyber_row[5]
                        src = current_cyber_row[2]
                        dst = current_cyber_row[3]

                        try:
                            CYBER_PACKET_BYTES.labels(protocol=protocol, source=src, destination=dst).set(float(length))
                            CYBER_PACKET_COUNTER.labels(protocol=protocol).inc(1)
                            packets_this_second += 1
                        except ValueError:
                            pass

                        # Grab the next cyber row to check it
                        try:
                            current_cyber_row = next(cyber_reader)
                        except StopIteration:
                            current_cyber_row = None # Reached end of cyber file
                            break
                    else:
                        # This packet belongs to the FUTURE. Stop aggregating and wait for the next physical row.
                        break

                print(f"   -> Aggregated {packets_this_second} packets for this time window.")

                # We emulate real-time pacing. Wait 1 second before processing the next physical window.
                time.sleep(1)

            except (ValueError, IndexError) as e:
                continue # Skip malformed rows

if __name__ == '__main__':
    print("--- Beginning Trace-Driven Emulation ---")
    run_synchronized_emulation()
    print("--- Emulation Complete. Standing by. ---")
    # Keep the script alive so the container doesn't crash,
    # but DON'T loop the data!
    while True:
        time.sleep(60)
