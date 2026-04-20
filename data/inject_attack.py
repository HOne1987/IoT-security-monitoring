import csv
import os

PHYSICAL_FILE = 'SensorData500Rows.csv'
CYBER_FILE = 'temp_00001_20221019140448.pcap_35000_rows.csv'

print("Aligning timelines and injecting synchronized Cyber-Physical Attack...")

# --- 1. Get Time Zero and Last Physical Time ---
with open(PHYSICAL_FILE, 'r') as f:
    lines = f.readlines()
    first_phys_row = lines[1].split(',') # Skip header
    time_zero_epoch = float(first_phys_row[0])

    last_phys_row = lines[-1].split(',')
    last_phys_time = float(last_phys_row[0])

# --- 2. Append 50 seconds of 60°C to Physical Data ---
with open(PHYSICAL_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    for i in range(1, 51):
        new_time = last_phys_time + i
        # 27 columns. Index 0: Time, Index 2: IP, Index 3: Temp
        row = [f"{new_time:.6f}", '2', '10.11.12.17', '60.00'] + ['False'] * 23
        writer.writerow(row)

# --- 3. Sync the Cyber Attack to the end of the Physical Time ---
with open(CYBER_FILE, 'r') as f:
    lines = f.readlines()
    last_cyber_row = lines[-1].split(',')
    frame_num = int(last_cyber_row[0])

with open(CYBER_FILE, 'a', newline='') as f:
    writer = csv.writer(f)

    # CRITICAL FIX: Calculate how many seconds into the future the cyber attack must wait
    # so it perfectly aligns with the 3.5-minute physical timestamp!
    cyber_attack_start_relative = last_phys_time - time_zero_epoch

    for second in range(1, 51):
        for packet in range(100):
            frame_num += 1
            # The cyber time is perfectly locked to the physical time anomaly
            new_relative_time = cyber_attack_start_relative + second + (packet * 0.01)
            row = [str(frame_num), f"{new_relative_time:.6f}", '198.51.100.44', '10.11.12.17', 'UDP', '60', 'Mirai UDP Flood']
            writer.writerow(row)

print("✅ Perfect Synchronization Achieved! Attack appended to both files.")
