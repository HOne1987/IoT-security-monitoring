# NEW universal_agent_ToN-IoT.py (Flow-Level Design)

import pandas as pd
import time
from prometheus_client import start_http_server, Gauge, Counter

CYBER_CSV = '/app/data/Network_dataset_1.csv'
PROMETHEUS_PORT = 8000

# ── METRICS (Flow-level, not packet-level) ──
FLOW_COUNT = Gauge('iot_cyber_flow_count', 'Active flows per window')
AVG_FLOW_DURATION = Gauge('iot_cyber_avg_flow_duration_sec', 'Avg flow duration')
AVG_FLOW_BYTES = Gauge('iot_cyber_avg_flow_bytes', 'Avg bytes per flow')
ATTACK_LABEL = Gauge('iot_cyber_attack_label', 'Ground truth: 1=attack, 0=normal')

start_http_server(PROMETHEUS_PORT)

def load_and_normalize():
    df = pd.read_csv(CYBER_CSV, low_memory=False)

    # Clean
    for col in ['src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('-', '0'),
                                 errors='coerce').fillna(0)

    # Normalize timestamps to T=0
    df['ts'] = df['ts'] - df['ts'].min()
    df = df.sort_values('ts').reset_index(drop=True)

    return df

def run_emulation(df):
    # Group flows into 10-second windows (coarser granularity for flow data)
    df['window'] = (df['ts'] / 10).astype(int)

    for window_id in df['window'].unique():
        window_df = df[df['window'] == window_id]

        # Flow-level statistics
        flow_count = len(window_df)
        avg_duration = window_df['duration'].mean()
        avg_bytes = (window_df['src_bytes'].sum() + window_df['dst_bytes'].sum()) / max(flow_count, 1)

        # Ground truth
        has_attack = window_df['label'].max()

        # Export metrics
        FLOW_COUNT.set(flow_count)
        AVG_FLOW_DURATION.set(avg_duration)
        AVG_FLOW_BYTES.set(avg_bytes)
        ATTACK_LABEL.set(has_attack)

        print(f"[Window {window_id}] Flows: {flow_count} | "
              f"Avg Duration: {avg_duration:.2f}s | Avg Bytes: {avg_bytes:.0f} | "
              f"Attack: {has_attack}")

        time.sleep(1)  # Simulate real-time streaming

if __name__ == '__main__':
    df = load_and_normalize()
    run_emulation(df)
