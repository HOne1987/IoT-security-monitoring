import pandas as pd
import numpy as np
import psutil
import os
import time
import matplotlib.pyplot as plt

# Adjust this path for device deployment (e.g. /app/data/Network_dataset_1.csv on Docker)
CYBER_CSV = 'data/Network_dataset_1.csv'

proc = psutil.Process(os.getpid())

# ── PHASE 1: LOAD & NORMALIZE ─────────────────────────────────────────────────
print("Profiling IoT Edge Agent resource usage...")
print(f"Dataset: {CYBER_CSV}\n")

ram_before_load = proc.memory_info().rss / 1024 / 1024
load_start = time.perf_counter()

df = pd.read_csv(CYBER_CSV, low_memory=False)

for col in ['src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace('-', '0'),
                             errors='coerce').fillna(0)

df['ts'] = df['ts'] - df['ts'].min()
df = df.sort_values('ts').reset_index(drop=True)
df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)
df['window'] = (df['ts'] / 10).astype(int)

load_time_ms = (time.perf_counter() - load_start) * 1000
ram_after_load = proc.memory_info().rss / 1024 / 1024
ram_load_delta = ram_after_load - ram_before_load

print(f"✅ Loaded {len(df):,} flows")
print(f"⏱️  Load + normalize time: {load_time_ms:.0f}ms")
print(f"💾 RAM after load: {ram_after_load:.0f} MiB (dataset footprint: +{ram_load_delta:.0f} MiB)\n")

# ── PHASE 2: PER-WINDOW EMULATION LOOP ────────────────────────────────────────
# Mirrors universal_agent_ToN-IoT.py exactly — no sleep, just processing overhead
print("Measuring per-window processing overhead...")
window_metrics = []

for window_id in df['window'].unique():
    window_df = df[df['window'] == window_id]
    valid_flows = window_df[window_df['duration'] > 0]

    cpu = proc.cpu_percent(interval=0.01)
    ram = proc.memory_info().rss / 1024 / 1024
    loop_start = time.perf_counter()

    if len(valid_flows) > 0:
        flow_count = len(valid_flows)
        avg_duration = valid_flows['duration'].mean()
        avg_bytes = (valid_flows['src_bytes'].sum() + valid_flows['dst_bytes'].sum()) / flow_count
    else:
        flow_count = 0
        avg_duration = 0.0
        avg_bytes = 0.0

    has_attack = int(window_df['label'].max())

    loop_latency_ms = (time.perf_counter() - loop_start) * 1000

    window_metrics.append({
        'window_id': window_id,
        'flow_count': flow_count,
        'avg_duration_s': avg_duration,
        'avg_bytes': avg_bytes,
        'attack': has_attack,
        'cpu_pct': cpu,
        'ram_mib': ram,
        'loop_latency_ms': loop_latency_ms,
    })

metrics_df = pd.DataFrame(window_metrics)

# ── RESULTS ───────────────────────────────────────────────────────────────────
print(f"Results ({len(metrics_df):,} windows measured):")
print("--------")
print(f"Load time:    {load_time_ms:.0f}ms")
print(f"Dataset RAM:  +{ram_load_delta:.0f} MiB")
print(f"CPU Usage:    {metrics_df['cpu_pct'].mean():.1f}% ± {metrics_df['cpu_pct'].std():.1f}%"
      f"  (max: {metrics_df['cpu_pct'].max():.1f}%)")
print(f"RAM Usage:    {metrics_df['ram_mib'].mean():.0f} MiB ± {metrics_df['ram_mib'].std():.0f} MiB")
print(f"Loop latency: {metrics_df['loop_latency_ms'].mean():.3f}ms ± {metrics_df['loop_latency_ms'].std():.3f}ms"
      f"  (max: {metrics_df['loop_latency_ms'].max():.3f}ms)")

# ── SAVE CSVs ─────────────────────────────────────────────────────────────────
metrics_df.to_csv('agent_resource_profile.csv', index=False)

summary_df = pd.DataFrame([
    {'metric': 'load_time',      'mean': load_time_ms,                        'std': 0,                                   'unit': 'ms'},
    {'metric': 'ram_load_delta', 'mean': ram_load_delta,                       'std': 0,                                   'unit': 'MiB'},
    {'metric': 'cpu_usage',      'mean': metrics_df['cpu_pct'].mean(),         'std': metrics_df['cpu_pct'].std(),         'unit': '%'},
    {'metric': 'ram_usage',      'mean': metrics_df['ram_mib'].mean(),         'std': metrics_df['ram_mib'].std(),         'unit': 'MiB'},
    {'metric': 'loop_latency',   'mean': metrics_df['loop_latency_ms'].mean(), 'std': metrics_df['loop_latency_ms'].std(), 'unit': 'ms'},
])
summary_df.to_csv('agent_resource_summary.csv', index=False)

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('IoT Edge Agent — Resource Profile', fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(metrics_df['window_id'], metrics_df['cpu_pct'], color='steelblue', linewidth=0.8)
ax.set_title('CPU Usage per Window')
ax.set_xlabel('Window ID (10s each)')
ax.set_ylabel('CPU (%)')
ax.set_ylim(bottom=0)

ax = axes[1]
ax.plot(metrics_df['window_id'], metrics_df['ram_mib'], color='seagreen', linewidth=0.8)
ax.set_title('RAM Usage per Window')
ax.set_xlabel('Window ID (10s each)')
ax.set_ylabel('RAM (MiB)')
ax.set_ylim(bottom=0)

ax = axes[2]
ax.hist(metrics_df['loop_latency_ms'], bins=40, color='steelblue', alpha=0.8)
ax.set_title('Per-Window Processing Latency')
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('agent_resource_profile.png', dpi=150, bbox_inches='tight')

print("\nFiles saved:")
print("  ✓ agent_resource_profile.csv   (per-window measurements)")
print("  ✓ agent_resource_summary.csv   (aggregated stats)")
print("  ✓ agent_resource_profile.png   (CPU, RAM, latency plots)")
