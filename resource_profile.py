import pandas as pd
import numpy as np
import psutil
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# ── LOAD & PREPARE ────────────────────────────────────────────────────────────
print("Loading ToN-IoT dataset...")
df = pd.read_csv('data/Network_dataset_1.csv', low_memory=False)
print(f"✅ Loaded {len(df):,} flows from ToN-IoT dataset")

for col in ['src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace('-', '0'),
                             errors='coerce').fillna(0)

print("⏱️  Normalizing timestamps to T=0...")
df['ts'] = df['ts'] - df['ts'].min()
df = df.sort_values('ts').reset_index(drop=True)

if 'duration' not in df.columns:
    raise ValueError("Dataset missing 'duration' column — cannot filter valid flows")

df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)
df_valid = df[df['duration'] > 0].copy()
if len(df_valid) == 0:
    raise ValueError("No valid flows after filtering duration > 0")
print(f"🔄 After duration > 0 filter: {len(df_valid):,} valid flows retained")

# ── WINDOW AGGREGATION (10-second windows) ────────────────────────────────────
df_valid['window'] = (df_valid['ts'] / 10).astype(int)
windowed = df_valid.groupby('window').agg(
    flow_count=('src_pkts', 'count'),
    duration=('duration', 'mean'),
    src_bytes=('src_bytes', 'sum'),
    dst_bytes=('dst_bytes', 'sum'),
    label=('label', 'max')
).reset_index()
windowed['avg_bytes'] = (windowed['src_bytes'] + windowed['dst_bytes']) / windowed['flow_count']

FEATURES = ['flow_count', 'duration', 'avg_bytes']

# ── WARM-UP PHASE (first 60s of benign flows only) ────────────────────────────
warmup_df = windowed[(windowed['window'] < 60) & (windowed['label'] == 0)]
print(f"📊 Warm-up: {len(warmup_df)} benign windows (0-600s)")

if len(warmup_df) == 0:
    raise ValueError("No benign warm-up data available in the first 60 seconds")

X_train = warmup_df[FEATURES].values
scaler = StandardScaler()
scaler.fit(X_train)
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(scaler.transform(X_train))

# ── INFERENCE HELPER ──────────────────────────────────────────────────────────
proc = psutil.Process(os.getpid())

def measure_inference(rows_df):
    results = []
    for _, row in rows_df.iterrows():
        X_scaled = scaler.transform(np.array(row[FEATURES]).reshape(1, -1))
        cpu = proc.cpu_percent(interval=0.01)
        ram = proc.memory_info().rss / 1024 / 1024
        start = time.perf_counter()
        pred = model.predict(X_scaled)[0]
        latency = (time.perf_counter() - start) * 1000
        score = model.decision_function(X_scaled)[0]
        results.append({'cpu': cpu, 'ram': ram, 'latency': latency, 'score': score, 'pred': pred})
    return results

def aggregate(raw):
    cpu = [m['cpu'] for m in raw]
    ram = [m['ram'] for m in raw]
    lat = [m['latency'] for m in raw]
    return {
        'num_samples': len(raw),
        'cpu_mean': np.mean(cpu), 'cpu_std': np.std(cpu),
        'ram_mean': np.mean(ram), 'ram_std': np.std(ram),
        'latency_mean': np.mean(lat), 'latency_std': np.std(lat),
    }

# ── BASELINE MEASUREMENTS (benign after warm-up) ──────────────────────────────
baseline_df = windowed[(windowed['window'] >= 6) & (windowed['label'] == 0)]
print(f"📊 Baseline: {len(baseline_df)} benign windows (after 60s)")

baseline_raw = measure_inference(baseline_df)
baseline_results = aggregate(baseline_raw)

# ── ATTACK MEASUREMENTS ───────────────────────────────────────────────────────
attack_df = windowed[windowed['label'] == 1]
print(f"📊 Attack: {len(attack_df)} attack windows scattered throughout")

attack_raw = []
if len(attack_df) == 0:
    print("⚠️  No attack flows found — skipping attack phase")
    attack_results = {k: 0.0 for k in baseline_results}
else:
    attack_raw = measure_inference(attack_df)
    attack_results = aggregate(attack_raw)

# ── TIME-TO-ALERT ─────────────────────────────────────────────────────────────
attack_mask = df['label'] == 1
if attack_mask.any():
    first_attack_ts = df_valid[df_valid['label'] == 1]['ts'].min()
    time_to_alert = first_attack_ts - 60
    print(f"⚠️  First attack appears at T+{time_to_alert:.1f}s (after warm-up)")

# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
print("\nResults:")
print("--------")
print(f"CPU Usage:   Baseline {baseline_results['cpu_mean']:.1f}% ± {baseline_results['cpu_std']:.1f}%"
      f"  |  Attack {attack_results['cpu_mean']:.1f}% ± {attack_results['cpu_std']:.1f}%")
print(f"RAM Usage:   Baseline {baseline_results['ram_mean']:.0f} MiB ± {baseline_results['ram_std']:.0f}"
      f"  |  Attack {attack_results['ram_mean']:.0f} MiB ± {attack_results['ram_std']:.0f}")
print(f"Latency:     Baseline {baseline_results['latency_mean']:.2f}ms ± {baseline_results['latency_std']:.2f}ms"
      f"  |  Attack {attack_results['latency_mean']:.2f}ms ± {attack_results['latency_std']:.2f}ms")

# ── SAVE resource_profile_results.csv ─────────────────────────────────────────
results_df = pd.DataFrame({
    'metric': ['cpu_usage', 'ram_usage', 'inference_latency'],
    'baseline_mean': [baseline_results['cpu_mean'], baseline_results['ram_mean'], baseline_results['latency_mean']],
    'baseline_std':  [baseline_results['cpu_std'],  baseline_results['ram_std'],  baseline_results['latency_std']],
    'attack_mean':   [attack_results['cpu_mean'],   attack_results['ram_mean'],   attack_results['latency_mean']],
    'attack_std':    [attack_results['cpu_std'],    attack_results['ram_std'],    attack_results['latency_std']],
    'unit': ['%', 'MiB', 'ms']
})
results_df.to_csv('resource_profile_results.csv', index=False)

# ── THRESHOLD ANALYSIS ────────────────────────────────────────────────────────
print("\nThreshold Analysis:")
contamination_levels = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
threshold_rows = []

eval_df = windowed[windowed['window'] >= 6].copy()
y_true = eval_df['label'].values
X_eval = scaler.transform(eval_df[FEATURES].values)

for c in contamination_levels:
    m = IsolationForest(contamination=c, random_state=42)
    m.fit(scaler.transform(X_train))
    preds = m.predict(X_eval)
    scores = m.decision_function(X_eval)
    y_pred = (preds == -1).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, -scores)
    except ValueError:
        auc = float('nan')

    marker = "✅ RECOMMENDED" if c == 0.05 else ""
    print(f"  contamination={c} → F1={f1:.3f}, ROC-AUC={auc:.3f} {marker}")
    threshold_rows.append({'contamination': c, 'precision': prec, 'recall': rec, 'f1_score': f1, 'roc_auc': auc})

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv('threshold_analysis.csv', index=False)

# ── PLOT ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Resource Profile: Isolation Forest on ToN-IoT Dataset', fontsize=14, fontweight='bold')

categories = ['Baseline', 'Attack']

# 1. CPU usage
ax = axes[0, 0]
ax.bar(categories,
       [baseline_results['cpu_mean'], attack_results['cpu_mean']],
       yerr=[baseline_results['cpu_std'], attack_results['cpu_std']],
       capsize=5, color=['steelblue', 'tomato'], width=0.4)
ax.set_title('CPU Usage: Baseline vs Attack')
ax.set_ylabel('CPU (%)')
ax.set_ylim(bottom=0)

# 2. RAM usage
ax = axes[0, 1]
ax.bar(categories,
       [baseline_results['ram_mean'], attack_results['ram_mean']],
       yerr=[baseline_results['ram_std'], attack_results['ram_std']],
       capsize=5, color=['steelblue', 'tomato'], width=0.4)
ax.set_title('RAM Usage: Baseline vs Attack')
ax.set_ylabel('RAM (MiB)')
ax.set_ylim(bottom=0)

# 3. Inference latency distribution
ax = axes[1, 0]
baseline_lats = [m['latency'] for m in baseline_raw]
ax.hist(baseline_lats, bins=30, alpha=0.6, color='steelblue', label='Baseline')
if attack_raw:
    attack_lats = [m['latency'] for m in attack_raw]
    ax.hist(attack_lats, bins=30, alpha=0.6, color='tomato', label='Attack')
ax.set_title('Inference Latency Distribution')
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Frequency')
ax.legend()

# 4. Contamination sensitivity (F1 + ROC-AUC)
ax = axes[1, 1]
ax.plot(threshold_df['contamination'], threshold_df['f1_score'],
        marker='o', color='steelblue', linewidth=2, label='F1-Score')
ax.plot(threshold_df['contamination'], threshold_df['roc_auc'],
        marker='s', color='tomato', linewidth=2, linestyle='--', label='ROC-AUC')
ax.axvline(x=0.01, color='gray', linestyle=':', linewidth=1, label='Default (0.01)')
ax.set_title('Contamination Sensitivity')
ax.set_xlabel('Contamination Parameter')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()
plt.savefig('resource_profile_comparison.png', dpi=150, bbox_inches='tight')

print("\nFiles saved:")
print("  ✓ resource_profile_results.csv")
print("  ✓ threshold_analysis.csv")
print("  ✓ resource_profile_comparison.png")
