import pandas as pd
import numpy as np
import psutil
import os
import time
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# ── LOAD PRE-TRAINED MODEL & SCALER ───────────────────────────────────────────
MODEL_DIR = 'AI_Analyzer/models'
print("Loading pre-trained Random Forest model...")
model  = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
model.verbose = 0
model.n_jobs  = 1  # single-threaded for per-prediction latency measurement
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
print(f"✅ Model loaded from {MODEL_DIR}/")

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
    flow_count=('duration', 'count'),
    avg_duration=('duration', 'mean'),
    src_bytes=('src_bytes', 'sum'),
    dst_bytes=('dst_bytes', 'sum'),
    label=('label', 'max')
).reset_index()
windowed['avg_bytes'] = (windowed['src_bytes'] + windowed['dst_bytes']) / windowed['flow_count']

FEATURES = ['flow_count', 'avg_duration', 'avg_bytes']

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
        score = model.predict_proba(X_scaled)[0, 1]  # attack probability
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

# ── BASELINE MEASUREMENTS (benign windows) ────────────────────────────────────
baseline_df = windowed[windowed['label'] == 0]
print(f"📊 Baseline: {len(baseline_df)} benign windows")

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

# ── THRESHOLD ANALYSIS (RF probability threshold sensitivity) ─────────────────
# For RF the tunable parameter is the decision probability threshold
# (default 0.5). Sweeping it shows the precision/recall/F1 trade-off.
print("\nThreshold Analysis (RF probability threshold sweep):")
prob_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
threshold_rows = []

X_all = scaler.transform(windowed[FEATURES].values)
y_true_all = windowed['label'].values
proba_all = model.predict_proba(X_all)[:, 1]

for t in prob_thresholds:
    y_pred = (proba_all >= t).astype(int)
    prec = precision_score(y_true_all, y_pred, zero_division=0)
    rec  = recall_score(y_true_all, y_pred, zero_division=0)
    f1   = f1_score(y_true_all, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true_all, proba_all)
    except ValueError:
        auc = float('nan')

    marker = "✅ DEFAULT" if t == 0.50 else ""
    print(f"  threshold={t:.2f} → Prec={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f} {marker}")
    threshold_rows.append({'contamination': t, 'precision': prec, 'recall': rec, 'f1_score': f1, 'roc_auc': auc})

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv('threshold_analysis.csv', index=False)

# ── PLOT ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Resource Profile: Random Forest on ToN-IoT Dataset', fontsize=14, fontweight='bold')

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

# 4. RF probability threshold sensitivity
ax = axes[1, 1]
ax.plot(threshold_df['contamination'], threshold_df['f1_score'],
        marker='o', color='steelblue', linewidth=2, label='F1-Score')
ax.plot(threshold_df['contamination'], threshold_df['precision'],
        marker='s', color='tomato', linewidth=2, linestyle='--', label='Precision')
ax.plot(threshold_df['contamination'], threshold_df['recall'],
        marker='^', color='seagreen', linewidth=2, linestyle=':', label='Recall')
ax.axvline(x=0.50, color='gray', linestyle=':', linewidth=1, label='Default (0.50)')
ax.set_title('RF Probability Threshold Sensitivity')
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()
plt.savefig('resource_profile_comparison.png', dpi=150, bbox_inches='tight')

print("\nFiles saved:")
print("  ✓ resource_profile_results.csv")
print("  ✓ threshold_analysis.csv")
print("  ✓ resource_profile_comparison.png")
