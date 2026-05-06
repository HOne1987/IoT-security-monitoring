# Batch 2: Resource Profiling Script (Enhanced)

## Setup
Read CLAUDE.md, then these files:
- `IoT_Device/universal_agent_ToN-IoT.py`
- `AI_Analyzer/detector.py`
- `evaluate.py` (root)
- `data/Network_dataset_1.csv` (the ToN-IoT dataset)

## Task
Create `resource_profile.py` (root directory) that measures system resource usage during normal and attack traffic using ToN-IoT ground truth labels.

---

## Requirements

### 1. Load & Prepare Dataset
- Read `data/Network_dataset_1.csv`
- Clean numeric columns: `src_bytes`, `dst_bytes`, `src_pkts`, `dst_pkts` (convert `-` to 0)
- Normalize timestamps: `ts = ts - ts.min()` (T=0 normalization)
- **CRITICAL**: Filter flows with `duration > 0` ONLY (removes malformed entries)
- Group flows into 10-second windows matching your agent's output
- Separate into: benign (`label=0`), attack (`label=1`)

### 2. Warm-Up Phase (Offline)
- Extract first 60 seconds of benign flows ONLY
- Train `StandardScaler` + `IsolationForest` on this window
- This mirrors your detector's live 60-second warm-up exactly
- Save scaler and model for reuse

### 3. Baseline Measurements
- Run inference on all benign flows (after 60s)
- Measure for each prediction:
  - CPU % (using `psutil.Process(os.getpid()).cpu_percent(interval=0.1)`)
  - RAM (using `psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024` for MiB)
  - Inference latency (milliseconds from predict start to decision)
  - Anomaly score (raw decision value from Isolation Forest)
- Aggregate: mean, std dev, min, max for each metric
- Output: `num_samples`, `cpu_mean`, `cpu_std`, `ram_mean`, `ram_std`, `latency_mean`, `latency_std`

### 4. Attack Measurements
- Same process but on all attack flows (`label=1`)
- Same metrics, same aggregation
- Note: Attack flows may span the entire dataset duration (not just after 60s)

### 5. Time-to-Alert Metric
- Find first row where `label=1` (first attack flow)
- Calculate: `time_to_alert_sec = ts_of_first_attack - 60` (relative to warm-up end)
- Print: `"⚠️  First attack appears at T+{time_to_alert_sec}s (after warm-up)"`

### 6. Threshold Analysis (Optional but Recommended)
- Vary Isolation Forest `contamination` parameter: [0.005, 0.01, 0.02, 0.05]
- For each contamination level:
  - Retrain on warm-up data
  - Evaluate on benign + attack combined
  - Calculate: TP, FP, FN, Precision, Recall, F1, ROC-AUC
- Output as table: `contamination | precision | recall | f1 | roc_auc`

---

## Output Files

### `resource_profile_results.csv`
```
metric,baseline_mean,baseline_std,attack_mean,attack_std,unit
cpu_usage,2.3,0.5,4.1,0.8,%
ram_usage,28,2,31,3,MiB
inference_latency,1.8,0.3,2.1,0.4,ms
```

### `threshold_analysis.csv`
```
contamination,precision,recall,f1_score,roc_auc
0.005,0.92,0.85,0.88,0.94
0.01,0.87,0.91,0.89,0.92
0.02,0.78,0.95,0.85,0.90
0.05,0.65,0.98,0.78,0.88
```

### `resource_profile_comparison.png`
Plots:
1. CPU usage: baseline vs attack (bar chart with error bars)
2. RAM usage: baseline vs attack (bar chart with error bars)
3. Inference latency: distribution histogram (both phases overlaid)
4. Contamination sensitivity: F1-score vs contamination parameter (line chart)

---

## Implementation Hints

```python
import pandas as pd
import numpy as np
import psutil
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/Network_dataset_1.csv', low_memory=False)

# Clean & normalize (as in evaluate.py)
for col in ['src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace('-', '0'), 
                             errors='coerce').fillna(0)
df['ts'] = df['ts'] - df['ts'].min()

# Filter duration > 0
df_valid = df[df['duration'] > 0].copy()

# Window aggregation (10-second windows)
df_valid['window'] = (df_valid['ts'] / 10).astype(int)
windowed = df_valid.groupby('window').agg({
    'flow_count': lambda x: len(x),  # Count of flows per window
    'duration': 'mean',
    'src_bytes': 'sum',
    'dst_bytes': 'sum',
    'label': 'max'
}).reset_index()
windowed['avg_bytes'] = (windowed['src_bytes'] + windowed['dst_bytes']) / windowed['flow_count']

# Warm-up: first 60s of benign data
warmup_window = windowed[(windowed['window'] < 6) & (windowed['label'] == 0)]  # window < 6 means ts < 60
X_train = warmup_window[['flow_count', 'duration', 'avg_bytes']].values
scaler = StandardScaler().fit(X_train)
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(scaler.transform(X_train))

# Baseline: benign after warm-up
baseline_window = windowed[(windowed['window'] >= 6) & (windowed['label'] == 0)]
# Measure CPU/RAM/latency for each prediction
baseline_metrics = []
for _, row in baseline_window.iterrows():
    X = np.array([row['flow_count'], row['duration'], row['avg_bytes']]).reshape(1, -1)
    start = time.perf_counter()
    _ = model.predict(scaler.transform(X))
    latency = (time.perf_counter() - start) * 1000
    
    baseline_metrics.append({
        'cpu': psutil.Process(os.getpid()).cpu_percent(interval=0.01),
        'ram': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
        'latency': latency
    })

# Aggregate baseline
baseline_results = {
    'cpu_mean': np.mean([m['cpu'] for m in baseline_metrics]),
    'cpu_std': np.std([m['cpu'] for m in baseline_metrics]),
    'ram_mean': np.mean([m['ram'] for m in baseline_metrics]),
    'ram_std': np.std([m['ram'] for m in baseline_metrics]),
    'latency_mean': np.mean([m['latency'] for m in baseline_metrics]),
    'latency_std': np.std([m['latency'] for m in baseline_metrics]),
    'num_samples': len(baseline_metrics)
}

# Attack: same process for label=1
# ... (repeat above for attack_window = windowed[windowed['label'] == 1])

# Time-to-alert
first_attack_ts = df[df['label'] == 1]['ts'].min()
time_to_alert = first_attack_ts - 60
print(f"⚠️  First attack appears at T+{time_to_alert:.1f}s")

# Save results
results_df = pd.DataFrame({
    'metric': ['cpu_usage', 'ram_usage', 'inference_latency'],
    'baseline_mean': [baseline_results['cpu_mean'], baseline_results['ram_mean'], baseline_results['latency_mean']],
    'baseline_std': [baseline_results['cpu_std'], baseline_results['ram_std'], baseline_results['latency_std']],
    'attack_mean': [attack_results['cpu_mean'], attack_results['ram_mean'], attack_results['latency_mean']],
    'attack_std': [attack_results['cpu_std'], attack_results['ram_std'], attack_results['latency_std']],
    'unit': ['%', 'MiB', 'ms']
})
results_df.to_csv('resource_profile_results.csv', index=False)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# ... plot code ...
plt.savefig('resource_profile_comparison.png', dpi=150, bbox_inches='tight')
```

---

## Error Handling

- **If no attack flows exist**: Print warning and skip attack phase; save baseline only
- **If psutil fails**: Provide fallback note (optional to measure CPU if unavailable)
- **If dataset malformed**: Clear error message with row count check
- **If all flows have duration=0**: Error: "No valid flows after filtering duration > 0"

---

## Expected Outputs (Example)

```
✅ Loaded 2,847 flows from ToN-IoT dataset
🔄 After duration > 0 filter: 1,203 valid flows retained
⏱️  Normalizing timestamps to T=0...
📊 Warm-up: 47 benign windows (0-60s)
📊 Baseline: 156 benign windows (60-1560s)
📊 Attack: 23 attack windows scattered throughout
⚠️  First attack appears at T+412.3s (after warm-up)

Results:
--------
CPU Usage:      Baseline 2.1% ± 0.4%  |  Attack 3.8% ± 0.7%
RAM Usage:      Baseline 28 MiB ± 2   |  Attack 30 MiB ± 2
Latency:        Baseline 1.7ms ± 0.2ms |  Attack 1.9ms ± 0.3ms

Threshold Analysis:
contamination=0.01 → F1=0.89, ROC-AUC=0.92 ✅ RECOMMENDED

Files saved:
  ✓ resource_profile_results.csv
  ✓ threshold_analysis.csv
  ✓ resource_profile_comparison.png
```

---

## Why This Approach Is Better

✅ **Uses ground truth labels** — No guessing what "attack" means
✅ **Matches your real system** — Offline simulation of live detector behavior
✅ **Quantifies overhead** — Shows resource cost of running AI
✅ **Finds best threshold** — Sensitivity analysis for your examiners
✅ **Honest metrics** — CPU/RAM measured from actual Python process
✅ **Reproducible** — Uses same flow-level features as your agent exports
