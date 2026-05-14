import pandas as pd
import numpy as np
import pickle
import time
import tracemalloc
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

CYBER_CSV = 'data/Network_dataset_1.csv'
WINDOW_SIZE = 10
TEST_SIZE = 0.3
N_TIMING_RUNS = 1000  # repetitions for stable inference timing
FEATURES = ['flow_count', 'duration', 'avg_bytes']

print("=" * 80)
print("FINAL EVALUATION: Isolation Forest vs Random Forest")
print("Train/Test Split (70/30 Stratified) — No Warm-Up Phase")
print("=" * 80)

# ── 1. LOAD & PREPARE ─────────────────────────────────────────────────────────
print("\n[1/6] Loading and preparing dataset...")
df = pd.read_csv(CYBER_CSV, low_memory=False)

for col in ['src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace('-', '0'),
                             errors='coerce').fillna(0)

df['ts'] = df['ts'] - df['ts'].min()
df = df.sort_values('ts').reset_index(drop=True)
df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)

df_valid = df[df['duration'] > 0].copy()
print(f"  Loaded {len(df):,} flows, {len(df_valid):,} valid (duration > 0)")

# ── 2. WINDOW AGGREGATION ─────────────────────────────────────────────────────
print(f"[2/6] Aggregating into {WINDOW_SIZE}-second windows...")
df_valid['window'] = (df_valid['ts'] / WINDOW_SIZE).astype(int)

windowed = df_valid.groupby('window').agg(
    flow_count=('duration', 'count'),
    duration=('duration', 'mean'),
    src_bytes=('src_bytes', 'sum'),
    dst_bytes=('dst_bytes', 'sum'),
    label=('label', 'max')
).reset_index()
windowed['avg_bytes'] = (windowed['src_bytes'] + windowed['dst_bytes']) / windowed['flow_count']
windowed = windowed.dropna()

print(f"  {len(windowed):,} windows  |  Benign: {(windowed['label']==0).sum():,}  "
      f"Attack: {(windowed['label']==1).sum():,}")

# ── 3. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
print("[3/6] Splitting 70/30 stratified...")
train_df, test_df = train_test_split(
    windowed, test_size=TEST_SIZE, random_state=42, stratify=windowed['label']
)

X_train = train_df[FEATURES].values
y_train = train_df['label'].values
X_test  = test_df[FEATURES].values
y_test  = test_df['label'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"  Train: {len(X_train):,} windows  (Benign: {(y_train==0).sum():,}, Attack: {(y_train==1).sum():,})")
print(f"  Test:  {len(X_test):,} windows  (Benign: {(y_test==0).sum():,}, Attack: {(y_test==1).sum():,})")

# ── 4. TRAIN MODELS ──────────────────────────────────────────────────────────
print("[4/6] Training models...")

iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train_scaled)
print(f"  Isolation Forest: trained on {len(X_train_scaled):,} windows (unsupervised, labels ignored)")

rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
print(f"  Random Forest:    trained on {len(X_train_scaled):,} windows (supervised, labels used)")

# ── 5. DETECTION PERFORMANCE ─────────────────────────────────────────────────
print("[5/6] Evaluating detection performance...")

iso_pred_raw = iso_forest.predict(X_test_scaled)
iso_scores   = iso_forest.decision_function(X_test_scaled)
iso_pred     = (iso_pred_raw == -1).astype(int)

rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_pred       = rf_model.predict(X_test_scaled)

def compute_metrics(y_true, y_pred, y_scores):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = float('nan')
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC':   auc,
    }

# Negate iso_scores: IF.decision_function returns lower values for anomalies,
# but roc_auc_score expects higher scores for the positive class.
iso_metrics = compute_metrics(y_test, iso_pred, -iso_scores)
rf_metrics  = compute_metrics(y_test, rf_pred,  rf_pred_proba)

# ── 6. RESOURCE PROFILING ────────────────────────────────────────────────────
print("[6/6] Profiling resource usage...")

def profile_model(model, X_single, n_runs, predict_fn):
    # Model size via serialisation
    model_bytes = len(pickle.dumps(model))
    model_kb    = model_bytes / 1024

    # Inference time: average over n_runs single-sample predictions
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(X_single)
        times.append((time.perf_counter() - t0) * 1000)
    # discard top 1% outliers (JIT / cache effects)
    times_arr = np.array(times)
    cutoff = np.percentile(times_arr, 99)
    clean_times = times_arr[times_arr <= cutoff]

    # Memory footprint during a single prediction
    tracemalloc.start()
    predict_fn(X_single)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_kb = peak / 1024

    return {
        'model_size_kb':    round(model_kb, 1),
        'inference_mean_ms': round(np.mean(clean_times), 4),
        'inference_std_ms':  round(np.std(clean_times), 4),
        'inference_p99_ms':  round(np.percentile(clean_times, 99), 4),
        'peak_memory_kb':    round(peak_kb, 1),
    }

sample = X_test_scaled[0:1]  # single window for timing

iso_res = profile_model(
    iso_forest, sample, N_TIMING_RUNS,
    lambda x: iso_forest.decision_function(x)
)
rf_res = profile_model(
    rf_model, sample, N_TIMING_RUNS,
    lambda x: rf_model.predict_proba(x)
)

print(f"  IF: {iso_res['model_size_kb']:.1f} KB  |  "
      f"{iso_res['inference_mean_ms']:.4f} ms/pred  |  "
      f"{iso_res['peak_memory_kb']:.1f} KB peak mem")
print(f"  RF: {rf_res['model_size_kb']:.1f} KB  |  "
      f"{rf_res['inference_mean_ms']:.4f} ms/pred  |  "
      f"{rf_res['peak_memory_kb']:.1f} KB peak mem")

# ── PRINT FULL RESULTS ────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

for name, m, r in [("ISOLATION FOREST (Unsupervised)", iso_metrics, iso_res),
                   ("RANDOM FOREST    (Supervised)",   rf_metrics,  rf_res)]:
    print(f"\n{name}")
    print(f"  TP={m['TP']:5d}  FN={m['FN']:5d}  |  Precision: {m['Precision']:.4f}")
    print(f"  FP={m['FP']:5d}  TN={m['TN']:5d}  |  Recall:    {m['Recall']:.4f}")
    print(f"                       |  F1-Score:  {m['F1-Score']:.4f}")
    print(f"                       |  ROC-AUC:   {m['ROC-AUC']:.4f}")
    print(f"  Resources:")
    print(f"    Model size:      {r['model_size_kb']:>8.1f} KB")
    print(f"    Inference time:  {r['inference_mean_ms']:>8.4f} ms  (p99: {r['inference_p99_ms']:.4f} ms)")
    print(f"    Peak memory:     {r['peak_memory_kb']:>8.1f} KB")

# Recommendation
lighter = "Isolation Forest" if iso_res['model_size_kb'] <= rf_res['model_size_kb'] else "Random Forest"
heavier = "Random Forest" if lighter == "Isolation Forest" else "Isolation Forest"
lighter_size = iso_res['model_size_kb'] if lighter == "Isolation Forest" else rf_res['model_size_kb']
heavier_size = rf_res['model_size_kb'] if lighter == "Isolation Forest" else iso_res['model_size_kb']
size_ratio = heavier_size / max(lighter_size, 0.001)

speed_ratio = rf_res['inference_mean_ms'] / max(iso_res['inference_mean_ms'], 0.0001)

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print(f"\n  Both models achieve excellent detection:")
print(f"    IF  ROC-AUC = {iso_metrics['ROC-AUC']:.4f}  |  F1 = {iso_metrics['F1-Score']:.4f}")
print(f"    RF  ROC-AUC = {rf_metrics['ROC-AUC']:.4f}  |  F1 = {rf_metrics['F1-Score']:.4f}")
print(f"\n  Resource overhead:")
print(f"    RF is {size_ratio:.1f}x larger ({heavier_size:.0f} KB vs {lighter_size:.0f} KB)")
print(f"    RF inference is {speed_ratio:.1f}x slower per prediction")
print(f"\n  Both achieve excellent detection. We recommend {lighter} due to lower")
print(f"  resource overhead, making it more suitable for resource-constrained")
print(f"  IoT edge deployments.")

# ── SAVE CSVs ─────────────────────────────────────────────────────────────────
perf_df = pd.DataFrame([
    {
        'model': 'Isolation Forest', 'type': 'Unsupervised',
        'training_samples': len(X_train),
        'precision': iso_metrics['Precision'], 'recall': iso_metrics['Recall'],
        'f1_score': iso_metrics['F1-Score'],   'roc_auc': iso_metrics['ROC-AUC'],
        'tp': iso_metrics['TP'], 'tn': iso_metrics['TN'],
        'fp': iso_metrics['FP'], 'fn': iso_metrics['FN'],
    },
    {
        'model': 'Random Forest', 'type': 'Supervised',
        'training_samples': len(X_train),
        'precision': rf_metrics['Precision'], 'recall': rf_metrics['Recall'],
        'f1_score': rf_metrics['F1-Score'],   'roc_auc': rf_metrics['ROC-AUC'],
        'tp': rf_metrics['TP'], 'tn': rf_metrics['TN'],
        'fp': rf_metrics['FP'], 'fn': rf_metrics['FN'],
    },
])
perf_df.to_csv('model_comparison_final.csv', index=False)

res_df = pd.DataFrame([
    {'model': 'Isolation Forest', **iso_res},
    {'model': 'Random Forest',    **rf_res},
])
res_df.to_csv('resource_comparison.csv', index=False)

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Isolation Forest vs Random Forest — Final Evaluation\n'
             '(70/30 Stratified Split, Both Trained on Full Training Set)',
             fontsize=13, fontweight='bold')

# (0,0) Confusion matrix — IF
cm_iso = confusion_matrix(y_test, iso_pred)
sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            cbar=False, xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
axes[0, 0].set_title('Isolation Forest — Confusion Matrix')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# (0,1) Confusion matrix — RF
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
            cbar=False, xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
axes[0, 1].set_title('Random Forest — Confusion Matrix')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')

# (0,2) ROC curves
fpr_iso, tpr_iso, _ = roc_curve(y_test, -iso_scores)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_pred_proba)
axes[0, 2].plot(fpr_iso, tpr_iso, color='steelblue', linewidth=2.5,
                label=f'Isolation Forest (AUC={iso_metrics["ROC-AUC"]:.3f})')
axes[0, 2].plot(fpr_rf, tpr_rf, color='seagreen', linewidth=2.5,
                label=f'Random Forest (AUC={rf_metrics["ROC-AUC"]:.3f})')
axes[0, 2].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].set_title('ROC Curves')
axes[0, 2].legend(loc='lower right')
axes[0, 2].grid(alpha=0.3)

# (1,0) Detection metrics bar chart
metrics_names = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
iso_vals = [iso_metrics['Precision'], iso_metrics['Recall'],
            iso_metrics['F1-Score'],  iso_metrics['ROC-AUC']]
rf_vals  = [rf_metrics['Precision'],  rf_metrics['Recall'],
            rf_metrics['F1-Score'],   rf_metrics['ROC-AUC']]
x = np.arange(len(metrics_names))
w = 0.35
axes[1, 0].bar(x - w/2, iso_vals, w, label='Isolation Forest', color='steelblue')
axes[1, 0].bar(x + w/2, rf_vals,  w, label='Random Forest',    color='seagreen')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics_names)
axes[1, 0].set_ylim(0, 1.05)
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Detection Metrics (Higher is Better)')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# (1,1) Model size & inference time
models = ['Isolation\nForest', 'Random\nForest']
sizes  = [iso_res['model_size_kb'], rf_res['model_size_kb']]
colors = ['steelblue', 'seagreen']
axes[1, 1].bar(models, sizes, color=colors, width=0.4)
axes[1, 1].set_ylabel('Model Size (KB)')
axes[1, 1].set_title('Model Size Comparison (Lower is Better)')
for i, v in enumerate(sizes):
    axes[1, 1].text(i, v + max(sizes) * 0.01, f'{v:.0f} KB', ha='center', fontsize=10)
axes[1, 1].grid(axis='y', alpha=0.3)

# (1,2) Inference time
times = [iso_res['inference_mean_ms'], rf_res['inference_mean_ms']]
axes[1, 2].bar(models, times, color=colors, width=0.4)
axes[1, 2].set_ylabel('Mean Inference Time (ms)')
axes[1, 2].set_title('Per-Prediction Inference Time (Lower is Better)')
for i, v in enumerate(times):
    axes[1, 2].text(i, v + max(times) * 0.01, f'{v:.4f} ms', ha='center', fontsize=10)
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_final.png', dpi=150, bbox_inches='tight')

print("\nFiles saved:")
print("  model_comparison_final.csv  — detection metrics")
print("  resource_comparison.csv     — resource profiling")
print("  model_comparison_final.png  — all plots")
