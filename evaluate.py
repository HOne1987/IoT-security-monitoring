import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

CYBER_CSV  = 'data/Network_dataset_1.csv'
WINDOW_SIZE = 10   # seconds — matches live agent
TEST_SIZE   = 0.3
FEATURES    = ['flow_count', 'avg_duration', 'avg_bytes']

print("=" * 80)
print("CHAPTER 4 FINAL EVALUATION: Three-Way Baseline Comparison")
print("Dataset: Network_dataset_1.csv  |  10-second windows  |  70/30 split")
print("=" * 80)

# ── 1. LOAD & CLEAN ───────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")
df = pd.read_csv(CYBER_CSV, low_memory=False)

for col in ['src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace('-', '0'),
                             errors='coerce').fillna(0)
df['ts'] = df['ts'] - df['ts'].min()
df = df.sort_values('ts').reset_index(drop=True)
df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)

df_valid = df[df['duration'] > 0].copy()
print(f"  {len(df):,} flows loaded, {len(df_valid):,} valid (duration > 0)")

# ── 2. WINDOW AGGREGATION ─────────────────────────────────────────────────────
print(f"[2/5] Aggregating into {WINDOW_SIZE}-second windows...")
df_valid['window'] = (df_valid['ts'] / WINDOW_SIZE).astype(int)

windowed = df_valid.groupby('window').agg(
    flow_count=('duration', 'count'),
    avg_duration=('duration', 'mean'),
    src_bytes=('src_bytes', 'sum'),
    dst_bytes=('dst_bytes', 'sum'),
    label=('label', 'max'),
).reset_index()
windowed['avg_bytes'] = (windowed['src_bytes'] + windowed['dst_bytes']) / windowed['flow_count']
windowed = windowed.dropna()

print(f"  {len(windowed):,} windows  |  "
      f"Benign: {(windowed['label']==0).sum():,}  Attack: {(windowed['label']==1).sum():,}")

# ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
print("[3/5] Splitting 70/30 stratified by label...")
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

print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

# ── 4. FIT ALL THREE MODELS ───────────────────────────────────────────────────
print("[4/5] Fitting models...")

# --- Baseline 1: Random Threshold -------------------------------------------
# Flags a window as attack if flow_count exceeds the 95th percentile of
# training-set benign windows — the simplest possible anomaly heuristic.
benign_train_fc = train_df.loc[y_train == 0, 'flow_count']
threshold_val   = np.percentile(benign_train_fc, 95)
thresh_pred  = (test_df['flow_count'].values > threshold_val).astype(int)
thresh_score = test_df['flow_count'].values  # raw count as anomaly score
print(f"  Random Threshold: flow_count > {threshold_val:.1f} (95th pctile of benign train)")

# --- Baseline 2: Isolation Forest --------------------------------------------
# Trained on the full training set (no warm-up); contamination=0.05 from
# threshold sensitivity analysis in resource_profile.py.
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train_scaled)
iso_raw   = iso.predict(X_test_scaled)
iso_pred  = (iso_raw == -1).astype(int)
# Negate: decision_function returns lower scores for anomalies, but
# roc_auc_score expects higher scores for the positive class.
iso_score = -iso.decision_function(X_test_scaled)
print(f"  Isolation Forest: trained on {len(X_train):,} windows (unsupervised)")

# --- Model 3: Random Forest --------------------------------------------------
# Supervised; trained on labeled windows that match the agent's Prometheus
# exports (flow_count, avg_duration, avg_bytes).
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1,
)
rf.fit(X_train_scaled, y_train)
rf_pred  = rf.predict(X_test_scaled)
rf_score = rf.predict_proba(X_test_scaled)[:, 1]
print(f"  Random Forest: trained on {len(X_train):,} windows (supervised)")

# ── 5. EVALUATE ───────────────────────────────────────────────────────────────
print("[5/5] Computing metrics...")

def metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_true, y_score),
    }

m_thresh = metrics(y_test, thresh_pred, thresh_score)
m_iso    = metrics(y_test, iso_pred,    iso_score)
m_rf     = metrics(y_test, rf_pred,     rf_score)

# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CHAPTER 4 — FINAL COMPARISON TABLE")
print("=" * 80)

header = f"{'Model':<28} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}"
print(f"\n{header}")
print("-" * 70)
rows = [
    ("Random Threshold (heuristic)", m_thresh),
    ("Isolation Forest (unsupervised)", m_iso),
    ("Random Forest (supervised)",     m_rf),
]
for name, m in rows:
    print(f"  {name:<26} {m['Precision']:>10.4f} {m['Recall']:>10.4f} "
          f"{m['F1-Score']:>10.4f} {m['ROC-AUC']:>10.4f}")

print("\n  Confusion matrices:")
print(f"  {'Model':<28}  TP     FP     FN     TN")
print(f"  {'-'*62}")
for name, m in rows:
    print(f"  {name:<28}  {m['TP']:5,}  {m['FP']:5,}  {m['FN']:5,}  {m['TN']:5,}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
best_f1  = max(rows, key=lambda x: x[1]['F1-Score'])
best_auc = max(rows, key=lambda x: x[1]['ROC-AUC'])
print(f"\n  Best F1-Score:  {best_f1[0]}  ({best_f1[1]['F1-Score']:.4f})")
print(f"  Best ROC-AUC:   {best_auc[0]}  ({best_auc[1]['ROC-AUC']:.4f})")
print(f"\n  RF improvement over Isolation Forest:")
print(f"    F1:      {m_iso['F1-Score']:.4f} → {m_rf['F1-Score']:.4f}  "
      f"(+{m_rf['F1-Score']-m_iso['F1-Score']:.4f})")
print(f"    Precision: {m_iso['Precision']:.4f} → {m_rf['Precision']:.4f}  "
      f"(+{m_rf['Precision']-m_iso['Precision']:.4f})")
print(f"\n  RF improvement over Random Threshold:")
print(f"    F1:      {m_thresh['F1-Score']:.4f} → {m_rf['F1-Score']:.4f}  "
      f"(+{m_rf['F1-Score']-m_thresh['F1-Score']:.4f})")

# ── SAVE CSV ──────────────────────────────────────────────────────────────────
results_df = pd.DataFrame([
    {'model': name, **{k: v for k, v in m.items() if k not in ('TP','TN','FP','FN')}}
    for name, m in rows
])
results_df.to_csv('evaluation_final.csv', index=False)
print("\n  Saved: evaluation_final.csv")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Chapter 4 — Final Baseline Comparison\n'
             'Network_dataset_1.csv  |  10-second windows  |  70/30 stratified split',
             fontsize=13, fontweight='bold')

cms    = [confusion_matrix(y_test, p) for p in [thresh_pred, iso_pred, rf_pred]]
titles = ['Random Threshold\n(heuristic baseline)',
          'Isolation Forest\n(unsupervised baseline)',
          'Random Forest\n(supervised, production)']
cmaps  = ['Oranges', 'Blues', 'Greens']

for i, (cm, title, cmap) in enumerate(zip(cms, titles, cmaps)):
    ax = axes[0, i]
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax, cbar=False,
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

# ROC curves
ax = axes[1, 0]
fpr_t, tpr_t, _ = roc_curve(y_test, thresh_score)
fpr_i, tpr_i, _ = roc_curve(y_test, iso_score)
fpr_r, tpr_r, _ = roc_curve(y_test, rf_score)
ax.plot(fpr_t, tpr_t, color='darkorange', linewidth=2,
        label=f'Random Threshold  (AUC={m_thresh["ROC-AUC"]:.3f})')
ax.plot(fpr_i, tpr_i, color='steelblue', linewidth=2,
        label=f'Isolation Forest  (AUC={m_iso["ROC-AUC"]:.3f})')
ax.plot(fpr_r, tpr_r, color='seagreen', linewidth=2,
        label=f'Random Forest     (AUC={m_rf["ROC-AUC"]:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Three Models', fontsize=11, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)

# Metrics bar chart
ax = axes[1, 1]
metric_names = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metric_names))
w = 0.25
vals = [[m[k] for k in metric_names] for _, m in rows]
colors_bar = ['darkorange', 'steelblue', 'seagreen']
labels_bar = ['Random Threshold', 'Isolation Forest', 'Random Forest']
for j, (v, c, l) in enumerate(zip(vals, colors_bar, labels_bar)):
    ax.bar(x + (j - 1) * w, v, w, label=l, color=c)
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Metrics Comparison (Higher is Better)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Summary text panel
ax = axes[1, 2]
ax.axis('off')
summary_lines = [
    "SUMMARY",
    "",
    f"Test set: {len(y_test):,} windows",
    f"  Benign:  {(y_test==0).sum():,}",
    f"  Attack:  {(y_test==1).sum():,}",
    "",
    "Model           Prec   Recall  F1",
]
for name, m in rows:
    short = name.split()[0] + " " + name.split()[1]
    summary_lines.append(
        f"  {short[:18]:<18} {m['Precision']:.3f}  {m['Recall']:.3f}  {m['F1-Score']:.3f}"
    )
summary_lines += [
    "",
    "ROC-AUC",
]
for name, m in rows:
    short = name.split()[0] + " " + name.split()[1]
    summary_lines.append(f"  {short[:18]:<18} {m['ROC-AUC']:.4f}")

ax.text(0.05, 0.95, '\n'.join(summary_lines),
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('evaluation_final.png', dpi=150, bbox_inches='tight')
print("  Saved: evaluation_final.png")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
