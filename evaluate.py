import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIG ──────────────────────────────────────────────────────────────────
CYBER_CSV   = 'data/Network_dataset_1.csv'
WARMUP_SECS = 60        # must match your live system
CONTAMINATION = 0.01    # must match your live detector.py

# ── LOAD & CLEAN ─────────────────────────────────────────────────────────────
print("Loading ToN-IoT dataset...")
df = pd.read_csv(CYBER_CSV, low_memory=False)

# Mirror the exact cleaning logic from universal_agent_ToN-IoT.py
metric_cols = ['src_pkts', 'dst_pkts', 'src_bytes', 'dst_bytes']
for col in metric_cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace('-', '0'), errors='coerce'
    ).fillna(0)

# Temporal normalization to T=0 (mirrors your agent)
df['ts'] = df['ts'] - df['ts'].min()
df = df.sort_values('ts').reset_index(drop=True)

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────
# Compute PPS per 1-second window (mirrors what Prometheus rate() sees)
df['ts_floor'] = df['ts'].astype(int)

windowed = df.groupby('ts_floor').agg(
    pps        = ('src_pkts', 'sum'),
    byte_rate  = ('src_bytes', 'sum'),
    label      = ('label', 'max')   # 1 if ANY attack in that second
).reset_index()

# ── WARM-UP / TRAIN SPLIT ────────────────────────────────────────────────────
warmup_mask  = windowed['ts_floor'] < WARMUP_SECS
train_df     = windowed[warmup_mask]
eval_df      = windowed[~warmup_mask].copy()

print(f"Warm-up windows : {len(train_df)}")
print(f"Evaluation windows: {len(eval_df)}")
print(f"Attack windows in eval: {eval_df['label'].sum()} "
      f"({eval_df['label'].mean()*100:.1f}%)")

# ── TRAIN ISOLATION FOREST ───────────────────────────────────────────────────
FEATURES = ['pps', 'byte_rate']

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[FEATURES])

model = IsolationForest(contamination=CONTAMINATION, random_state=42)
model.fit(X_train)
print("Isolation Forest trained on warm-up data.")

# ── PREDICT ON EVALUATION SET ─────────────────────────────────────────────────
X_eval = scaler.transform(eval_df[FEATURES])
raw_predictions = model.predict(X_eval)   # sklearn: -1 = anomaly, 1 = normal

# Convert to binary: 1 = attack detected, 0 = normal
eval_df['predicted'] = (raw_predictions == -1).astype(int)
eval_df['anomaly_score'] = model.decision_function(X_eval)

# Ground truth
y_true = eval_df['label'].values
y_pred = eval_df['predicted'].values

# ── CONFUSION MATRIX ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n══════════════════════════════════════════")
print("         EVALUATION RESULTS")
print("══════════════════════════════════════════")
print(f"  True Positives  (TP): {tp}")
print(f"  True Negatives  (TN): {tn}")
print(f"  False Positives (FP): {fp}  ← benign traffic flagged as attack")
print(f"  False Negatives (FN): {fn}  ← attacks missed")
print(f"\n  Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"  Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"  F1-Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}")
print("══════════════════════════════════════════\n")

# ── PLOT CONFUSION MATRIX ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
sns.heatmap(cm,
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Attack'],
            yticklabels=['Actual Normal', 'Actual Attack'],
            ax=axes[0])
axes[0].set_title('Confusion Matrix — Isolation Forest\n(ToN-IoT Evaluation Set)')
axes[0].set_ylabel('Ground Truth')
axes[0].set_xlabel('Model Prediction')

# Anomaly score over time
axes[1].plot(eval_df['ts_floor'], eval_df['anomaly_score'],
             label='Anomaly Score', color='steelblue', linewidth=0.8)
attack_mask = eval_df['label'] == 1
axes[1].fill_between(eval_df['ts_floor'], eval_df['anomaly_score'].min(),
                     eval_df['anomaly_score'].max(),
                     where=attack_mask, alpha=0.3,
                     color='red', label='Ground Truth Attack Window')
axes[1].axhline(0, color='red', linestyle='--', linewidth=1,
                label='Decision Boundary (score=0)')
axes[1].set_title('Anomaly Score vs. Ground Truth Attack Windows')
axes[1].set_xlabel('Time (seconds from T=0)')
axes[1].set_ylabel('Isolation Forest Decision Score')
axes[1].legend()

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
print("Plot saved to evaluation_results.png")
