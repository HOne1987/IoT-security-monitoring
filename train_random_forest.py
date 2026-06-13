import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# train_test_network.csv has no timestamp column, so temporal windowing is not
# possible with it.  Network_dataset_1.csv is what the agent actually reads and
# has 'ts', so windowed features derived from it exactly match what Prometheus
# exports (iot_cyber_flow_count, iot_cyber_avg_flow_duration_sec,
# iot_cyber_avg_flow_bytes).

CYBER_CSV = 'data/Network_dataset_1.csv'
WINDOW_SIZE = 10        # seconds — matches agent loop
TEST_SIZE = 0.3
MODEL_OUTPUT_DIR = 'AI_Analyzer/models'
FEATURES = ['flow_count', 'avg_duration', 'avg_bytes']

print("=" * 80)
print("TRAINING: Random Forest on 10-second Windowed Features")
print(f"Features: {FEATURES}  (matches agent Prometheus exports)")
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

# Agent filters duration > 0 before computing per-window stats
df_valid = df[df['duration'] > 0].copy()
print(f"  {len(df):,} flows loaded, {len(df_valid):,} valid (duration > 0)")

# ── 2. WINDOW AGGREGATION (mirrors agent logic exactly) ───────────────────────
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

print(f"  {len(windowed):,} windows created")
print(f"  Benign: {(windowed['label']==0).sum():,}  |  Attack: {(windowed['label']==1).sum():,}")

# ── 3. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
print("\n[3/5] Splitting 70/30 stratified by label...")
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

# ── 4. TRAIN ──────────────────────────────────────────────────────────────────
print("\n[4/5] Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
rf_model.fit(X_train_scaled, y_train)
print("  Training complete")

# ── 5. EVALUATE ───────────────────────────────────────────────────────────────
print("\n[5/5] Evaluating on test set...")
y_pred       = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 80)
print("EVALUATION RESULTS  (windowed features, unseen test set)")
print("=" * 80)
print(f"\n  Confusion Matrix:")
print(f"    TP={tp:5,} | FN={fn:5,}")
print(f"    FP={fp:5,} | TN={tn:5,}")
print(f"\n  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

print("\n  Feature importances:")
for feat, imp in sorted(zip(FEATURES, rf_model.feature_importances_),
                         key=lambda x: -x[1]):
    print(f"    {feat:20s}: {imp:.4f}")

# ── SAVE ─────────────────────────────────────────────────────────────────────
print(f"\nSaving artifacts to {MODEL_OUTPUT_DIR}/...")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

joblib.dump(rf_model, os.path.join(MODEL_OUTPUT_DIR, 'random_forest_model.pkl'))
joblib.dump(scaler,   os.path.join(MODEL_OUTPUT_DIR, 'scaler.pkl'))

with open(os.path.join(MODEL_OUTPUT_DIR, 'features.txt'), 'w') as f:
    f.write('\n'.join(FEATURES) + '\n')

with open(os.path.join(MODEL_OUTPUT_DIR, 'training_metadata.txt'), 'w') as f:
    f.write("Random Forest Training Metadata\n")
    f.write("=" * 50 + "\n")
    f.write(f"Training date: {pd.Timestamp.now()}\n")
    f.write(f"Dataset: {CYBER_CSV}\n")
    f.write(f"Window size: {WINDOW_SIZE}s\n")
    f.write(f"Features: {', '.join(FEATURES)}\n")
    f.write(f"Train windows: {len(X_train):,}\n")
    f.write(f"Test windows:  {len(X_test):,}\n")
    f.write(f"\nTest Set Performance:\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall:    {recall:.4f}\n")
    f.write(f"  F1-Score:  {f1:.4f}\n")
    f.write(f"  ROC-AUC:   {roc_auc:.4f}\n")

print("  random_forest_model.pkl")
print("  scaler.pkl")
print("  features.txt")
print("  training_metadata.txt")

# ── CONFUSION MATRIX PLOT ─────────────────────────────────────────────────────
n_total = len(y_test)
cm_data = [[tn, fp], [fn, tp]]
cm_labels = [['TN', 'FP'], ['FN', 'TP']]

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(
    cm_data, annot=False, fmt='d', cmap='Blues', ax=ax, cbar=False,
    xticklabels=['Normal', 'Attack'],
    yticklabels=['Normal', 'Attack'],
    linewidths=0.5, linecolor='white',
)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
ax.set_ylabel('True Label', fontsize=11)
ax.set_title(
    f'Random Forest Detector — Confusion Matrix\n(ToN-IoT, $n$ = {n_total:,})',
    fontsize=12, fontweight='bold', pad=14,
)

norm = tn + fp + fn + tp
for i, (row_vals, row_keys) in enumerate(zip(cm_data, cm_labels)):
    for j, (val, key) in enumerate(zip(row_vals, row_keys)):
        pct  = val / norm * 100
        cell_max = max(tn, fp, fn, tp)
        color = 'white' if val > cell_max * 0.5 else 'black'
        ax.text(j + 0.5, i + 0.38, f'{val:,}',
                ha='center', va='center', fontsize=14, fontweight='bold', color=color)
        ax.text(j + 0.5, i + 0.65, f'({pct:.1f} %)',
                ha='center', va='center', fontsize=9, color=color)

plt.tight_layout()
plt.savefig('confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print("  confusion_matrix_rf.png")

print("\n" + "=" * 80)
print("TRAINING COMPLETE — model ready for deployment in detector.py")
print("=" * 80)
