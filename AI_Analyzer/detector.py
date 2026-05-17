import time
import requests
import numpy as np
import joblib
import os
from prometheus_client import start_http_server, Gauge

# ═══════════════════════════════════════════════════════════════════════════════
# DETECTOR: Random Forest-Based Anomaly Detection
# Uses pre-trained RF model on ToN-IoT labeled data
# ═══════════════════════════════════════════════════════════════════════════════

# --- Configuration ---
PROMETHEUS_URL = 'http://prometheus:9090/api/v1/query'
UPDATE_INTERVAL = 5
MODEL_DIR = 'models'

# --- Metrics ---
AI_ANOMALY_SCORE = Gauge('ai_anomaly_score', 'AI Anomaly Score (1 = Threat, 0 = Normal)')
AI_ATTACK_PROBABILITY = Gauge('ai_attack_probability', 'Probability of Attack (0-1)')
INFERENCE_LATENCY_MS = Gauge('ai_inference_latency_ms', 'Time to make prediction (milliseconds)')

# --- Load Pre-trained Model ---
print("[INIT] Loading Random Forest model and scaler...", flush=True)

model_path = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
features_path = os.path.join(MODEL_DIR, 'features.txt')

if not os.path.exists(model_path):
    print(f"[ERROR] Model not found at {model_path}", flush=True)
    print(f"[ERROR] Run: python train_random_forest.py first", flush=True)
    exit(1)

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load feature names
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines()]

    print(f"[INIT] ✓ Model loaded successfully", flush=True)
    print(f"[INIT] ✓ Scaler loaded successfully", flush=True)
    print(f"[INIT] ✓ Features: {features}", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}", flush=True)
    exit(1)


def fetch_metric(metric_name):
    """Query Prometheus for a single metric value."""
    try:
        response = requests.get(
            PROMETHEUS_URL,
            params={'query': metric_name},
            timeout=2
        )
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'success' and data['data']['result']:
            return float(data['data']['result'][0]['value'][1])
        return 0.0
    except Exception as e:
        print(f"[ERROR] Failed to fetch {metric_name}: {e}", flush=True)
        return 0.0


def fetch_flow_metrics():
    """Fetch flow-level metrics from Prometheus."""
    flow_count = fetch_metric('iot_cyber_flow_count')
    avg_duration = fetch_metric('iot_cyber_avg_flow_duration_sec')
    avg_bytes = fetch_metric('iot_cyber_avg_flow_bytes')
    return [flow_count, avg_duration, avg_bytes]


def analyze():
    """Run anomaly detection on current metrics."""
    # Fetch metrics
    metrics = fetch_flow_metrics()
    flow_count, avg_duration, avg_bytes = metrics

    # Skip if no data
    if all(m == 0.0 for m in metrics):
        return

    # Reshape for scaler (must be 2D)
    metrics_array = np.array(metrics).reshape(1, -1)

    # Scale using trained scaler
    try:
        metrics_scaled = scaler.transform(metrics_array)
    except Exception as e:
        print(f"[ERROR] Scaler transform failed: {e}", flush=True)
        return

    # --- PREDICTION ---
    start_time = time.perf_counter()

    try:
        # Get prediction (0 = normal, 1 = attack)
        prediction = model.predict(metrics_scaled)[0]

        # Get prediction probability (confidence)
        prediction_proba = model.predict_proba(metrics_scaled)[0]
        attack_probability = prediction_proba[1]  # Probability of class 1 (attack)
    except Exception as e:
        print(f"[ERROR] Model prediction failed: {e}", flush=True)
        return

    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000

    # --- UPDATE METRICS ---
    AI_ANOMALY_SCORE.set(prediction)
    AI_ATTACK_PROBABILITY.set(attack_probability)
    INFERENCE_LATENCY_MS.set(inference_time_ms)

    # --- LOG OUTPUT ---
    status = "🚨 THREAT DETECTED" if prediction == 1 else "✅ NORMAL"
    print(
        f"[DETECTION] Flow Count: {flow_count:.1f} | "
        f"Avg Duration: {avg_duration:.2f}s | "
        f"Avg Bytes: {avg_bytes:.0f} | "
        f"Attack Probability: {attack_probability:.3f} | "
        f"Latency: {inference_time_ms:.2f}ms | "
        f"{status}",
        flush=True
    )


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("Starting Anomaly Detector (Random Forest)", flush=True)
    print("=" * 80, flush=True)

    start_http_server(8001)
    print("[INIT] Prometheus exporter listening on port 8001", flush=True)

    # Give Prometheus time to boot
    time.sleep(10)

    print("[INIT] Starting detection loop...", flush=True)
    print("=" * 80, flush=True)

    while True:
        analyze()
        time.sleep(UPDATE_INTERVAL)
