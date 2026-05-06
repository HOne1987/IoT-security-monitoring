import time
import requests
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from prometheus_client import start_http_server, Gauge

# --- Configuration ---
PROMETHEUS_URL = 'http://prometheus:9090/api/v1/query'
UPDATE_INTERVAL = 5
WARMUP_PERIOD_CHECKS = 12  # 12 checks * 5 seconds = 60 seconds of learning

# --- Metrics ---
AI_ANOMALY_SCORE = Gauge('ai_anomaly_score', 'AI Anomaly Score (1 = Threat, 0 = Normal)')
INFERENCE_LATENCY_MS = Gauge('ai_inference_latency_ms', 'Time to make prediction (milliseconds)')

scaler = StandardScaler()
# contamination=0.01 means we only expect 1% of data to be severe anomalies
model = IsolationForest(contamination=0.01, random_state=42)

baseline_data = []
is_trained = False

# --- FLOW-LEVEL FEATURES ---
FEATURES = ['flow_count', 'avg_duration', 'avg_bytes']


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
    """Fetch all three flow-level features from Prometheus."""
    flow_count = fetch_metric('iot_cyber_flow_count')
    avg_duration = fetch_metric('iot_cyber_avg_flow_duration_sec')
    avg_bytes = fetch_metric('iot_cyber_avg_flow_bytes')
    return [flow_count, avg_duration, avg_bytes]


def analyze():
    global is_trained, baseline_data

    # Fetch flow-level metrics
    metrics = fetch_flow_metrics()
    flow_count, avg_duration, avg_bytes = metrics

    # Skip if no data yet
    if all(m == 0.0 for m in metrics):
        return

    # --- PHASE 1: DYNAMIC WARM-UP & TRAINING ---
    if not is_trained:
        baseline_data.append(metrics)
        print(
            f"[WARM-UP {len(baseline_data)}/{WARMUP_PERIOD_CHECKS}] "
            f"Flow Count: {flow_count:.1f}, Avg Duration: {avg_duration:.2f}s, "
            f"Avg Bytes: {avg_bytes:.0f}",
            flush=True
        )

        if len(baseline_data) >= WARMUP_PERIOD_CHECKS:
            print(
                "\n[AI] Warm-up complete! Fitting Scaler and Isolation Forest to real environment data...",
                flush=True
            )
            # Train the AI on the actual data we just collected
            np_baseline = np.array(baseline_data)
            scaler.fit(np_baseline)
            scaled_baseline = scaler.transform(np_baseline)
            model.fit(scaled_baseline)
            is_trained = True
            print("[AI] Training Complete. Switching to active Threat Detection Mode.\n", flush=True)
        return

    # --- PHASE 2: ACTIVE THREAT DETECTION ---
    # Reshape for scaler (must be 2D)
    metrics_array = np.array(metrics).reshape(1, -1)
    scaled_live = scaler.transform(metrics_array)

    # Start the stopwatch
    start_time = time.perf_counter()

    prediction = model.predict(scaled_live)[0]
    raw_score = model.decision_function(scaled_live)[0]

    # Stop the stopwatch
    end_time = time.perf_counter()

    # Calculate inference time in milliseconds
    inference_time_ms = (end_time - start_time) * 1000

    # Convert prediction to binary: -1 (anomaly) -> 1, 1 (normal) -> 0
    score = 1 if prediction == -1 else 0
    AI_ANOMALY_SCORE.set(score)
    INFERENCE_LATENCY_MS.set(inference_time_ms)

    status = "🚨 THREAT DETECTED" if score == 1 else "✅ NORMAL"
    print(
        f"[AI] Flow Count: {flow_count:.1f} | Avg Duration: {avg_duration:.2f}s | "
        f"Avg Bytes: {avg_bytes:.0f} | ML Score: {raw_score:.3f} | "
        f"Latency: {inference_time_ms:.2f}ms | {status}",
        flush=True
    )


if __name__ == '__main__':
    print("Starting Adaptive AI Anomaly Detector (Flow-Level)...", flush=True)
    start_http_server(8001)

    # Give Prometheus a few seconds to boot and start scraping
    time.sleep(10)

    while True:
        analyze()
        time.sleep(UPDATE_INTERVAL)
