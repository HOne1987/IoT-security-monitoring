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

scaler = StandardScaler()
# contamination=0.01 means we only expect 1% of data to be severe anomalies (prevents over-sensitivity)
model = IsolationForest(contamination=0.01, random_state=42)

baseline_data = []
is_trained = False

# detector.py (Flow-Level Features)

FEATURES = ['flow_count', 'avg_duration', 'avg_bytes']

def fetch_metrics():
    flow_count = fetch_metric('iot_cyber_flow_count')
    avg_duration = fetch_metric('iot_cyber_avg_flow_duration_sec')
    avg_bytes = fetch_metric('iot_cyber_avg_flow_bytes')
    return [flow_count, avg_duration, avg_bytes]

# Rest of the warm-up and inference stays the same

def analyze():
    global is_trained, baseline_data

    pps = fetch_metric('iot_cyber_pps')
    byte_rate = fetch_metric('iot_cyber_byte_rate')

    if pps == 0.0 and byte_rate == 0.0:
        return

    # --- PHASE 1: DYNAMIC WARM-UP & TRAINING ---
    if not is_trained:
        baseline_data.append([pps, byte_rate])
        print(f"[WARM-UP {len(baseline_data)}/{WARMUP_PERIOD_CHECKS}] Learning environment... PPS: {pps:.1f}, Byte Rate: {byte_rate:.1f}", flush=True)

        if len(baseline_data) >= WARMUP_PERIOD_CHECKS:
            print("\n[AI] Warm-up complete! Fitting Scaler and Isolation Forest to real environment data...", flush=True)
            # Train the AI on the actual data we just collected
            np_baseline = np.array(baseline_data)
            scaler.fit(np_baseline)
            scaled_baseline = scaler.transform(np_baseline)
            model.fit(scaled_baseline)
            is_trained = True
            print("[AI] Training Complete. Switching to active Threat Detection Mode.\n", flush=True)
        return


    # --- PHASE 2: ACTIVE THREAT DETECTION ---
    scaled_live = scaler.transform([[pps, byte_rate]])

    # Start the stopwatch
    start_time = time.perf_counter()

    prediction = model.predict(scaled_live)[0]
    raw_score = model.decision_function(scaled_live)[0]

    # Stop the stopwatch
    end_time = time.perf_counter()

    # Calculate inference time in milliseconds
    inference_time_ms = (end_time - start_time) * 1000

    score = 1 if prediction == -1 else 0
    AI_ANOMALY_SCORE.set(score)

    status = "🚨 THREAT DETECTED" if score == 1 else "✅ NORMAL"
    print(f"[AI] PPS: {pps:.1f} | Byte Rate: {byte_rate:.1f} | ML Score: {raw_score:.3f} | Latency: {inference_time_ms:.2f}ms | {status}", flush=True)

if __name__ == '__main__':
    print("Starting Adaptive AI Anomaly Detector...", flush=True)
    start_http_server(8001)

    # Give Prometheus a few seconds to boot and start scraping
    time.sleep(10)

    while True:
        analyze()
        time.sleep(UPDATE_INTERVAL)
