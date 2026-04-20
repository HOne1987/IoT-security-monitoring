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
model = IsolationForest(contamination=0.03, random_state=42)

baseline_data = []
is_trained = False

def fetch_metric(query):
    try:
        response = requests.get(PROMETHEUS_URL, params={'query': query})
        results = response.json().get('data', {}).get('result', [])
        if results:
            return float(results[0]['value'][1])
    except Exception:
        pass
    return 0.0

def analyze():
    global is_trained, baseline_data

    # Using [1] instead of [3] to ensure we read Temperature, not Noise!
    temp = fetch_metric('iot_physical_temperature_c{device_ip="10.11.12.17"}')
    pps = fetch_metric('sum(irate(iot_cyber_packets_total[1m]))')

    if temp == 0.0 and pps == 0.0:
        return

    # --- PHASE 1: DYNAMIC WARM-UP & TRAINING ---
    if not is_trained:
        baseline_data.append([temp, pps])
        # flush=True forces Docker to print to the terminal immediately!
        print(f"[WARM-UP {len(baseline_data)}/{WARMUP_PERIOD_CHECKS}] Learning environment... Temp: {temp:.1f}, PPS: {pps:.1f}", flush=True)

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
    scaled_live = scaler.transform([[temp, pps]])

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
    print(f"[AI] Temp: {temp:.1f}°C | PPS: {pps:.1f} | ML Score: {raw_score:.3f} | Latency: {inference_time_ms:.2f}ms | {status}", flush=True)

if __name__ == '__main__':
    print("Starting Adaptive AI Anomaly Detector...", flush=True)
    start_http_server(8001)

    # Give Prometheus a few seconds to boot and start scraping
    time.sleep(10)

    while True:
        analyze()
        time.sleep(UPDATE_INTERVAL)
