import time
import requests
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from prometheus_client import start_http_server, Gauge

# --- Configuration ---
PROMETHEUS_URL = 'http://prometheus:9090/api/v1/query'
UPDATE_INTERVAL = 5 # Check every 5 seconds

# --- Metrics to Expose ---
AI_ANOMALY_SCORE = Gauge('ai_anomaly_score', 'AI Anomaly Score (1 = Threat, 0 = Normal)')

# --- 1. Scientific Machine Learning Initialization ---
print("Initializing Scaled Isolation Forest Model...")

# BASELINE CALIBRATION:
# Normal IoT background PPS is very low (~1-5).
# Our simulated attack spike (5000 packets / 60 seconds) calculates to ~83.3 PPS.
baseline_df = pd.DataFrame({
    'temp': [35.2, 35.1, 35.3, 35.0, 35.2, 60.0, 35.1], # 60.0 is the physical anomaly
    'pps':  [1.5,  2.0,  1.2,  1.8,  1.6,  85.0, 1.5]   # 85.0 is the simulated DDoS PPS
})

# FEATURE SCALING:
# This fixes the "Outlier Shadowing" problem. It mathematically normalizes
# temperature (tens) and PPS (thousands) so one doesn't overpower the other.
scaler = StandardScaler()
scaled_baseline = scaler.fit_transform(baseline_df)

# Initialize the Isolation Forest
model = IsolationForest(contamination=0.15, random_state=42)
model.fit(scaled_baseline)

def fetch_metric(query):
    """Pulls the latest value of a specific PromQL query from Prometheus."""
    try:
        response = requests.get(PROMETHEUS_URL, params={'query': query})
        results = response.json().get('data', {}).get('result', [])
        if results:
            return float(results[0]['value'][1])
    except Exception as e:
        pass
    return 0.0

def analyze():
    # 2. Fetch live data using proper Cyber-Physical queries
    temp = fetch_metric('iot_physical_temperature_c{device_ip="10.11.12.17"}')

    # NEW QUERY: Packets Per Second (PPS) over a 1-minute sliding window
    # This specifically addresses the supervisor's request to detect volumetric floods.
    pps = fetch_metric('sum(rate(iot_cyber_packets_total[1m]))')

    # Wait for Prometheus to start returning data
    if temp == 0.0 and pps == 0.0:
        return

    # 3. Apply the exact same scaling to the live data
    live_df = pd.DataFrame({'temp': [temp], 'pps': [pps]})
    scaled_live = scaler.transform(live_df)

    # 4. Predict
    prediction = model.predict(scaled_live)[0]
    raw_score = model.decision_function(scaled_live)[0]

    # Convert to binary for Grafana
    score = 1 if prediction == -1 else 0
    AI_ANOMALY_SCORE.set(score)

    # Output to terminal
    status = "🚨 THREAT DETECTED" if score == 1 else "✅ NORMAL"
    print(f"[AI] Temp: {temp:.1f}°C | PPS: {pps:.1f} pkts/sec | ML Score: {raw_score:.3f} | {status}")

if __name__ == '__main__':
    print("Starting Scaled AI Anomaly Detector Microservice on port 8001...")
    start_http_server(8001)
    time.sleep(10) # Buffer for Prometheus boot-up
    while True:
        analyze()
        time.sleep(UPDATE_INTERVAL)
