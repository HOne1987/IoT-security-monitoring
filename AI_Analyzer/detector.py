import time
import requests
import pandas as pd
from sklearn.ensemble import IsolationForest
from prometheus_client import start_http_server, Gauge

# --- Configuration ---
PROMETHEUS_URL = 'http://prometheus:9090/api/v1/query'
UPDATE_INTERVAL = 5 # Check every 5 seconds

# --- Metrics to Expose ---
AI_ANOMALY_SCORE = Gauge('ai_anomaly_score', 'AI Anomaly Score (1 = Anomaly, 0 = Normal)')

# --- Initialize the Machine Learning Model ---
print("Initializing Isolation Forest AI Model...")
# We provide a synthetic baseline of 'normal' and 'anomalous' data so the model 
# can mathematically define the boundaries without needing weeks of historical training data.
baseline_data = pd.DataFrame({
    'temp': [35.2, 35.1, 35.3, 35.0, 35.2, 45.0, 35.1], # 45.0 is a physical anomaly
    'cyber_load': [1500, 2000, 1200, 1800, 1600, 15000000, 1500] # 15MB is a Mirai DDoS anomaly
})

# contamination=0.1 means we expect roughly 10% of the dataset to be anomalies
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(baseline_data)

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
    # 1. Fetch live synchronized data from your datasets
    temp = fetch_metric('iot_physical_temperature_c{device_ip="10.11.12.17"}')
    cyber_load = fetch_metric('sum(iot_cyber_packet_size_bytes)')

    # Skip analysis if Prometheus hasn't scraped the first data points yet
    if temp == 0.0 and cyber_load == 0.0:
        return

    # 2. Feed live data into the Machine Learning Model
    live_df = pd.DataFrame({'temp': [temp], 'cyber_load': [cyber_load]})
    prediction = model.predict(live_df)[0] # IsolationForest returns 1 for normal, -1 for anomaly

    # 3. Convert prediction to a Grafana-friendly score and push to Prometheus
    score = 1 if prediction == -1 else 0
    AI_ANOMALY_SCORE.set(score)

    # 4. Print logs for terminal debugging
    status = "🚨 ANOMALY DETECTED" if score == 1 else "✅ NORMAL"
    print(f"[AI Engine] Temp: {temp:.1f}°C | Cyber Load: {cyber_load:.0f} bytes | Result: {status}")

if __name__ == '__main__':
    print("Starting AI Anomaly Detector Microservice on port 8001...")
    start_http_server(8001)
    
    # Give Prometheus a few seconds to boot up before we start querying it
    time.sleep(10)
    
    while True:
        analyze()
        time.sleep(UPDATE_INTERVAL)
