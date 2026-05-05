# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSc Cybersecurity thesis project: a containerized IoT security monitoring framework that correlates **Digital Security** (network anomaly detection) with **Physical Safety** (environmental sensors). It uses a Cloud-Edge Collaborative model — an IoT edge node feeds telemetry into a Prometheus/Grafana observability stack, with an AI anomaly detector running Isolation Forest.

## Stack & Services

```
docker-compose up --build -d   # Build all images and start the full stack
docker-compose down            # Stop the stack
docker-compose logs -f         # Stream logs from all services
docker-compose logs -f iot-device    # Logs for a specific service
```

Four services defined in `docker-compose.yml`:
- **`iot-device`** (port 8000) — Red Hat UBI 9 hardened edge node; runs the telemetry agent
- **`prometheus`** (port 9090) — scrapes `iot-device:8000` and `ai-analyzer:8001` every 5s
- **`grafana`** (port 3000, login `admin`/`admin`) — dashboards; Prometheus auto-provisioned via `Hub_Configs/grafana/provisioning/`
- **`ai-analyzer`** (port 8001) — pulls metrics from Prometheus REST API, runs ML inference

## IoT Agent Variants

There are two interchangeable telemetry agents in `IoT_Device/`. The active one is set in `IoT_Device/Dockerfile`:

| File | Dataset (Physical) | Dataset (Cyber) | Notes |
|------|--------------------|-----------------|-------|
| `universal_agent_ToN-IoT.py` | `IoT_Thermostat.csv` | `Network_dataset_1.csv` | Uses pandas; temporal normalization; **currently active** |
| `universal_agent_Majibetal.py` | `SensorData500Rows.csv` | `temp_00001_20221019140448.pcap_35000_rows.csv` | Uses stdlib `csv`; column-index based |

To switch agents, update the `COPY` line and `CMD` in `IoT_Device/Dockerfile`.

Both agents implement a **trace-driven emulation loop**: they replay CSV data in real-time (1 physical second = 1 wall-clock second), synchronizing physical sensor rows with cyber network flow rows by timestamp. Physical data drives the outer loop; cyber packets are batched and flushed per epoch window.

## Key Prometheus Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `iot_physical_temperature_c` | Gauge | `device` | Physical sensor temperature (°C) |
| `iot_cyber_packets_total` | Counter | `protocol` | Cumulative packet count per protocol |
| `iot_cyber_packet_size_bytes` | Gauge | `protocol`, `source`, `destination` | Latest flow byte size |
| `ai_anomaly_score` | Gauge | — | `1` = threat detected, `0` = normal |

## AI Analyzer Logic (`AI_Analyzer/detector.py`)

Two-phase operation:
1. **Warm-up** (first 60s): collects `[temperature, packets_per_second]` vectors, then fits `StandardScaler` and `IsolationForest(contamination=0.01)` on real observed data.
2. **Active detection**: scales live metrics and calls `model.predict()` every 5s; sets `ai_anomaly_score` accordingly.

The analyzer fetches metrics via the Prometheus HTTP API (`http://prometheus:9090/api/v1/query`), not by scraping directly.

## Attack Injection

`data/inject_attack.py` appends a synchronized cyber-physical attack scenario to the Majibetal CSV datasets — 50 seconds of 60°C temperature spikes + 100 UDP flood packets/second, with relative timestamps locked to align with the end of the physical timeline. Run it from the `data/` directory before starting the stack if you want the attack scenario in the Majibetal datasets.

## Data Files

All CSVs mount into the `iot-device` container at `/app/data/` via `volumes: - ./data:/app/data`.

- `IoT_Thermostat.csv` / `Network_dataset_1.csv` — ToN-IoT dataset
- `SensorData500Rows.csv` / `temp_00001_20221019140448.pcap_35000_rows.csv` — Majib et al. dataset

## Grafana Dashboard

`grafana_dashboard.json` contains the pre-built dashboard definition. Import it via Grafana UI (Dashboards → Import) after the stack is running, or add it to `Hub_Configs/grafana/provisioning/dashboards/` with a provider config to auto-provision.
