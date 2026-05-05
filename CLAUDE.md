# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# IoT Security Monitoring — MSc Thesis Project

## Project Overview

MSc Cybersecurity thesis project: a containerized IoT security monitoring framework that correlates **Digital Security** (network anomaly detection) with **Physical Safety** (environmental sensors). It uses a Cloud-Edge Collaborative model — an IoT edge node feeds telemetry into a Prometheus/Grafana observability stack, with an AI anomaly detector running Isolation Forest.

## Architecture
- Edge Node: `IoT_Device/universal_agent_ToN-IoT.py` — streams CSV data as Prometheus metrics
- Hub: Prometheus (port 9090) + Grafana (port 3000)
- AI Layer: `AI_Analyzer/detector.py` — Isolation Forest anomaly detection

## Current Task
Simplifying to Cyber-only design using ToN-IoT Network dataset only.
Physical/temperature data is EXCLUDED from this evaluation phase.

## Key Files
- Agent: `IoT_Device/universal_agent_ToN-IoT.py`
- Detector: `AI_Analyzer/detector.py`
- Compose: `docker-compose.yml`

## Dataset
- Cyber ONLY: `data/Network_dataset_1.csv`
- Columns used: ts, proto, src_ip, dst_ip, src_pkts, dst_bytes, label
- Physical dataset (IoT_Thermostat.csv) is NOT used in this phase

## Conventions
- Python, Prometheus client library, scikit-learn
- Prometheus metrics prefix: `iot_cyber_`
- Docker network: `iot-secure-net`
