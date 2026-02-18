# 🛡️ Lightweight Cloud-Based Security Monitoring & Safety Assurance

**MSc Cybersecurity Project | Edge-Hub Architecture | Red Hat UBI 9**

This repository contains the implementation artifacts for an MSc Cybersecurity thesis focused on **"A Lightweight Cloud-Based Security Monitoring and Safety Assurance Framework for Residential IoT."**

The project provides a deployable, containerized solution that correlates **Digital Security** (e.g., Mirai Botnet detection) with **Physical Safety** (e.g., CO2 hazards) using an open-source observability stack.

## 🏗️ System Architecture
The system follows a **Cloud-Edge Collaborative Model** (specifically an On-Premises Edge/Hub deployment) to ensure low latency and data privacy.



* **The Edge Node (IoT Device):** A hardened **Red Hat Universal Base Image (UBI 9)** container acting as the "Digital Twin" of a secure residential IoT device. It runs a custom Python telemetry agent.
* **The Monitoring Hub:** A local orchestration of **Prometheus** (Time-Series Database) and **Grafana** (Visualization) that scrapes metrics from the edge node in near real-time (5s interval).

## 🧪 Scientific Simulation Methodology
Unlike standard monitoring tools that rely on random data generation, this framework implements **scientifically accurate attack profiles** based on the **CicIoT2023 Dataset**.

The `universal_agent.py` simulates the following scenarios based on empirical traffic signatures:
1.  **Benign Operation:** Low CPU (<5%) and minimal network traffic.
2.  **Mirai Botnet / UDP Flood:** Simulates an infected device participating in a DDoS attack.
    * *Signature:* Network traffic spikes to 5-15 MB/s; CPU utilization > 80%.
3.  **Brute Force Attack:** Simulates a dictionary attack on the device's login service.
    * *Signature:* High rate of failed login attempts (`syn_flag` correlation).
4.  **Physical Safety Hazard:** Simulates environmental anomalies (e.g., CO2 > 2000 ppm) to demonstrate Cyber-Physical assurance.

## 🚀 Getting Started

### Prerequisites
* Docker & Docker Compose

### Installation & Deployment
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/HOne1987/IoT-security-monitoring.git](https://github.com/HOne1987/IoT-security-monitoring.git)
    cd IoT-security-monitoring
    ```

2.  **Launch the Stack:**
    Build the hardened UBI images and start the services.
    ```bash
    docker-compose up --build -d
    ```

3.  **Access the Dashboard:**
    * Open your browser to: `http://localhost:3000`
    * **Login:** `admin` / `admin`
    * *Note:* The data source (Prometheus) is **automatically provisioned** via configuration files, so you do not need to configure connections manually.

### Usage
* **Visualizing Attacks:** The simulation loop runs automatically. Watch the dashboard for 60 seconds to see the cycle:
    * `0s - 30s`: Normal Traffic (Green status).
    * `30s - 45s`: **Mirai Attack** (Network/CPU Spike).
    * `45s - 60s`: **Brute Force** (Login Counter Spike).
* **Persistence:** Grafana dashboards and users are saved to a Docker volume (`grafana_storage`), so your changes persist across restarts.

## 📂 Project Structure
```text
.
├── IoT_Device/             # The Hardened Edge Artifact
│   ├── Dockerfile          # Red Hat UBI 9 Minimal Config (NIST Hardened)
│   └── universal_agent.py  # Telemetry Agent & CicIoT2023 Simulator
├── Hub_Configs/            # Monitoring Infrastructure
│   ├── prometheus.yml      # Scrape Configs
│   └── grafana/            # Provisioning (Auto-connect Data Sources)
└── docker-compose.yml      # Service Orchestration
