# IoT-security-monitoring
This repository is for my MSc -in Cybersecurity- project that's about monitoring the security of IoT devices using Grafana Alloy, Grafana Loki, Prometheus and Grafana in order to monitor and visualize data that correlate to each IoT device.
The data that should be useful for monitoring each IoT device's security status would be as follows:
-Systemd Journals
-System logs (e.g. boot.log)
-Grype/syft analysis of CVEs found in packages that are installed in the system (WIP)
-Overall Usage of System Resources (e.g. CPU Usage, RAM usage, etc..)

The repository includes two folders which help show the user what to install in a server/client architecture, the clients would be the IoT devices and the Server is the main server used for analysis.
These folders mainly contain Grafana Alloy/Loki configs as well as the main Grafana server's Configuration for visualization of the previously explained security features for each IoT device.
