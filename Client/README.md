# Client Configurations
In this folder we have the Grafana Alloy Configuration, which retrieves logs from journalctl and boot.log, the configuration was derived from a template made by the grafana team in https://github.com/grafana/alloy-scenarios which helps facilitate the collection of logs in the above mentioned places.
And a modified loki configuration which simply helps with integration with grafana alloy and makes it to be used with grafana as a data source.
There is also a script for generating Grype logs, which help show CVEs of packages inside the system. This script should be run periodically via cron or systemd in order to generate periodical logs of CVEs present in the system. An example would be the following crontab file: "* * * * 1 /usr/local/bin/run_grype_scan.sh".

# Data Collection
Grafana Alloy collects the following data from the following sources:
- Bootloader logs from /var/log/boot.log
- Journald logs from journald
- System Package CVEs from Grype Logs

# WIP (Work In Progress)
- A solution for intrusion detection of packets sent/recieved from/to the IoT client
