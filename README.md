# IoT-security-monitoring
This repository is for my MSc -in Cybersecurity- project that's about monitoring the security of IoT devices using Prometheus and Grafana, the python file "IoT_Device/universal_agent.py" exports metrics of the IoT device for prometheus to get the data as well as a simulated attack that happens with a chance of 10% to be able to define alerts that coorelate to these attacks and their effect on system resources. As well as a simulated CO2 sensor which also has a chance of 5% to provide inaccurate readings for us to detect and visualize as well.

## How to use
You can easily spin up the containers defined in the docker-compose file using the following command "docker-compose up --build" and when a chance of anomaly happens it'll show in the terminal.
The grafana instance is accessible at localhost:3000 and prometheus can be connected using the container's ip address. a grafana JSON file is also provided for ease of importing an already defined dashboard which shows the system's metrics and anomaly metrics as well.
