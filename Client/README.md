#Client Configurations

In this folder we have the Grafana Alloy Configuration, which retrieves logs from journalctl and boot.log, the configuration was derived from a template made by the grafana team in https://github.com/grafana/alloy-scenarios which helps facilitate the collection of logs in the above mentioned places.
And a modified loki configuration which simply helps with integration with grafana alloy and makes it to be used with grafana as a data source.

Data collection from Grype/Syft is still under progress..
