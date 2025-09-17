#!/bin/bash
##Done with the help of copilot AI

SCAN_LOG="/var/log/grype/grype_scan.json"
TMP_JSON="/tmp/full_grype.json"

grype dir:/ --output json > "$TMP_JSON"
jq -c '.matches[]
  | select(.vulnerability.severity | test("Critical|High|Medium"))
  | select((.vulnerability.fix == null) or (.vulnerability.fix.versions == []))' "$TMP_JSON" >> "$SCAN_LOG"
