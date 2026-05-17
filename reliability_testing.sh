# 1. Start with proper dependency order
docker-compose down
docker-compose up > logs/full-system.log 2>&1 &
COMPOSE_PID=$!

# 2. Wait for warm-up to complete (120 * 5s = 600s)
echo "Waiting for warm-up to complete (10 minutes)..."
sleep 600

# 3. Log container stats
echo "System stability test (1 hour)..."
docker stats --no-stream > logs/docker_stats_baseline.txt
sleep 3600

# 4. Capture logs
docker-compose logs prometheus > logs/prometheus.log 2>&1
docker-compose logs ai-analyzer > logs/detector.log 2>&1

# 5. Analyze errors
echo "=== PROMETHEUS ERRORS ===" 
grep -i "error\|failed" logs/prometheus.log | head -20
echo ""
echo "=== DETECTOR ERRORS ===" 
grep -i "error\|failed" logs/detector.log | head -20

# 6. Summary
echo ""
echo "=== SUMMARY ===" 
echo "Total prometheus errors: $(grep -ic 'error\|failed' logs/prometheus.log)"
echo "Total detector errors: $(grep -ic 'error\|failed' logs/detector.log)"
echo "Container uptime: $(docker ps -a --format 'table {{.Names}}\t{{.Status}}')"
