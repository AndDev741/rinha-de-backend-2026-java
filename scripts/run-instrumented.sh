#!/bin/bash
# Orchestrates: build → up → wait ready → reset stats → load → collect.
#
# Outputs land in ./test/instrument/:
#   k6-summary.txt           — k6 console output
#   stats-api-1.json         — per-span histograms from api-1
#   stats-api-2.json         — per-span histograms from api-2
#   nginx-timing.log         — request_time + upstream_response_time per req
#
# Usage: ./scripts/run-instrumented.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/test/instrument"
mkdir -p "$OUT"

cd "$ROOT"

echo "[run] bringing stack up"
docker compose -f docker-compose.native.yaml up -d

echo "[run] waiting for nginx /ready"
for i in $(seq 1 60); do
  if curl -sf -o /dev/null http://127.0.0.1:9999/ready; then
    echo "[run] ready after ${i}s"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "[run] FAIL: not ready after 60s" >&2
    docker compose -f docker-compose.native.yaml logs --tail 50 >&2
    exit 1
  fi
  sleep 1
done

echo "[run] resetting per-backend stats"
curl -sf -X POST http://127.0.0.1:9999/stats/api-1/reset -o /dev/null
curl -sf -X POST http://127.0.0.1:9999/stats/api-2/reset -o /dev/null

# Truncate nginx log so we get only this run's data.
docker compose -f docker-compose.native.yaml exec -T lb sh -c 'echo -n > /tmp/nginx-timing.log'

echo "[run] running k6 load test (30s @ 800 RPS)"
k6 run --quiet --summary-export "$OUT/k6-summary.json" \
       "$ROOT/scripts/load-test.js" 2>&1 | tee "$OUT/k6-summary.txt"

# Let any in-flight requests drain and nginx flush its log buffer.
sleep 2

echo "[run] collecting stats"
curl -sf http://127.0.0.1:9999/stats/api-1 > "$OUT/stats-api-1.json"
curl -sf http://127.0.0.1:9999/stats/api-2 > "$OUT/stats-api-2.json"
docker compose -f docker-compose.native.yaml exec -T lb cat /tmp/nginx-timing.log \
    > "$OUT/nginx-timing.log"

echo "[run] sizes:"
ls -la "$OUT/"

echo "[run] done. Results in $OUT"
