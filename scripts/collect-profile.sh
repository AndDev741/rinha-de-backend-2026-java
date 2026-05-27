#!/bin/sh
# Drives the instrumented native binary to produce default.iprof.
#
# Started in background, waits for /ready, fires a diverse workload
# that covers the main classifier branches, then SIGTERMs the binary
# so the GraalVM runtime flushes the profile cleanly on shutdown.
#
# Profile lands in $PWD/default.iprof, which is what the final
# `--pgo=` build reads.
#
# IMPORTANT: this script expects the instrumented binary to be run
# with SKIP_WARMUP=1 so the App.java synthetic warmup (2000 random
# near-self queries) doesn't pollute the profile. The Dockerfile
# sets that env var before invoking this script.

set -eu

BIN="$1"
PORT="${PORT:-9999}"
URL="http://127.0.0.1:${PORT}"

T_START=$(date +%s)
echo "[profile] starting instrumented binary: $BIN"
"$BIN" >/tmp/app.log 2>&1 &
APP_PID=$!
echo "[profile] app pid=$APP_PID"

# Wait for /ready. Without warmup, this should be ~1–2s (just
# dataset load). With instrumentation, expect 3–8s.
i=0
until curl -sf -o /dev/null "${URL}/ready"; do
  i=$((i+1))
  if [ "$i" -gt 120 ]; then
    echo "[profile] FAIL: /ready never came up after ${i}s" >&2
    echo "[profile] --- app log ---" >&2
    cat /tmp/app.log >&2 || true
    kill -9 "$APP_PID" 2>/dev/null || true
    exit 1
  fi
  if [ $((i % 5)) -eq 0 ]; then
    echo "[profile] waiting for /ready... ${i}s elapsed"
  fi
  sleep 1
done
T_READY=$(date +%s)
echo "[profile] /ready up after $((T_READY - T_START))s — workload starts"

# 10 payloads spanning the main branches of Vectorizer + KnnSearcher.
# Each row varies multiple dimensions (amount, installments, hour,
# known_merchants membership, last_transaction null/present, online
# and card_present combinations, mcc category, km_from_home).
#
#  P1:  mid amount, merchant IN known, near home, recent prior tx
#  P2:  high amount, merchant OUT, far from home, no prior tx
#  P3:  low amount, business hours, very recent prior tx, trusted
#  P4:  mid-high amount, empty known_merchants, last_tx far
#  P5:  low amount, mid-list known_merchants, online, no prior tx
#  P6:  very high amount, 24 installments, dawn, merchant OUT
#  P7:  tiny amount, 1 installment, weekend afternoon, no last_tx
#  P8:  mid amount, online but card NOT present, long known list
#  P9:  high amount but merchant IN known list, mid-day
#  P10: ultra-high amount, terminal offline, 7-merchant known list
P1='{"id":"a","transaction":{"amount":250.5,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":180.0,"tx_count_24h":4,"known_merchants":["M-007","M-002"]},"merchant":{"id":"M-007","mcc":"5411","avg_amount":220.0},"terminal":{"is_online":true,"card_present":true,"km_from_home":12.5},"last_transaction":{"timestamp":"2026-03-11T18:00:00Z","km_from_current":5.2}}'
P2='{"id":"b","transaction":{"amount":4500.0,"installments":12,"requested_at":"2026-03-11T03:14:00Z"},"customer":{"avg_amount":50.0,"tx_count_24h":1,"known_merchants":["M-100"]},"merchant":{"id":"M-999","mcc":"5812","avg_amount":3200.0},"terminal":{"is_online":false,"card_present":false,"km_from_home":85.0},"last_transaction":null}'
P3='{"id":"c","transaction":{"amount":15.0,"installments":1,"requested_at":"2026-03-11T12:00:00Z"},"customer":{"avg_amount":18.0,"tx_count_24h":8,"known_merchants":["M-001","M-002","M-003","M-004"]},"merchant":{"id":"M-001","mcc":"5411","avg_amount":20.0},"terminal":{"is_online":true,"card_present":true,"km_from_home":1.5},"last_transaction":{"timestamp":"2026-03-11T11:30:00Z","km_from_current":0.3}}'
P4='{"id":"d","transaction":{"amount":950.0,"installments":6,"requested_at":"2026-03-11T22:45:00Z"},"customer":{"avg_amount":300.0,"tx_count_24h":2,"known_merchants":[]},"merchant":{"id":"M-555","mcc":"5732","avg_amount":850.0},"terminal":{"is_online":true,"card_present":false,"km_from_home":45.0},"last_transaction":{"timestamp":"2026-03-11T21:00:00Z","km_from_current":40.0}}'
P5='{"id":"e","transaction":{"amount":75.0,"installments":2,"requested_at":"2026-03-11T09:30:00Z"},"customer":{"avg_amount":80.0,"tx_count_24h":5,"known_merchants":["M-010","M-020","M-030"]},"merchant":{"id":"M-020","mcc":"5411","avg_amount":78.0},"terminal":{"is_online":true,"card_present":true,"km_from_home":3.0},"last_transaction":null}'
P6='{"id":"f","transaction":{"amount":7800.0,"installments":24,"requested_at":"2026-03-11T04:50:00Z"},"customer":{"avg_amount":120.0,"tx_count_24h":1,"known_merchants":["M-050","M-051"]},"merchant":{"id":"M-666","mcc":"7995","avg_amount":6500.0},"terminal":{"is_online":true,"card_present":false,"km_from_home":120.0},"last_transaction":null}'
P7='{"id":"g","transaction":{"amount":1.5,"installments":1,"requested_at":"2026-03-08T14:15:00Z"},"customer":{"avg_amount":4.0,"tx_count_24h":3,"known_merchants":["M-300","M-301"]},"merchant":{"id":"M-300","mcc":"5499","avg_amount":2.0},"terminal":{"is_online":true,"card_present":true,"km_from_home":0.5},"last_transaction":null}'
P8='{"id":"h","transaction":{"amount":420.0,"installments":3,"requested_at":"2026-03-11T16:05:00Z"},"customer":{"avg_amount":350.0,"tx_count_24h":6,"known_merchants":["M-A","M-B","M-C","M-D","M-E","M-F","M-G"]},"merchant":{"id":"M-Z","mcc":"5969","avg_amount":380.0},"terminal":{"is_online":true,"card_present":false,"km_from_home":22.0},"last_transaction":{"timestamp":"2026-03-11T15:30:00Z","km_from_current":18.0}}'
P9='{"id":"i","transaction":{"amount":2100.0,"installments":6,"requested_at":"2026-03-11T13:40:00Z"},"customer":{"avg_amount":900.0,"tx_count_24h":2,"known_merchants":["M-777","M-200","M-007"]},"merchant":{"id":"M-777","mcc":"5944","avg_amount":1800.0},"terminal":{"is_online":true,"card_present":true,"km_from_home":8.0},"last_transaction":{"timestamp":"2026-03-11T12:00:00Z","km_from_current":2.1}}'
P10='{"id":"j","transaction":{"amount":8500.0,"installments":18,"requested_at":"2026-03-11T01:20:00Z"},"customer":{"avg_amount":600.0,"tx_count_24h":4,"known_merchants":["M-1","M-2","M-3","M-4","M-5","M-6","M-7"]},"merchant":{"id":"M-X","mcc":"6011","avg_amount":7200.0},"terminal":{"is_online":false,"card_present":true,"km_from_home":60.0},"last_transaction":{"timestamp":"2026-03-10T23:00:00Z","km_from_current":55.0}}'

# 3 rounds × 10 payloads = 30 requests, strictly serial.
# Doubles coverage vs the previous version while keeping time
# bounded (~30–60s on the instrumented binary, no warmup).
ROUNDS=3
PAYLOAD_NAMES="P1 P2 P3 P4 P5 P6 P7 P8 P9 P10"
T_WORKLOAD=$(date +%s)

for round in $(seq 1 $ROUNDS); do
  T_ROUND=$(date +%s)
  idx=0
  for P in "$P1" "$P2" "$P3" "$P4" "$P5" "$P6" "$P7" "$P8" "$P9" "$P10"; do
    idx=$((idx+1))
    NAME=$(echo $PAYLOAD_NAMES | cut -d' ' -f$idx)
    STATS=$(curl -s --max-time 30 -X POST \
                 -H 'Content-Type: application/json' \
                 --data-binary "$P" \
                 -o /dev/null \
                 -w '%{http_code} %{time_total}s' \
                 "${URL}/fraud-score")
    echo "[profile] r${round} ${NAME}: ${STATS}"
  done
  echo "[profile] round ${round}/${ROUNDS} done in $(( $(date +%s) - T_ROUND ))s"
done

T_END_WORKLOAD=$(date +%s)
echo "[profile] workload complete in $((T_END_WORKLOAD - T_WORKLOAD))s, SIGTERMing"
kill -TERM "$APP_PID"
wait "$APP_PID" || true

T_END=$(date +%s)
echo "[profile] total time: $((T_END - T_START))s"

if [ ! -s default.iprof ]; then
  echo "[profile] FAIL: default.iprof missing or empty" >&2
  echo "[profile] --- app log ---" >&2
  cat /tmp/app.log >&2 || true
  exit 1
fi
echo "[profile] default.iprof: $(wc -c < default.iprof) bytes"
