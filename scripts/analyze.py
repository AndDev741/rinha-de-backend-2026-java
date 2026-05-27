#!/usr/bin/env python3
"""Parse instrumentation outputs and emit a markdown timeline table.

Inputs (in test/instrument/):
  stats-api-1.json, stats-api-2.json  — app-side per-span histograms
  nginx-timing.log                     — nginx access log w/ timing fields

Output: prints a markdown table to stdout summarising p50, p90, p99, p999
per span (across api-1+api-2 combined) plus the nginx-vs-upstream split.
"""
import json
import math
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "test" / "instrument"


def load_stats(path):
    with open(path) as f:
        return json.load(f)


def merge_hist(s1, s2):
    """Combine two backend histograms element-wise per span."""
    spans = {}
    for name in s1["spans"]:
        h1 = s1["spans"][name]["hist"]
        h2 = s2["spans"][name]["hist"]
        spans[name] = [a + b for a, b in zip(h1, h2)]
    return spans


def percentile_us(hist, buckets_us, p):
    total = sum(hist)
    if total == 0:
        return 0
    target = math.ceil(total * p)
    acc = 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= target:
            return buckets_us[i]
    return buckets_us[-1]


def fmt_us(us):
    if us is None:
        return "n/a"
    if us >= 1000:
        return f"{us/1000:.2f}ms"
    return f"{us}µs"


def parse_nginx_log(path):
    """Return list of (request_time_ms, upstream_response_time_ms)."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                req = float(parts[1]) * 1000  # seconds → ms
                ups_field = parts[2]
                # may be '-' for short-circuit
                ups = float(ups_field) * 1000 if ups_field != "-" else None
                rows.append((req, ups))
            except ValueError:
                continue
    return rows


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(math.ceil(q * len(values))) - 1))
    return values[idx]


def main():
    s1_path = OUT / "stats-api-1.json"
    s2_path = OUT / "stats-api-2.json"
    nginx_path = OUT / "nginx-timing.log"

    if not s1_path.exists() or not s2_path.exists():
        print(f"missing stats files in {OUT}", file=sys.stderr)
        sys.exit(1)

    s1 = load_stats(s1_path)
    s2 = load_stats(s2_path)
    # Use bucket bounds; null in JSON = +inf.
    buckets_us = [b if b is not None else float("inf") for b in s1["buckets_us"]]
    merged = merge_hist(s1, s2)

    order = ["READ", "PARSE", "VEC", "KNN", "RESP", "TOTAL"]
    total_count = sum(merged["TOTAL"])

    print("# Latency timeline (real measurements)\n")
    print(f"**Sample size:** {total_count:,} requests across api-1 + api-2\n")

    print("## App-side spans (instrumented in FraudHandler)\n")
    print("| Span | count | p50 | p90 | p99 | p999 |")
    print("|---|---:|---:|---:|---:|---:|")
    for span in order:
        h = merged[span]
        cnt = sum(h)
        p50 = percentile_us(h, buckets_us, 0.50)
        p90 = percentile_us(h, buckets_us, 0.90)
        p99 = percentile_us(h, buckets_us, 0.99)
        p999 = percentile_us(h, buckets_us, 0.999)
        print(f"| {span} | {cnt:,} | {fmt_us(p50)} | {fmt_us(p90)} | "
              f"{fmt_us(p99)} | {fmt_us(p999)} |")

    print("\n## nginx vs upstream (nginx access log)\n")
    if nginx_path.exists():
        rows = parse_nginx_log(nginx_path)
        print(f"**Sample size:** {len(rows):,} requests in nginx log\n")
        req_times = [r for r, _ in rows]
        ups_times = [u for _, u in rows if u is not None]
        # Per-request gap between request_time and upstream_response_time.
        # This is what nginx + network + connect added on top of upstream.
        gaps = [r - u for r, u in rows if u is not None]

        print("| Metric | p50 | p90 | p99 | p999 |")
        print("|---|---:|---:|---:|---:|")
        for label, vals in [
            ("request_time (total at nginx)", req_times),
            ("upstream_response_time (app)", ups_times),
            ("nginx + net overhead (req − ups)", gaps),
        ]:
            p50 = quantile(vals, 0.50)
            p90 = quantile(vals, 0.90)
            p99 = quantile(vals, 0.99)
            p999 = quantile(vals, 0.999)
            print(f"| {label} | {p50:.2f}ms | {p90:.2f}ms | {p99:.2f}ms | "
                  f"{p999:.2f}ms |")
    else:
        print("(nginx-timing.log missing — skipping nginx breakdown)")

    print("\n## Histogram detail (TOTAL span)\n")
    print("| upper µs | count |")
    print("|---:|---:|")
    h_total = merged["TOTAL"]
    for i, c in enumerate(h_total):
        if c == 0:
            continue
        ub = buckets_us[i]
        label = "+∞" if ub == float("inf") else f"{int(ub)}"
        print(f"| {label} | {c:,} |")


if __name__ == "__main__":
    main()
