package com.andre.rinha.trace;

import java.util.concurrent.atomic.AtomicLongArray;

/**
 * Lock-free latency histograms per span.
 *
 * Six fixed spans (READ, PARSE, VEC, KNN, RESP, TOTAL). Each has a
 * 20-bucket log-spaced histogram in microseconds. Every span record
 * atomically increments one bucket — ~50ns under contention with two
 * workers, irrelevant against the request budget.
 *
 * Snapshot reads bucket counts directly. Counters may drift slightly
 * during reads under load, but the error is bounded by the request rate
 * × snapshot duration and we only use this for offline analysis.
 *
 * Buckets are upper-bounds (microseconds):
 *   1, 2, 5, 10, 20, 50, 100, 200, 500,
 *   1000, 2000, 5000, 10000, 20000, 50000,
 *   100000, 200000, 500000, 1000000, +inf
 */
public final class Traces {

    public enum Span { READ, PARSE, VEC, KNN, RESP, TOTAL }

    // Refined buckets: extra resolution in the 100µs–10ms range where
    // KNN p50/p99 actually lives.
    public static final long[] BUCKET_UPPER_US = {
            1, 2, 5, 10, 20, 50,
            100, 150, 200, 300, 400, 500, 700,
            1_000, 1_500, 2_000, 3_000, 4_000, 5_000, 7_000,
            10_000, 15_000, 20_000, 30_000, 50_000, 70_000,
            100_000, 200_000, 500_000, 1_000_000, Long.MAX_VALUE
    };

    public static final int NUM_BUCKETS = BUCKET_UPPER_US.length;
    public static final int NUM_SPANS = Span.values().length;

    private static final AtomicLongArray[] HIST = new AtomicLongArray[NUM_SPANS];

    static {
        for (int i = 0; i < NUM_SPANS; i++) {
            HIST[i] = new AtomicLongArray(NUM_BUCKETS);
        }
    }

    private Traces() {}

    /** Record one observation. Pass elapsed time in nanoseconds. */
    public static void recordNs(Span span, long nanos) {
        long us = nanos / 1_000;
        int bucket = bucketFor(us);
        HIST[span.ordinal()].incrementAndGet(bucket);
    }

    /** Returns the snapshot of bucket counts for a span. */
    public static long[] snapshot(Span span) {
        AtomicLongArray a = HIST[span.ordinal()];
        long[] copy = new long[NUM_BUCKETS];
        for (int i = 0; i < NUM_BUCKETS; i++) copy[i] = a.get(i);
        return copy;
    }

    /** Reset all counters (useful between test phases). */
    public static void reset() {
        for (AtomicLongArray a : HIST) {
            for (int i = 0; i < NUM_BUCKETS; i++) a.set(i, 0);
        }
    }

    /**
     * Returns the upper bound (µs) of the bucket containing the given
     * percentile. Computed by walking buckets until cumulative count
     * reaches the target.
     */
    public static long percentileUs(long[] buckets, double percentile) {
        long total = 0;
        for (long c : buckets) total += c;
        if (total == 0) return 0;
        long target = (long) Math.ceil(total * percentile);
        long acc = 0;
        for (int i = 0; i < buckets.length; i++) {
            acc += buckets[i];
            if (acc >= target) return BUCKET_UPPER_US[i];
        }
        return BUCKET_UPPER_US[buckets.length - 1];
    }

    public static long countOf(long[] buckets) {
        long t = 0;
        for (long c : buckets) t += c;
        return t;
    }

    private static int bucketFor(long us) {
        // Linear scan: 20 entries, faster than branch-heavy binary search
        // for this size on amd64.
        for (int i = 0; i < BUCKET_UPPER_US.length; i++) {
            if (us <= BUCKET_UPPER_US[i]) return i;
        }
        return BUCKET_UPPER_US.length - 1;
    }
}
