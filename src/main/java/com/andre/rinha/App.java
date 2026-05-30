package com.andre.rinha;

import com.andre.rinha.http.MicrohttpServer;
import com.andre.rinha.vector.Dataset;
import com.andre.rinha.vector.KnnSearcher;

import java.nio.file.Path;
import java.util.Random;

/**
 * Entry point — v9 (int16 IVF with bbox repair, microhttp NIO event loop).
 *
 * Environment variables:
 *   PORT         — HTTP port (default 9999, as required by the challenge)
 *   DATA_DIR     — directory with vectors-i16.bin, centroids-i16.bin, bbox.bin,
 *                  cluster_offsets.bin, labels.bin (default ./data)
 *   CONCURRENCY  — microhttp connection event-loop threads (default 1; a single
 *                  non-blocking loop is ideal on the rinha 0.45 CPU budget)
 *
 * Initialization order:
 *   1. Load the int16 IVF dataset (~84 MB heap).
 *   2. Pre-touch the dataset pages so the first request never page-faults.
 *   3. Warm the search hot path against real-shape queries.
 *   4. Start the microhttp event loop. /ready only answers once we reach here.
 */
public final class App {

    public static void main(String[] args) throws Exception {
        int port = envInt("PORT", 9999);
        Path dataDir = Path.of(env("DATA_DIR", "./data"));

        System.out.println("[app] search: " + KnnSearcher.simdInfo());

        long t0 = System.currentTimeMillis();
        System.out.println("[app] loading int16 IVF dataset from " + dataDir.toAbsolutePath());
        Dataset dataset = Dataset.load(dataDir);
        long loadMs = System.currentTimeMillis() - t0;
        System.out.printf("[app] dataset loaded: %d vectors, %d clusters in %d ms (%.1f MB heap)%n",
                dataset.count(), dataset.k(), loadMs,
                dataset.vectors().length * 2 / (1024.0 * 1024.0));

        preTouch(dataset);
        warmup(dataset);

        // Single-threaded non-blocking NIO event loop. On the rinha 0.45 CPU
        // budget one event-loop thread runs flat-out with no self-contention,
        // which keeps the p99 tail low under the k6 ramp.
        int concurrency = envInt("CONCURRENCY", 1);
        MicrohttpServer.start(port, dataset, concurrency);
    }

    /**
     * Warm up the hot path against the real dataset. Random queries mostly bbox-
     * prune to nothing, so they don't exercise the cluster-scan hot path that
     * real fraud queries trigger. We sample 2000 dataset vectors as near-self
     * queries: each one forces a full cluster scan plus bbox-repair against a
     * few neighbours, which mirrors real traffic.
     */
    private static void warmup(Dataset ds) {
        var searcher = new KnnSearcher(ds);
        short[] vectors = ds.vectors();
        int dims = Dataset.DIMS;
        int count = ds.count();
        float invScale = 1f / Dataset.SCALE;
        float[] q = new float[dims];
        Random rng = new Random(42);
        for (int i = 0; i < 2000; i++) {
            int base = rng.nextInt(count) * dims;
            for (int j = 0; j < dims; j++) {
                q[j] = vectors[base + j] * invScale + (rng.nextFloat() - 0.5f) * 0.005f;
            }
            searcher.fraudScore(q);
        }
        System.out.println("[app] warmup complete");
    }

    /**
     * Touch one element per 4 KB page of every dataset array so the OS
     * commits and pre-faults the pages while we're still on startup,
     * not on the first request that needs them. Without this, native
     * Serial-GC heap pages backing the 84 MB vectors[] array fault in
     * lazily — explaining ~1% of requests hitting 50–100 ms in our
     * 800 RPS measurements.
     *
     * Sums the touched values into a volatile sink so neither the JIT
     * nor native-image's static analyzer can elide the loop as dead.
     */
    private static volatile long BLACKHOLE;

    private static void preTouch(Dataset ds) {
        long t0 = System.currentTimeMillis();
        long sum = 0;
        // short = 2 bytes → 2048 shorts per 4 KB page.
        final int stride = 2048;
        short[] v = ds.vectors();
        for (int i = 0; i < v.length; i += stride) sum += v[i];
        short[] c = ds.centroids();
        for (int i = 0; i < c.length; i += stride) sum += c[i];
        short[] bMin = ds.bboxMin();
        for (int i = 0; i < bMin.length; i += stride) sum += bMin[i];
        short[] bMax = ds.bboxMax();
        for (int i = 0; i < bMax.length; i += stride) sum += bMax[i];
        BLACKHOLE = sum;
        System.out.printf("[app] pre-touched in %d ms (sink=%d)%n",
                System.currentTimeMillis() - t0, sum);
    }

    private static String env(String key, String def) {
        String v = System.getenv(key);
        return v == null ? def : v;
    }

    private static int envInt(String key, int def) {
        String v = System.getenv(key);
        return v == null ? def : Integer.parseInt(v);
    }
}
