package com.andre.rinha;

import com.andre.rinha.http.FraudHandler;
import com.andre.rinha.http.ReadyHandler;
import com.andre.rinha.http.StatsHandler;
import com.andre.rinha.vector.Dataset;
import com.andre.rinha.vector.KnnSearcher;
import com.sun.net.httpserver.HttpServer;

import java.net.InetSocketAddress;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.Executors;

/**
 * Entry point — v7 (int16 IVF with bbox repair).
 *
 * Environment variables:
 *   PORT      — HTTP port (default 9999, as required by the challenge)
 *   DATA_DIR  — directory with vectors-i16.bin, centroids-i16.bin, bbox.bin,
 *               cluster_offsets.bin, labels.bin (default ./data)
 *
 * Initialization order:
 *   1. Load the int16 IVF dataset (~84 MB heap).
 *   2. JIT warmup over the hot path.
 *   3. Bind and start. /ready only responds once we reach this point.
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

        warmup(dataset);

        // Backlog of 4096 absorbs the rinha k6 ramp (peaks ~900 RPS).
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 4096);
        server.createContext("/ready", new ReadyHandler());
        server.createContext("/fraud-score", new FraudHandler(dataset));
        server.createContext("/stats", new StatsHandler());
        // v7.1 (virtual threads) hit -6000 on rinha because every request
        // competed for the same 0.45 CPU, dragging p99 to the 2001 ms k6
        // timeout. v7.2 reverts to a small fixed pool: bounded concurrency
        // protects per-request latency, the accept backlog absorbs bursts.
        int workers = envInt("WORKERS", 6);
        server.setExecutor(Executors.newFixedThreadPool(workers));
        server.start();

        System.out.printf("[app] listening on :%d with %d worker threads%n", port, workers);
    }

    /**
     * Warm up the JIT against the real dataset. Random queries mostly bbox-
     * prune to nothing, so they don't exercise the cluster-scan hot path
     * that real fraud queries trigger. We sample 2000 dataset vectors as
     * near-self queries: each one forces a full cluster scan plus bbox-
     * repair against a few neighbours, which mirrors real traffic.
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

    private static String env(String key, String def) {
        String v = System.getenv(key);
        return v == null ? def : v;
    }

    private static int envInt(String key, int def) {
        String v = System.getenv(key);
        return v == null ? def : Integer.parseInt(v);
    }
}
