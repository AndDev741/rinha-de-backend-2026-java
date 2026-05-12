package com.andre.rinha;

import com.andre.rinha.http.FraudHandler;
import com.andre.rinha.http.ReadyHandler;
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

        // Backlog of 1024 keeps short bursts from hitting the OS accept queue
        // limit. Anything past that returns RST and looks like 502 at nginx.
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 1024);
        server.createContext("/ready", new ReadyHandler());
        server.createContext("/fraud-score", new FraudHandler(dataset));
        // Virtual threads (JDK 21+): each request is its own VT, which gets
        // unmounted while it waits on the socket. We don't need to size a
        // pool against the cgroup CPU limit — the JVM only schedules the
        // ones actively computing the k-NN. This is the fix for the v7
        // rinha submission, which hit 45.9 % errors with a fixed 2-thread pool.
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.start();

        System.out.printf("[app] listening on :%d with virtual threads%n", port);
    }

    /**
     * "Warms up" the JIT by running 1k searches with random vectors. Makes C2
     * compile the hot paths before the first real request, reducing the early
     * p99 of the test.
     */
    private static void warmup(Dataset ds) {
        var searcher = new KnnSearcher(ds);
        float[] q = new float[Dataset.DIMS];
        Random rng = new Random(42);
        for (int i = 0; i < 1000; i++) {
            // Random floats in [0, 1] — matches Vectorizer output range.
            for (int j = 0; j < q.length; j++) q[j] = rng.nextFloat();
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
