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
 * Entry point.
 *
 * Environment variables:
 *   PORT      — HTTP port (default 9999, as required by the challenge)
 *   DATA_DIR  — directory containing vectors.bin + labels.bin (default ./data)
 *   THREADS   — HttpServer thread pool size (default = number of CPUs)
 *
 * Initialization order matters:
 *   1. Load the dataset FIRST. Takes ~3-10s for 168MB of data.
 *   2. Only then bind/start the socket.
 *   3. /ready only responds once we get here — the load balancer detects
 *      readiness automatically.
 */
public final class App {

    public static void main(String[] args) throws Exception {
        int port = envInt("PORT", 9999);
        Path dataDir = Path.of(env("DATA_DIR", "./data"));
        int threads = envInt("THREADS", Runtime.getRuntime().availableProcessors());

        long t0 = System.currentTimeMillis();
        System.out.println("[app] loading dataset from " + dataDir.toAbsolutePath());
        Dataset dataset = Dataset.load(dataDir);
        long loadMs = System.currentTimeMillis() - t0;
        System.out.printf("[app] dataset loaded: %d vectors in %d ms%n",
                dataset.count(), loadMs);

        // Minimal JIT warmup: runs a dummy search so the first real request
        // doesn't pay the cost of compilation.
        warmup(dataset);

        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/ready", new ReadyHandler());
        server.createContext("/fraud-score", new FraudHandler(dataset));
        // Fixed pool. For 1 vCPU, 2-4 threads is enough — more than that just
        // generates context-switch overhead with no real gain.
        server.setExecutor(Executors.newFixedThreadPool(threads));
        server.start();

        System.out.printf("[app] listening on :%d with %d threads%n", port, threads);
    }

    /**
     * "Warms up" the JIT by running 1k searches with random vectors. Makes C2
     * compile the hot paths before the first real request, reducing the early
     * p99 of the test.
     */
    private static void warmup(Dataset ds) {
        var searcher = new KnnSearcher(ds);
        float[] q = new float[Dataset.DIMS];
        Random rng = new java.util.Random(42);
        for (int i = 0; i < 1000; i++) {
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
