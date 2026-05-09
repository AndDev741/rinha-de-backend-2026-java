package com.andre.rinha.http;

import com.andre.rinha.json.JsonReader;
import com.andre.rinha.json.Payload;
import com.andre.rinha.vector.Dataset;
import com.andre.rinha.vector.KnnSearcher;
import com.andre.rinha.vector.Vectorizer;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.InputStream;

/**
 * POST /fraud-score — main endpoint.
 *
 * Per-request flow:
 *   1. Read the entire body (just a few KB — streaming isn't worth it).
 *   2. Parse JSON manually → Payload.
 *   3. Vectorize into float[14].
 *   4. Brute-force k-NN → fraud_score.
 *   5. Write JSON response manually.
 *
 * Concurrency decisions:
 *   - The JDK HttpServer is thread-per-request via an ExecutorService.
 *   - Vectorizer is stateless (only static methods).
 *   - KnnSearcher has a reusable heap per instance, so we use
 *     ThreadLocal<KnnSearcher>. Avoids reallocating per request.
 *   - Query buffer (float[14]) is also ThreadLocal.
 *
 * Response: JSON bytes built by hand. Only two possible shapes, so a
 * simple StringBuilder is more than enough.
 */
public final class FraudHandler implements HttpHandler {

    private final Dataset dataset;

    // ThreadLocals: each thread has its own KnnSearcher and query buffer.
    private final ThreadLocal<KnnSearcher> searcherTl;
    private final ThreadLocal<float[]> queryTl = ThreadLocal.withInitial(() -> new float[Vectorizer.DIMS]);

    public FraudHandler(Dataset dataset) {
        this.dataset = dataset;
        this.searcherTl = ThreadLocal.withInitial(() -> new KnnSearcher(this.dataset));
    }

    @Override
    public void handle(HttpExchange ex) throws IOException {
        try {
            if (!"POST".equals(ex.getRequestMethod())) {
                ex.sendResponseHeaders(405, -1);
                return;
            }

            // 1. Read the entire body.
            byte[] body = readAll(ex.getRequestBody());

            // 2. Parse.
            Payload p = JsonReader.parse(body);

            // 3. Vectorize.
            float[] q = queryTl.get();
            Vectorizer.vectorize(p, q);

            // 4. k-NN.
            float score = searcherTl.get().fraudScore(q);
            boolean approved = score < 0.6f;

            // 5. Response.
            byte[] resp = buildResponse(approved, score);
            ex.getResponseHeaders().add("Content-Type", "application/json");
            ex.sendResponseHeaders(200, resp.length);
            ex.getResponseBody().write(resp);
        } catch (Exception e) {
            // Minimal logging — in production we'd need to know, but without
            // bloating the response.
            System.err.println("[fraud] error: " + e.getMessage());
            try {
                ex.sendResponseHeaders(500, -1);
            } catch (IOException ignored) { }
        } finally {
            ex.close();
        }
    }

    private static byte[] readAll(InputStream is) throws IOException {
        // Typical body ~500 bytes. 2KB is more than enough as initial capacity.
        return is.readAllBytes();
    }

    /**
     * Builds {"approved":true,"fraud_score":0.0}.
     *
     * Using Float.toString for the score: acceptable for the 0.0/0.2/0.4/0.6/0.8/1.0
     * range produced by the 0..5 / 5 dividend.
     */
    private static byte[] buildResponse(boolean approved, float score) {
        StringBuilder sb = new StringBuilder(40);
        sb.append("{\"approved\":").append(approved)
          .append(",\"fraud_score\":").append(score)
          .append('}');
        return sb.toString().getBytes();
    }
}
