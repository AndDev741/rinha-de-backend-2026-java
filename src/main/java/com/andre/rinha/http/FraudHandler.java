package com.andre.rinha.http;

import com.andre.rinha.json.JsonReader;
import com.andre.rinha.json.Payload;
import com.andre.rinha.trace.Traces;
import com.andre.rinha.vector.Dataset;
import com.andre.rinha.vector.KnnSearcher;
import com.andre.rinha.vector.Vectorizer;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

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
 *
 * Instrumentation: nanoTime() spans around each stage feed Traces. The
 * five timestamps cost ~5×50ns ≈ 0.25µs per request — irrelevant against
 * a ~1ms budget. Disable by stripping the trace calls if you ever need
 * the absolute last microsecond.
 */
public final class FraudHandler implements HttpHandler {

    // Pre-built response bytes — KnnSearcher returns one of 6 scores
    // (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), each pairs with a single approved
    // value (score < 0.6 ⇒ approved=true). So there are exactly 6 possible
    // responses. Skip StringBuilder + String.toBytes per request.
    private static final byte[][] RESPONSES = new byte[6][];

    static {
        for (int frauds = 0; frauds <= 5; frauds++) {
            float score = frauds / 5f;
            boolean approved = score < 0.6f;
            String body = "{\"approved\":" + approved
                        + ",\"fraud_score\":" + score + "}";
            RESPONSES[frauds] = body.getBytes(StandardCharsets.UTF_8);
        }
    }

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
        final long t0 = System.nanoTime();
        try {
            if (!"POST".equals(ex.getRequestMethod())) {
                ex.sendResponseHeaders(405, -1);
                return;
            }

            byte[] body = readAll(ex.getRequestBody());
            final long tRead = System.nanoTime();

            Payload p = JsonReader.parse(body);
            final long tParse = System.nanoTime();

            float[] q = queryTl.get();
            Vectorizer.vectorize(p, q);
            final long tVec = System.nanoTime();

            float score = searcherTl.get().fraudScore(q);
            final long tKnn = System.nanoTime();

            // score = frauds/5, so frauds = round(score*5). Multiplication is
            // exact for these denominators — no floating-point surprises.
            int frauds = Math.round(score * 5f);
            byte[] resp = RESPONSES[frauds];
            ex.getResponseHeaders().add("Content-Type", "application/json");
            ex.sendResponseHeaders(200, resp.length);
            ex.getResponseBody().write(resp);
            final long tEnd = System.nanoTime();

            Traces.recordNs(Traces.Span.READ,  tRead  - t0);
            Traces.recordNs(Traces.Span.PARSE, tParse - tRead);
            Traces.recordNs(Traces.Span.VEC,   tVec   - tParse);
            Traces.recordNs(Traces.Span.KNN,   tKnn   - tVec);
            Traces.recordNs(Traces.Span.RESP,  tEnd   - tKnn);
            Traces.recordNs(Traces.Span.TOTAL, tEnd   - t0);
        } catch (Exception e) {
            System.err.println("[fraud] error: " + e.getMessage());
            try {
                ex.sendResponseHeaders(500, -1);
            } catch (IOException ignored) { }
        } finally {
            ex.close();
        }
    }

    private static byte[] readAll(InputStream is) throws IOException {
        return is.readAllBytes();
    }
}
