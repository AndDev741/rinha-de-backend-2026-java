package com.andre.rinha.http;

import com.andre.rinha.json.JsonReader;
import com.andre.rinha.json.Payload;
import com.andre.rinha.vector.Dataset;
import com.andre.rinha.vector.KnnSearcher;
import com.andre.rinha.vector.Vectorizer;
import org.microhttp.EventLoop;
import org.microhttp.Header;
import org.microhttp.Options;
import org.microhttp.Response;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * v9 spike — POST /fraud-score served by a single-threaded, non-blocking NIO
 * event loop (microhttp) instead of the JDK {@code com.sun.net.httpserver}.
 *
 * Why: the JDK server is thread-per-request and blocking. On the rinha box each
 * API gets 0.45 CPU, so a thread pool only contends with itself and the p99 tail
 * balloons under the k6 ramp. A single event-loop thread runs flat-out with no
 * self-contention and no per-request thread handoff — the model the top solutions
 * use. This keeps the proven Vectorizer / KnnSearcher / JsonReader untouched and
 * swaps only the HTTP layer, so the rinha score isolates the transport win.
 *
 * Concurrency model (microhttp):
 *   - one acceptor thread (mostly idle under keep-alive)
 *   - {@code concurrency} connection event-loop threads; the handler runs on them.
 *   With {@code concurrency=1} the hot path is single-threaded. ThreadLocal state
 *   keeps it correct if concurrency is bumped for experiments.
 *
 * The handler MUST NOT block: vectorize + search is ~microseconds, well under the
 * event-loop budget, so we compute the response inline and invoke the callback
 * synchronously (no offload, no extra thread).
 */
public final class MicrohttpServer {

    private static final List<Header> JSON_HEADERS =
            List.of(new Header("Content-Type", "application/json"));
    private static final List<Header> NO_HEADERS = List.of();

    private static final String FRAUD_SCORE_URI = "/fraud-score";
    private static final String READY_URI = "/ready";

    /**
     * Six pre-built responses, byte-identical to {@link FraudHandler}: KnnSearcher
     * returns frauds in 0..5, score = frauds/5, approved = score < 0.6. Reusing the
     * immutable {@link Response} records avoids allocating Response/Header per request.
     */
    private static final Response[] FRAUD_RESPONSES = new Response[6];

    /** Fail-safe: on any parse/search error, deny (score 0.6) with a 200 instead of a
     *  5xx. An HTTP error weighs 5 in the detection score AND counts as a failure;
     *  a single FP/FN is far cheaper. Valid payloads never hit this path. */
    private static final int FAIL_SAFE_FRAUDS = 3;

    private static final Response READY_RESPONSE =
            new Response(200, "OK", NO_HEADERS, new byte[0]);
    private static final Response NOT_FOUND =
            new Response(404, "Not Found", NO_HEADERS, new byte[0]);

    static {
        for (int frauds = 0; frauds <= 5; frauds++) {
            float score = frauds / 5f;
            boolean approved = score < 0.6f;
            String body = "{\"approved\":" + approved
                        + ",\"fraud_score\":" + score + "}";
            FRAUD_RESPONSES[frauds] =
                    new Response(200, "OK", JSON_HEADERS, body.getBytes(StandardCharsets.UTF_8));
        }
    }

    private MicrohttpServer() {}

    public static void start(int port, Dataset dataset, int concurrency) throws IOException {
        // Per-event-loop-thread state — no synchronization on the hot path.
        ThreadLocal<KnnSearcher> searcherTl =
                ThreadLocal.withInitial(() -> new KnnSearcher(dataset));
        ThreadLocal<float[]> queryTl =
                ThreadLocal.withInitial(() -> new float[Vectorizer.DIMS]);

        Options options = Options.builder()
                .withHost("0.0.0.0")          // bind all interfaces; nginx connects over the docker net
                .withPort(port)
                .withConcurrency(concurrency) // 1 = single non-blocking event-loop thread
                .withReuseAddr(true)
                .withAcceptLength(1024)       // accept backlog absorbs the k6 ramp burst
                .build();

        EventLoop eventLoop = new EventLoop(options, (request, callback) -> {
            String uri = request.uri();
            if (FRAUD_SCORE_URI.equals(uri) && "POST".equals(request.method())) {
                int frauds;
                try {
                    Payload p = JsonReader.parse(request.body());
                    float[] q = queryTl.get();
                    Vectorizer.vectorize(p, q);
                    frauds = Math.round(searcherTl.get().fraudScore(q) * 5f);
                } catch (Exception e) {
                    frauds = FAIL_SAFE_FRAUDS;
                }
                callback.accept(FRAUD_RESPONSES[frauds]);
            } else if (READY_URI.equals(uri)) {
                callback.accept(READY_RESPONSE);
            } else {
                callback.accept(NOT_FOUND);
            }
        });

        eventLoop.start();
        System.out.printf("[app] microhttp listening on :%d (concurrency=%d)%n", port, concurrency);
    }
}
