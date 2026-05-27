package com.andre.rinha.http;

import com.andre.rinha.trace.Traces;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;

/**
 * GET /stats — dump the per-span latency histograms as JSON.
 *
 * POST /stats/reset — zero all counters (used between test phases).
 *
 * Format:
 *   {
 *     "buckets_us": [1,2,5,...,1000000],
 *     "spans": {
 *        "READ":  {"count":N, "p50":X, "p90":Y, "p99":Z, "p999":W, "hist":[...]},
 *        ...
 *     }
 *   }
 *
 * Histograms are read with a single AtomicLongArray.get per bucket;
 * counters may shift slightly during the snapshot under load but the
 * skew is bounded and doesn't matter for offline analysis.
 */
public final class StatsHandler implements HttpHandler {

    @Override
    public void handle(HttpExchange ex) throws IOException {
        String method = ex.getRequestMethod();
        String path = ex.getRequestURI().getPath();
        try {
            if ("POST".equals(method) && path.endsWith("/reset")) {
                Traces.reset();
                byte[] resp = "{\"ok\":true}".getBytes();
                ex.getResponseHeaders().add("Content-Type", "application/json");
                ex.sendResponseHeaders(200, resp.length);
                ex.getResponseBody().write(resp);
                return;
            }
            if (!"GET".equals(method)) {
                ex.sendResponseHeaders(405, -1);
                return;
            }
            byte[] body = buildJson();
            ex.getResponseHeaders().add("Content-Type", "application/json");
            ex.sendResponseHeaders(200, body.length);
            ex.getResponseBody().write(body);
        } finally {
            ex.close();
        }
    }

    private static byte[] buildJson() {
        StringBuilder sb = new StringBuilder(4096);
        sb.append("{\"buckets_us\":[");
        for (int i = 0; i < Traces.BUCKET_UPPER_US.length; i++) {
            if (i > 0) sb.append(',');
            long u = Traces.BUCKET_UPPER_US[i];
            sb.append(u == Long.MAX_VALUE ? "null" : Long.toString(u));
        }
        sb.append("],\"spans\":{");
        Traces.Span[] spans = Traces.Span.values();
        for (int s = 0; s < spans.length; s++) {
            if (s > 0) sb.append(',');
            Traces.Span span = spans[s];
            long[] hist = Traces.snapshot(span);
            long count = Traces.countOf(hist);
            sb.append('"').append(span.name()).append("\":{");
            sb.append("\"count\":").append(count);
            sb.append(",\"p50\":").append(Traces.percentileUs(hist, 0.50));
            sb.append(",\"p90\":").append(Traces.percentileUs(hist, 0.90));
            sb.append(",\"p99\":").append(Traces.percentileUs(hist, 0.99));
            sb.append(",\"p999\":").append(Traces.percentileUs(hist, 0.999));
            sb.append(",\"hist\":[");
            for (int i = 0; i < hist.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(hist[i]);
            }
            sb.append("]}");
        }
        sb.append("}}");
        return sb.toString().getBytes();
    }
}
