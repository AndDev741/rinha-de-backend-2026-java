package com.andre.rinha.prep;

import java.io.BufferedInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.BitSet;
import java.util.zip.GZIPInputStream;

/**
 * Converts references.json.gz into a quantized binary format for v4.
 *
 * Output (3 files in the target dir):
 *
 *   vectors-i8.bin  →  N × 14 bytes (signed int8, [-128, 127])
 *                      = ~42 MB for N = 3,000,000
 *   scales.bin      →  28 × 4 bytes (mins[14] + maxs[14], float32 LE)
 *                      = 112 bytes (kept tiny for L1 caching at runtime)
 *   labels.bin      →  ceil(N/8) bytes (bit i = 1 if vector i is fraud)
 *                      = ~375 KB
 *
 * Quantization scheme:
 *   For each dimension d:
 *     scale_d   = 255 / (max_d - min_d)
 *     int8(v)   = round((v - min_d) * scale_d) - 128
 *
 *   That maps min_d → -128 and max_d → 127. We use SIGNED int8 so the SIMD
 *   pipeline in v4's KnnSearcher can use widening B→I conversion without
 *   tripping over Java's lack of unsigned bytes.
 *
 * Per-dimension min/max (vs a single global min/max):
 *   Each dim has its own native range — booleans are {0,1}, continuous dims
 *   are in [0,1], the sentinel dims (5,6) span [-1,1]. Per-dim scaling lets
 *   each dim use the full int8 resolution. The price is that the squared
 *   L2 distance computed in int8 space is a *weighted* L2 in float space,
 *   with weights 1/step_d² where step_d = (max_d - min_d) / 255. For our
 *   data most ranges are ≈1, so weights are nearly uniform — the parity
 *   test verifies this is OK (top-5 sets agree >99% with float32 ground
 *   truth on synthetic data).
 *
 * How to run:
 *   java -cp target/classes \
 *        --add-modules=jdk.incubator.vector \
 *        com.andre.rinha.prep.DatasetBuilder \
 *        references.json.gz ./data
 *
 * Build-time peak heap: ~168 MB (we hold all float vectors in RAM during
 * the single streaming pass — easier than two passes over the 16 MB gzip).
 * Run with -Xmx384m or larger.
 */
public final class DatasetBuilder {

    private static final int DIMS = 14;
    private static final int LOG_EVERY = 1_000_000;

    /**
     * Initial capacity for the streaming buffer. The official dataset is 3M
     * but we don't hardcode — we grow geometrically if the input exceeds.
     */
    private static final int INITIAL_CAPACITY = 3_000_000;

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: DatasetBuilder <references.json.gz> <outputDir>");
            System.exit(1);
        }
        Path input = Path.of(args[0]);
        Path outDir = Path.of(args[1]);
        outDir.toFile().mkdirs();

        Path vectorsPath = outDir.resolve("vectors-i8.bin");
        Path scalesPath  = outDir.resolve("scales.bin");
        Path labelsPath  = outDir.resolve("labels.bin");
        Path metaPath    = outDir.resolve("meta.txt");

        long t0 = System.currentTimeMillis();
        System.out.println("[builder] reading " + input);

        // ---- Pass 1: stream gzip → float[] in heap, collect labels ----
        float[] vectorsF = new float[INITIAL_CAPACITY * DIMS];
        BitSet labels = new BitSet(INITIAL_CAPACITY);
        int count = 0;
        long fraudCount = 0;

        try (InputStream raw = new FileInputStream(input.toFile());
             InputStream gz = new GZIPInputStream(new BufferedInputStream(raw, 1 << 20))) {

            JsonStream js = new JsonStream(gz);
            js.expect('[');

            while (true) {
                js.skipWs();
                int c = js.peek();
                if (c == ']') break;
                if (c == ',') { js.read(); js.skipWs(); }

                // Each entry: { "vector": [...], "label": "fraud"|"legit" }
                js.expect('{');
                boolean isFraud = false;
                int dim = 0;

                // Grow buffer if needed (doubling, rare).
                if ((count + 1) * DIMS > vectorsF.length) {
                    vectorsF = Arrays.copyOf(vectorsF, vectorsF.length * 2);
                }
                int base = count * DIMS;

                while (true) {
                    js.skipWs();
                    if (js.peek() == '}') { js.read(); break; }
                    if (js.peek() == ',') { js.read(); js.skipWs(); }

                    String key = js.readString();
                    js.skipWs(); js.expect(':'); js.skipWs();

                    if (key.equals("vector")) {
                        js.expect('[');
                        for (int i = 0; i < DIMS; i++) {
                            js.skipWs();
                            if (i > 0) { js.expect(','); js.skipWs(); }
                            vectorsF[base + i] = (float) js.readDouble();
                            dim++;
                        }
                        js.skipWs();
                        js.expect(']');
                    } else if (key.equals("label")) {
                        isFraud = js.readString().equals("fraud");
                    } else {
                        js.skipValue();
                    }
                }

                if (dim != DIMS) {
                    throw new IllegalStateException("Row " + count + " has " + dim + " dimensions");
                }

                if (isFraud) { labels.set(count); fraudCount++; }
                count++;
                if (count % LOG_EVERY == 0) {
                    System.out.printf("[builder] %d processed (%.1fs)%n",
                            count, (System.currentTimeMillis() - t0) / 1000.0);
                }
            }
        }

        System.out.printf("[builder] read %d vectors in %.1fs, computing global min/max%n",
                count, (System.currentTimeMillis() - t0) / 1000.0);

        // ---- Pass 2: compute GLOBAL min/max (same scale for all dims) ----
        //
        // We initially designed this with per-dimension min/max to give each
        // dim full int8 resolution. The math works out to a per-dim *weighted*
        // L2 distance — and our data has a sentinel dim (range = 2) alongside
        // booleans (range = 1), so the weights came out to a ~4x ratio. The
        // parity test (int8 vs float32 ground truth) crashed to 14% top-5
        // agreement, way under our 95% target.
        //
        // Switching to a global min/max means every dim shares the same
        // quantization step, so int8 distance is just a constant scaling of
        // the float32 distance — ordering is preserved up to rounding noise.
        // We still pay a precision cost: dims in [0, 1] only use ~half the
        // int8 range when the global range is [-1, 1] (because of sentinels).
        // Half resolution = 128 distinct values per dim — empirically enough
        // for our case (verified by the parity test, now passing >99%).
        float globalMin = Float.POSITIVE_INFINITY;
        float globalMax = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < count * DIMS; i++) {
            float v = vectorsF[i];
            if (v < globalMin) globalMin = v;
            if (v > globalMax) globalMax = v;
        }

        // The Dataset API still takes per-dim mins[]/maxs[] arrays — we just
        // populate them all with the same global values. Keeps the storage
        // format flexible if we ever revisit per-dim later.
        float[] mins = new float[DIMS];
        float[] maxs = new float[DIMS];
        Arrays.fill(mins, globalMin);
        Arrays.fill(maxs, globalMax);

        System.out.printf("[builder] global range: min=%+.4f max=%+.4f step=%.6f%n",
                globalMin, globalMax, (globalMax - globalMin) / 255f);

        // ---- Pass 3: quantize float → int8 ----
        // Pre-compute scale factors so the inner loop is just one mul.
        float[] scaleFactor = new float[DIMS];
        for (int d = 0; d < DIMS; d++) {
            float range = maxs[d] - mins[d];
            scaleFactor[d] = (range > 0f) ? (255f / range) : 0f;
        }

        byte[] vectorsI8 = new byte[count * DIMS];
        for (int i = 0; i < count; i++) {
            int base = i * DIMS;
            for (int d = 0; d < DIMS; d++) {
                float v = vectorsF[base + d];
                int q = Math.round((v - mins[d]) * scaleFactor[d]) - 128;
                // Clamp defensively — covers float rounding at the edges
                if (q < -128) q = -128;
                if (q >  127) q =  127;
                vectorsI8[base + d] = (byte) q;
            }
        }

        // Free the big float[] so the writes that follow don't bloat heap.
        vectorsF = null;

        // ---- Write output files ----
        Files.write(vectorsPath, vectorsI8);

        ByteBuffer scalesBuf = ByteBuffer
                .allocate((mins.length + maxs.length) * 4)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (float m : mins) scalesBuf.putFloat(m);
        for (float m : maxs) scalesBuf.putFloat(m);
        Files.write(scalesPath, scalesBuf.array());

        // Labels — BitSet → packed bytes, little-endian-ish bit order matching
        // the v3/v2 reader (bit i = bit (i % 8) of byte (i / 8)).
        int byteCount = (count + 7) / 8;
        byte[] labelBytes = new byte[byteCount];
        for (int i = 0; i < count; i++) {
            if (labels.get(i)) {
                labelBytes[i / 8] |= (byte) (1 << (i % 8));
            }
        }
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(labelsPath.toFile()))) {
            out.write(labelBytes);
        }

        Files.writeString(metaPath, "count=" + count + "\nfraud=" + fraudCount + "\n");

        System.out.printf("[builder] OK: %d vectors, %d frauds (%.2f%%) in %.1fs%n",
                count, fraudCount, fraudCount * 100.0 / count,
                (System.currentTimeMillis() - t0) / 1000.0);
        System.out.printf("[builder] sizes: vectors-i8.bin=%d  scales.bin=%d  labels.bin=%d%n",
                Files.size(vectorsPath), Files.size(scalesPath), Files.size(labelsPath));
    }

    /* -------------------- Minimal streaming JSON tokenizer -------------------- */

    /**
     * Streaming JSON tokenizer over an InputStream. Doesn't load everything
     * into memory. Simplified version of JsonReader, adapted for incremental
     * parsing.
     */
    private static final class JsonStream {
        private final InputStream in;
        private int peeked = -2;

        JsonStream(InputStream in) { this.in = in; }

        int peek() throws IOException {
            if (peeked == -2) peeked = in.read();
            return peeked;
        }

        int read() throws IOException {
            int c = peek();
            peeked = -2;
            return c;
        }

        void expect(int c) throws IOException {
            int got = read();
            if (got != c) {
                throw new IllegalStateException("Expected '" + (char) c + "', got '"
                        + (got == -1 ? "EOF" : (char) got) + "'");
            }
        }

        void skipWs() throws IOException {
            while (true) {
                int c = peek();
                if (c == ' ' || c == '\n' || c == '\r' || c == '\t') read();
                else return;
            }
        }

        String readString() throws IOException {
            expect('"');
            StringBuilder sb = new StringBuilder(16);
            while (true) {
                int c = read();
                if (c == '"') break;
                if (c == '\\') {
                    int esc = read();
                    sb.append((char) esc); // dataset doesn't use unicode escapes
                } else {
                    sb.append((char) c);
                }
            }
            return sb.toString();
        }

        double readDouble() throws IOException {
            StringBuilder sb = new StringBuilder(16);
            while (true) {
                int c = peek();
                if (c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' || (c >= '0' && c <= '9')) {
                    sb.append((char) read());
                } else break;
            }
            return Double.parseDouble(sb.toString());
        }

        void skipValue() throws IOException {
            skipWs();
            int c = peek();
            switch (c) {
                case '"' -> readString();
                case 't', 'f', 'n' -> { while (Character.isLetter(peek())) read(); }
                case '{' -> {
                    int depth = 0;
                    do {
                        int x = read();
                        if (x == '{') depth++;
                        else if (x == '}') depth--;
                        else if (x == '"') { while (read() != '"'); }
                    } while (depth > 0);
                }
                case '[' -> {
                    int depth = 0;
                    do {
                        int x = read();
                        if (x == '[') depth++;
                        else if (x == ']') depth--;
                        else if (x == '"') { while (read() != '"'); }
                    } while (depth > 0);
                }
                default -> readDouble();
            }
        }
    }
}
