package com.andre.rinha.prep;

import com.andre.rinha.vector.KMeans;

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
 * v7: converts references.json.gz into an IVF index with int16 storage and
 * per-cluster bounding boxes for exact k-NN search.
 *
 * Output (5 files in the target dir):
 *
 *   vectors-i16.bin       →  N × 14 × 2 bytes (int16 LE, scaled by 10000)
 *                            = ~84 MB for N = 3,000,000
 *                            VECTORS ARE REORDERED so cluster c occupies
 *                            offsets[c]..offsets[c+1].
 *   centroids-i16.bin     →  K × 14 × 2 bytes (int16 LE, same scaling)
 *                            = ~7 KB for K = 256
 *   bbox.bin              →  K × 14 × 2 × 2 bytes (min[K][14] then max[K][14])
 *                            = ~14 KB for K = 256
 *   cluster_offsets.bin   →  K+1 × 4 bytes (int32 LE)
 *   labels.bin            →  ceil(N/8) bytes (bit i = 1 if vector i is fraud)
 *
 * Why int16 with scale 10000 (vs v4-v6's int8 with min/max scaling):
 *
 *   Source data is in [-1, 1] with the 5,6 sentinel dims hitting exactly -1.
 *   Multiplying by 10000 maps to [-10000, 10000], which fits comfortably in
 *   int16 (±32767). That preserves ~4 decimal digits of precision —
 *   effectively LOSSLESS vs the original float values for our purposes.
 *
 *   Quantization step: 1/10000 = 0.0001. Compare to v4-v6's int8 step of
 *   1/127 ≈ 0.008 — almost two orders of magnitude tighter. The cost is
 *   2× memory (84 MB vs 42 MB), still well under the 350 MB rinha cap.
 *
 *   The bonus: distance in int16 space is *exactly* the float distance × 1e8.
 *   No per-dim weighting headaches. Same ordering as float32 brute force.
 *
 * Why per-cluster bounding boxes:
 *
 *   With BB pruning, v7's KnnSearcher can use nprobe=1 + exact repair: it
 *   scans the single closest cluster first, then for each OTHER cluster
 *   computes a lower-bound distance from the query to the cluster's
 *   bounding box. If that lower bound is already worse than the current
 *   top-5 worst, the cluster is provably uninteresting and gets skipped.
 *   This recovers exact k-NN with most clusters never scanned.
 *
 * How to run:
 *   java -cp target/classes com.andre.rinha.prep.DatasetBuilder \
 *        references.json.gz ./data
 *
 * Build-time peak heap: ~168 MB. Run with -Xmx384m or larger.
 */
public final class DatasetBuilder {

    private static final int DIMS = 14;
    private static final int LOG_EVERY = 1_000_000;

    /** Quantization scale: float × SCALE → int16. */
    private static final int SCALE = 10_000;

    /** Number of clusters for IVF coarse partitioning. */
    private static final int K_CLUSTERS = 256;
    /** Max iterations for Lloyd's k-means. */
    private static final int KMEANS_MAX_ITERS = 20;
    /** Deterministic seed so the dataset build is reproducible. */
    private static final long KMEANS_SEED = 42L;

    private static final int INITIAL_CAPACITY = 3_000_000;

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: DatasetBuilder <references.json.gz> <outputDir>");
            System.exit(1);
        }
        Path input = Path.of(args[0]);
        Path outDir = Path.of(args[1]);
        outDir.toFile().mkdirs();

        Path vectorsPath   = outDir.resolve("vectors-i16.bin");
        Path centroidsPath = outDir.resolve("centroids-i16.bin");
        Path bboxPath      = outDir.resolve("bbox.bin");
        Path offsetsPath   = outDir.resolve("cluster_offsets.bin");
        Path labelsPath    = outDir.resolve("labels.bin");
        Path metaPath      = outDir.resolve("meta.txt");

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

                js.expect('{');
                boolean isFraud = false;
                int dim = 0;

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

        System.out.printf("[builder] read %d vectors in %.1fs%n",
                count, (System.currentTimeMillis() - t0) / 1000.0);

        // ---- Pass 2: quantize float → int16 (× SCALE) ----
        //
        // Source data is in [-1, 1]. After × 10000 it lands in [-10000, 10000],
        // safely under short max ±32767. We still clamp defensively just in
        // case some input has rounding overshoot.
        short[] vectorsI16 = new short[count * DIMS];
        for (int i = 0; i < count; i++) {
            int base = i * DIMS;
            for (int d = 0; d < DIMS; d++) {
                int q = Math.round(vectorsF[base + d] * SCALE);
                if (q < Short.MIN_VALUE) q = Short.MIN_VALUE;
                if (q > Short.MAX_VALUE) q = Short.MAX_VALUE;
                vectorsI16[base + d] = (short) q;
            }
        }

        // ---- Pass 3: k-means clustering (still uses float vectors) ----
        //
        // We cluster on the FLOAT vectors because clustering quality matters
        // and float gives the cleanest centroids. The int16 vectors we just
        // built will be reordered by cluster after.
        System.out.printf("%n[builder] running k-means: K=%d, max_iters=%d, seed=%d%n",
                K_CLUSTERS, KMEANS_MAX_ITERS, KMEANS_SEED);
        long tKmeans = System.currentTimeMillis();
        KMeans.Result km = KMeans.fit(vectorsF, count, DIMS, K_CLUSTERS,
                                       KMEANS_MAX_ITERS, KMEANS_SEED);
        System.out.printf("[builder] k-means done in %.1fs (%d iterations)%n",
                (System.currentTimeMillis() - tKmeans) / 1000.0, km.iterations());

        vectorsF = null;  // free the big float buffer

        // ---- Pass 4: reorder vectors and labels by cluster ----
        int[] assignments = km.assignments();
        int[] counts = new int[K_CLUSTERS];
        for (int a : assignments) counts[a]++;
        int[] offsets = new int[K_CLUSTERS + 1];
        for (int c = 0; c < K_CLUSTERS; c++) offsets[c + 1] = offsets[c] + counts[c];

        int minCluster = Integer.MAX_VALUE, maxCluster = 0, emptyCount = 0;
        for (int n : counts) {
            if (n == 0) emptyCount++;
            if (n < minCluster) minCluster = n;
            if (n > maxCluster) maxCluster = n;
        }
        System.out.printf("[builder] cluster size: min=%d max=%d mean=%d empty=%d%n",
                minCluster, maxCluster, count / K_CLUSTERS, emptyCount);

        short[] reorderedVecs = new short[count * DIMS];
        BitSet reorderedLabels = new BitSet(count);
        int[] writePos = offsets.clone();
        for (int oldIdx = 0; oldIdx < count; oldIdx++) {
            int c = assignments[oldIdx];
            int newIdx = writePos[c]++;
            System.arraycopy(vectorsI16, oldIdx * DIMS, reorderedVecs, newIdx * DIMS, DIMS);
            if (labels.get(oldIdx)) reorderedLabels.set(newIdx);
        }
        vectorsI16 = null;
        labels = null;

        // ---- Pass 5: compute per-cluster bounding boxes (in int16 space) ----
        //
        // For each cluster c and each dim d: bbox_min[c][d] = min over all
        // vectors in cluster c of vector[d]. Same for max. These tell us the
        // tightest axis-aligned box that contains every vector in cluster c.
        //
        // KnnSearcher uses these to compute a lower-bound distance from the
        // query to the cluster. If that lower bound exceeds the current top-5
        // worst, we can skip the whole cluster with mathematical certainty.
        short[] bboxMin = new short[K_CLUSTERS * DIMS];
        short[] bboxMax = new short[K_CLUSTERS * DIMS];
        for (int c = 0; c < K_CLUSTERS; c++) {
            int base = c * DIMS;
            for (int d = 0; d < DIMS; d++) {
                bboxMin[base + d] = Short.MAX_VALUE;
                bboxMax[base + d] = Short.MIN_VALUE;
            }
            int start = offsets[c];
            int end = offsets[c + 1];
            for (int i = start; i < end; i++) {
                int vBase = i * DIMS;
                for (int d = 0; d < DIMS; d++) {
                    short v = reorderedVecs[vBase + d];
                    if (v < bboxMin[base + d]) bboxMin[base + d] = v;
                    if (v > bboxMax[base + d]) bboxMax[base + d] = v;
                }
            }
            // For empty clusters, leave bbox at MAX/MIN so the lower-bound
            // distance becomes huge → search auto-skips them.
        }

        // ---- Convert centroids to int16 (same scale as data) ----
        float[] centroidsF = km.centroids();
        short[] centroidsI16 = new short[centroidsF.length];
        for (int i = 0; i < centroidsF.length; i++) {
            int q = Math.round(centroidsF[i] * SCALE);
            if (q < Short.MIN_VALUE) q = Short.MIN_VALUE;
            if (q > Short.MAX_VALUE) q = Short.MAX_VALUE;
            centroidsI16[i] = (short) q;
        }

        // ---- Write output files ----
        Files.write(vectorsPath, shortsToBytes(reorderedVecs));
        Files.write(centroidsPath, shortsToBytes(centroidsI16));

        // bbox.bin: mins first (K*DIMS shorts), then maxs (K*DIMS shorts).
        short[] bboxConcat = new short[bboxMin.length + bboxMax.length];
        System.arraycopy(bboxMin, 0, bboxConcat, 0, bboxMin.length);
        System.arraycopy(bboxMax, 0, bboxConcat, bboxMin.length, bboxMax.length);
        Files.write(bboxPath, shortsToBytes(bboxConcat));

        ByteBuffer offBuf = ByteBuffer.allocate((K_CLUSTERS + 1) * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int o : offsets) offBuf.putInt(o);
        Files.write(offsetsPath, offBuf.array());

        int byteCount = (count + 7) / 8;
        byte[] labelBytes = new byte[byteCount];
        for (int i = 0; i < count; i++) {
            if (reorderedLabels.get(i)) {
                labelBytes[i / 8] |= (byte) (1 << (i % 8));
            }
        }
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(labelsPath.toFile()))) {
            out.write(labelBytes);
        }

        Files.writeString(metaPath,
                "count=" + count + "\n"
              + "fraud=" + fraudCount + "\n"
              + "scale=" + SCALE + "\n"
              + "k=" + K_CLUSTERS + "\n"
              + "kmeans_iters=" + km.iterations() + "\n"
              + "cluster_min=" + minCluster + "\n"
              + "cluster_max=" + maxCluster + "\n");

        System.out.printf("%n[builder] OK: %d vectors, %d frauds (%.2f%%) in %.1fs%n",
                count, fraudCount, fraudCount * 100.0 / count,
                (System.currentTimeMillis() - t0) / 1000.0);
        System.out.printf("[builder] sizes: vectors=%d  centroids=%d  bbox=%d  offsets=%d  labels=%d%n",
                Files.size(vectorsPath), Files.size(centroidsPath),
                Files.size(bboxPath), Files.size(offsetsPath), Files.size(labelsPath));
    }

    /** Pack short[] into little-endian byte[] for file output. */
    private static byte[] shortsToBytes(short[] src) {
        ByteBuffer bb = ByteBuffer.allocate(src.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (short s : src) bb.putShort(s);
        return bb.array();
    }

    /* -------------------- Minimal streaming JSON tokenizer -------------------- */

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
                    sb.append((char) esc);
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
