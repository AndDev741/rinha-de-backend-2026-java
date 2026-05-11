package com.andre.rinha.vector;

/**
 * v7 IVF k-NN with bounding-box repair — exact top-5 with aggressive pruning.
 *
 * Algorithm per request:
 *   1. Quantize the float query into short[14] (× 10000 scaling).
 *   2. Compute squared int distance from query to all K centroids.
 *      Pick the single nearest cluster.
 *   3. Scan all vectors in that cluster (scalar manual loop with early-exit
 *      per dimension), maintaining top-5 in five long fields.
 *   4. For each OTHER cluster, compute the lower-bound distance from the
 *      query to the cluster's axis-aligned bounding box. If the bound
 *      exceeds the current top-5 worst, the whole cluster is provably
 *      uninteresting — skip it. Otherwise scan it and update top-5.
 *   5. Count frauds among the top-5.
 *
 * Why this is fast AND exact:
 *   - With well-clustered data and a tight initial top-5 from the closest
 *     cluster, the bbox check rejects most other clusters in constant time
 *     (one accumulating loop over 14 dims with a short-circuit on overshoot).
 *   - When a cluster is scanned, the inner per-vector loop is 14 scalar int
 *     subtractions+squares with `if (dist > worst) continue` between dims —
 *     the JIT/C2 inlines and (where possible) auto-vectorizes this. AOT
 *     (GraalVM) also handles scalar shorts well.
 *   - No approximation: same top-5 as float32 brute force, up to int16
 *     rounding (~0.0001 per dim, negligible).
 *
 * Why no Vector API:
 *   v6 demonstrated that explicit jdk.incubator.vector ops (convertShape,
 *   fma) don't compile well under GraalVM native-image, and even under JIT
 *   they tie or lose to a tight scalar loop for 14 dims with early-exit.
 *   Scalar wins for THIS problem shape (small DIMS, big speedup possible
 *   from per-dim short-circuit).
 *
 * Top-5 representation:
 *   Five long distances (d0..d4) and five int positions (pos0..pos4)
 *   stored as plain fields. No heap, no array. The JIT keeps them in
 *   registers across the hot loop, and the `add()` cascade does only as
 *   much shifting as needed — usually replacing just one or two slots.
 */
public final class KnnSearcher {

    public static final int K = 5;

    private final Dataset dataset;

    // Reusable per-request quantized query buffer.
    private final short[] qBytes = new short[Dataset.DIMS];

    // Top-5 slot state (inlined; rebuilt each fraudScore call).
    private long d0, d1, d2, d3, d4;
    private int  pos0, pos1, pos2, pos3, pos4;

    public KnnSearcher(Dataset dataset) {
        this.dataset = dataset;
    }

    public static String simdInfo() {
        return "scalar int16 (no Vector API), bbox-repair IVF";
    }

    /** Computes fraud_score = (#frauds in top-5) / 5. */
    public float fraudScore(float[] query) {
        // 1. Quantize the query.
        Dataset.quantize(query, qBytes);

        // 2. Find the nearest cluster via centroid distance.
        final short[] centroids = dataset.centroids();
        final int kClusters = dataset.k();
        int chosen = 0;
        long bestD = Long.MAX_VALUE;
        for (int c = 0; c < kClusters; c++) {
            long d = centroidDistance(centroids, c);
            if (d < bestD) {
                bestD = d;
                chosen = c;
            }
        }

        // 3. Reset top-5 and scan the chosen cluster.
        resetTop();
        scanCluster(chosen);

        // 4. Bbox repair pass: visit every OTHER cluster and prune via
        //    lower-bound distance. Empty clusters have a huge bbox (we
        //    initialized them with MAX/MIN at build time) so they auto-skip.
        final short[] bbMin = dataset.bboxMin();
        final short[] bbMax = dataset.bboxMax();
        for (int c = 0; c < kClusters; c++) {
            if (c == chosen) continue;
            long worst = d4;
            if (bboxMayBeat(bbMin, bbMax, c, worst)) {
                scanCluster(c);
            }
        }

        // 5. Count frauds.
        return countFrauds() / (float) K;
    }

    /* =================================================================== *
     * Centroid distance — scalar manual unrolled                           *
     * =================================================================== */

    private long centroidDistance(short[] centroids, int c) {
        final int base = c * Dataset.DIMS;
        final short[] q = qBytes;
        long s = 0;
        int x;

        x = centroids[base     ] - q[0];  s += (long) x * x;
        x = centroids[base +  1] - q[1];  s += (long) x * x;
        x = centroids[base +  2] - q[2];  s += (long) x * x;
        x = centroids[base +  3] - q[3];  s += (long) x * x;
        x = centroids[base +  4] - q[4];  s += (long) x * x;
        x = centroids[base +  5] - q[5];  s += (long) x * x;
        x = centroids[base +  6] - q[6];  s += (long) x * x;
        x = centroids[base +  7] - q[7];  s += (long) x * x;
        x = centroids[base +  8] - q[8];  s += (long) x * x;
        x = centroids[base +  9] - q[9];  s += (long) x * x;
        x = centroids[base + 10] - q[10]; s += (long) x * x;
        x = centroids[base + 11] - q[11]; s += (long) x * x;
        x = centroids[base + 12] - q[12]; s += (long) x * x;
        x = centroids[base + 13] - q[13]; s += (long) x * x;
        return s;
    }

    /* =================================================================== *
     * Bbox lower-bound — exact pruning                                     *
     *                                                                       *
     * For each dim d, the minimum possible squared distance from q[d] to    *
     * ANY value in [bbMin[c,d], bbMax[c,d]] is:                             *
     *     0                       if  bbMin[c,d] <= q[d] <= bbMax[c,d]      *
     *     (bbMin[c,d] - q[d])^2   if  q[d] <  bbMin[c,d]                    *
     *     (q[d] - bbMax[c,d])^2   if  q[d] >  bbMax[c,d]                    *
     *                                                                       *
     * Summing those per-dim minima gives a STRICT lower bound on the        *
     * distance from q to any vector in cluster c. If that already exceeds   *
     * the current top-5 worst, the whole cluster can be skipped.            *
     * =================================================================== */

    private boolean bboxMayBeat(short[] bbMin, short[] bbMax, int c, long worst) {
        final int base = c * Dataset.DIMS;
        final short[] q = qBytes;
        long s = 0;
        int v, diff;

        v = q[0];  diff = v < bbMin[base     ] ? bbMin[base     ] - v : (v > bbMax[base     ] ? v - bbMax[base     ] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[1];  diff = v < bbMin[base +  1] ? bbMin[base +  1] - v : (v > bbMax[base +  1] ? v - bbMax[base +  1] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[2];  diff = v < bbMin[base +  2] ? bbMin[base +  2] - v : (v > bbMax[base +  2] ? v - bbMax[base +  2] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[3];  diff = v < bbMin[base +  3] ? bbMin[base +  3] - v : (v > bbMax[base +  3] ? v - bbMax[base +  3] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[4];  diff = v < bbMin[base +  4] ? bbMin[base +  4] - v : (v > bbMax[base +  4] ? v - bbMax[base +  4] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[5];  diff = v < bbMin[base +  5] ? bbMin[base +  5] - v : (v > bbMax[base +  5] ? v - bbMax[base +  5] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[6];  diff = v < bbMin[base +  6] ? bbMin[base +  6] - v : (v > bbMax[base +  6] ? v - bbMax[base +  6] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[7];  diff = v < bbMin[base +  7] ? bbMin[base +  7] - v : (v > bbMax[base +  7] ? v - bbMax[base +  7] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[8];  diff = v < bbMin[base +  8] ? bbMin[base +  8] - v : (v > bbMax[base +  8] ? v - bbMax[base +  8] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[9];  diff = v < bbMin[base +  9] ? bbMin[base +  9] - v : (v > bbMax[base +  9] ? v - bbMax[base +  9] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[10]; diff = v < bbMin[base + 10] ? bbMin[base + 10] - v : (v > bbMax[base + 10] ? v - bbMax[base + 10] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[11]; diff = v < bbMin[base + 11] ? bbMin[base + 11] - v : (v > bbMax[base + 11] ? v - bbMax[base + 11] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[12]; diff = v < bbMin[base + 12] ? bbMin[base + 12] - v : (v > bbMax[base + 12] ? v - bbMax[base + 12] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[13]; diff = v < bbMin[base + 13] ? bbMin[base + 13] - v : (v > bbMax[base + 13] ? v - bbMax[base + 13] : 0); s += (long) diff * diff;
        return s <= worst;
    }

    /* =================================================================== *
     * Cluster scan — scalar manual unrolled, early-exit per dim            *
     *                                                                       *
     * The JIT inlines this and (in many runs) auto-vectorizes the first    *
     * few dims before the first short-circuit. Even when it doesn't, the   *
     * per-dim continue keeps far candidates from paying the full 14-dim    *
     * cost. With well-clustered data, most loop iterations terminate at    *
     * dim 2-4.                                                              *
     * =================================================================== */

    private void scanCluster(int c) {
        final short[] vecs = dataset.vectors();
        final short[] q = qBytes;
        final int start = dataset.clusterStart(c);
        final int end   = dataset.clusterEnd(c);

        final int q0  = q[0],  q1  = q[1],  q2  = q[2],  q3  = q[3],
                  q4  = q[4],  q5  = q[5],  q6  = q[6],  q7  = q[7],
                  q8  = q[8],  q9  = q[9],  q10 = q[10], q11 = q[11],
                  q12 = q[12], q13 = q[13];

        long worst = d4;

        for (int i = start; i < end; i++) {
            int b = i * Dataset.DIMS;
            long dist;
            int x;

            x = vecs[b     ] - q0;  dist = (long) x * x;       if (dist > worst) continue;
            x = vecs[b +  1] - q1;  dist += (long) x * x;       if (dist > worst) continue;
            x = vecs[b +  2] - q2;  dist += (long) x * x;       if (dist > worst) continue;
            x = vecs[b +  3] - q3;  dist += (long) x * x;       if (dist > worst) continue;
            x = vecs[b +  4] - q4;  dist += (long) x * x;       if (dist > worst) continue;
            x = vecs[b +  5] - q5;  dist += (long) x * x;       if (dist > worst) continue;
            x = vecs[b +  6] - q6;  dist += (long) x * x;       if (dist > worst) continue;
            x = vecs[b +  7] - q7;  dist += (long) x * x;       if (dist > worst) continue;
            x = vecs[b +  8] - q8;  dist += (long) x * x;       if (dist > worst) continue;
            x = vecs[b +  9] - q9;  dist += (long) x * x;       if (dist > worst) continue;
            x = vecs[b + 10] - q10; dist += (long) x * x;       if (dist > worst) continue;
            x = vecs[b + 11] - q11; dist += (long) x * x;       if (dist > worst) continue;
            x = vecs[b + 12] - q12; dist += (long) x * x;       if (dist > worst) continue;
            x = vecs[b + 13] - q13; dist += (long) x * x;       if (dist > worst) continue;

            insertTop(dist, i);
            worst = d4;
        }
    }

    /* =================================================================== *
     * Top-5 maintenance — five long/int fields, manual cascade             *
     *                                                                       *
     * Invariant: d0 <= d1 <= d2 <= d3 <= d4. d4 is the worst-of-top-5      *
     * (the "kick out" candidate).                                           *
     * =================================================================== */

    private void resetTop() {
        d0 = d1 = d2 = d3 = d4 = Long.MAX_VALUE;
        pos0 = pos1 = pos2 = pos3 = pos4 = -1;
    }

    private void insertTop(long dist, int pos) {
        // dist is strictly less than d4 (caller already checked).
        // Cascade insert in sorted order.
        if (dist < d0) {
            d4 = d3; pos4 = pos3;
            d3 = d2; pos3 = pos2;
            d2 = d1; pos2 = pos1;
            d1 = d0; pos1 = pos0;
            d0 = dist; pos0 = pos;
        } else if (dist < d1) {
            d4 = d3; pos4 = pos3;
            d3 = d2; pos3 = pos2;
            d2 = d1; pos2 = pos1;
            d1 = dist; pos1 = pos;
        } else if (dist < d2) {
            d4 = d3; pos4 = pos3;
            d3 = d2; pos3 = pos2;
            d2 = dist; pos2 = pos;
        } else if (dist < d3) {
            d4 = d3; pos4 = pos3;
            d3 = dist; pos3 = pos;
        } else {
            d4 = dist; pos4 = pos;
        }
    }

    private int countFrauds() {
        int n = 0;
        if (pos0 >= 0 && dataset.isFraud(pos0)) n++;
        if (pos1 >= 0 && dataset.isFraud(pos1)) n++;
        if (pos2 >= 0 && dataset.isFraud(pos2)) n++;
        if (pos3 >= 0 && dataset.isFraud(pos3)) n++;
        if (pos4 >= 0 && dataset.isFraud(pos4)) n++;
        return n;
    }
}
