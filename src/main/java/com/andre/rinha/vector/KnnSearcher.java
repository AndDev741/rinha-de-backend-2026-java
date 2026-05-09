package com.andre.rinha.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Brute-force k-NN search over the Dataset.
 *
 * Algorithm:
 *   1. For each reference vector, compute the squared L2 distance.
 *      (Squared because sqrt is expensive and doesn't change ordering.)
 *   2. Maintain a fixed-size K=5 binary max-heap with the best candidates.
 *      The top is the "worst of the top-5" — if a new candidate beats the
 *      top, replace it and restore the heap.
 *   3. At the end, count how many of the K neighbors are fraud.
 *
 * Distance computation modes — pick at construction time:
 *   - {@link Mode#SCALAR} — plain float-by-float loop, relies on C2
 *     auto-vectorization. The v1 baseline.
 *   - {@link Mode#VECTOR} — explicit jdk.incubator.vector with FMA.
 *     Deterministic SIMD, no dependence on the JIT's mood.
 *
 * Why both coexist: A/B testing on the same binary, same JVM, same dataset.
 * Switching modes changes only the distance function — the heap, the
 * data layout, and the I/O are identical. That isolates the SIMD gain.
 *
 * Selection: read from the KNN_MODE env var in App.java. Default is VECTOR.
 */
public final class KnnSearcher {

    public static final int K = 5;

    public enum Mode { SCALAR, VECTOR }

    /**
     * SIMD lane width. SPECIES_PREFERRED picks the widest vector available
     * on the running CPU at class-load time:
     *   - AVX-512   → 16 lanes  (Skylake-X+, EPYC Zen 4)
     *   - AVX2      →  8 lanes  (Mac Mini 2014 Haswell — the rinha test box)
     *   - SSE/NEON  →  4 lanes  (older x86, ARM)
     */
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    /**
     * Last index aligned with full SIMD width — anything below this loops
     * with full lanes; anything above goes through the masked tail.
     *
     *   DIMS=14, SPECIES.length()=16 → LOOP_BOUND=0  (whole vector is tail)
     *   DIMS=14, SPECIES.length()=8  → LOOP_BOUND=8  (one full chunk + tail)
     *   DIMS=14, SPECIES.length()=4  → LOOP_BOUND=12 (three full chunks + tail)
     */
    private static final int LOOP_BOUND = SPECIES.loopBound(Dataset.DIMS);

    /**
     * Mask for the tail. True lanes correspond to real DIMS positions;
     * false lanes are padding (their contributions are zeroed out by the
     * masked-load API and produce zero in the squared-distance sum).
     */
    private static final VectorMask<Float> TAIL_MASK =
            SPECIES.indexInRange(LOOP_BOUND, Dataset.DIMS);

    private final Dataset dataset;
    private final Mode mode;

    // Reusable buffers per search to avoid hot-path allocation.
    // WARNING: KnnSearcher is stateful per instance — create one per thread.
    private final float[] heapDist = new float[K];
    private final int[]   heapIdx  = new int[K];

    public KnnSearcher(Dataset dataset, Mode mode) {
        this.dataset = dataset;
        this.mode = mode;
    }

    public Mode mode() { return mode; }

    /** Convenience: log line the App can print at startup. */
    public static String simdInfo() {
        return SPECIES + " (" + SPECIES.length() + " lanes, "
                + SPECIES.vectorBitSize() + "-bit, LOOP_BOUND=" + LOOP_BOUND + ")";
    }

    /**
     * Computes fraud_score = (#frauds among K nearest neighbors) / K.
     *
     * Branches on mode ONCE (per request, outside the 3M-iteration loop) so
     * the JIT can compile each path without speculative dispatch overhead.
     */
    public float fraudScore(float[] query) {
        return mode == Mode.VECTOR ? fraudScoreVector(query) : fraudScoreScalar(query);
    }

    /* =================================================================== */
    /*  SCALAR mode (v1 baseline, kept for A/B comparison)                  */
    /* =================================================================== */

    private float fraudScoreScalar(float[] query) {
        final float[] vecs = dataset.vectors();
        final int count = dataset.count();
        final int dims = Dataset.DIMS;

        for (int i = 0; i < K; i++) {
            heapDist[i] = squaredDistanceScalar(query, vecs, i * dims, dims);
            heapIdx[i] = i;
        }
        for (int i = K / 2 - 1; i >= 0; i--) siftDown(i);

        for (int i = K; i < count; i++) {
            int offset = i * dims;
            float topDist = heapDist[0];
            float d = squaredDistanceScalarBounded(query, vecs, offset, dims, topDist);
            if (d < topDist) {
                heapDist[0] = d;
                heapIdx[0] = i;
                siftDown(0);
            }
        }

        return countFrauds() / (float) K;
    }

    private static float squaredDistanceScalar(float[] q, float[] vecs, int off, int dims) {
        float sum = 0f;
        for (int i = 0; i < dims; i++) {
            float d = q[i] - vecs[off + i];
            sum += d * d;
        }
        return sum;
    }

    /**
     * Scalar with mid-loop bound check. The branch breaks auto-vectorization,
     * but with only 14 dims the early-exit savings on far candidates outweigh
     * the lost SIMD on near ones. v1 measured this empirically.
     */
    private static float squaredDistanceScalarBounded(float[] q, float[] vecs, int off, int dims, float bound) {
        float sum = 0f;
        int half = dims >>> 1;
        for (int i = 0; i < half; i++) {
            float d = q[i] - vecs[off + i];
            sum += d * d;
        }
        if (sum >= bound) return Float.POSITIVE_INFINITY;
        for (int i = half; i < dims; i++) {
            float d = q[i] - vecs[off + i];
            sum += d * d;
        }
        return sum;
    }

    /* =================================================================== */
    /*  VECTOR mode (jdk.incubator.vector, explicit SIMD)                   */
    /* =================================================================== */

    private float fraudScoreVector(float[] query) {
        final float[] vecs = dataset.vectors();
        final int count = dataset.count();

        // Pre-load the query exactly once for the whole search.
        // qFull is null when the SIMD width is wider than DIMS (AVX-512 case).
        final FloatVector qFull = LOOP_BOUND > 0
                ? FloatVector.fromArray(SPECIES, query, 0)
                : null;
        // qTail always exists — it covers what's left after LOOP_BOUND, with
        // padding lanes masked out. Loaded with a mask so the read can extend
        // past the array end safely (masked-out lanes are not actually read).
        final FloatVector qTail = FloatVector.fromArray(
                SPECIES, query, LOOP_BOUND, TAIL_MASK);

        // Heap initialization — first K candidates.
        for (int i = 0; i < K; i++) {
            heapDist[i] = squaredDistanceVector(qFull, qTail, vecs, i * Dataset.DIMS);
            heapIdx[i] = i;
        }
        for (int i = K / 2 - 1; i >= 0; i--) siftDown(i);

        // Hot loop — 3M iterations. No early-exit branch here: the conditional
        // would inhibit speculative execution of the SIMD chunk.
        for (int i = K; i < count; i++) {
            int offset = i * Dataset.DIMS;
            float d = squaredDistanceVector(qFull, qTail, vecs, offset);
            if (d < heapDist[0]) {
                heapDist[0] = d;
                heapIdx[0] = i;
                siftDown(0);
            }
        }

        return countFrauds() / (float) K;
    }

    /**
     * Squared L2 distance using explicit SIMD.
     *
     * Layout for DIMS=14 with AVX2 (8-lane species):
     *   chunk 0 — full width:  diff = qFull - vecs[off..off+7]
     *   tail    — masked:      diff = qTail - vecs[off+8..off+13] (masked)
     *
     * FMA (fused multiply-add) is used everywhere: `diff.fma(diff, sum)` is
     * `sum + diff*diff` in one instruction with one rounding step. Faster
     * than `mul → add` AND more numerically precise.
     */
    private static float squaredDistanceVector(FloatVector qFull, FloatVector qTail,
                                                float[] vecs, int off) {
        FloatVector sum = FloatVector.zero(SPECIES);

        // Full-width chunks. For DIMS=14 with AVX2 this iterates exactly once.
        // The for(...) header constants let the JIT unroll completely.
        for (int i = 0; i < LOOP_BOUND; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromArray(SPECIES, vecs, off + i);
            // qFull is non-null whenever LOOP_BOUND > 0 (this branch is reached).
            FloatVector diff = qFull.sub(v);
            sum = diff.fma(diff, sum);
        }

        // Masked tail — covers DIMS not aligned with the SIMD width.
        // Padding lanes contribute 0 to the sum, so they don't affect the result.
        FloatVector vTail = FloatVector.fromArray(SPECIES, vecs, off + LOOP_BOUND, TAIL_MASK);
        FloatVector diffTail = qTail.sub(vTail);
        sum = diffTail.fma(diffTail, sum);

        // Horizontal reduction — sum all lanes into a single float.
        return sum.reduceLanes(VectorOperators.ADD);
    }

    /* =================================================================== */
    /*  Shared utilities                                                    */
    /* =================================================================== */

    private int countFrauds() {
        int n = 0;
        for (int i = 0; i < K; i++) {
            if (dataset.isFraud(heapIdx[i])) n++;
        }
        return n;
    }

    /**
     * Classic max-heap sift-down: pushes the element at position `i` downward
     * until the heap property is restored. Position 0 holds the largest.
     */
    private void siftDown(int i) {
        final int n = K;
        while (true) {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int largest = i;
            if (left  < n && heapDist[left]  > heapDist[largest]) largest = left;
            if (right < n && heapDist[right] > heapDist[largest]) largest = right;
            if (largest == i) return;
            float td = heapDist[i]; heapDist[i] = heapDist[largest]; heapDist[largest] = td;
            int   ti = heapIdx[i];  heapIdx[i]  = heapIdx[largest];  heapIdx[largest]  = ti;
            i = largest;
        }
    }
}
