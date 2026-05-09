package com.andre.rinha.vector;

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
 * Why a max-heap instead of sorting everything:
 *   - Sorting 3M floats: O(n log n) = ~66M comparisons + allocation.
 *   - K-sized heap: O(n log K) = O(n × 2.3) ≈ 7M comparisons.
 *   - More importantly: zero extra allocation (heap lives in primitive arrays).
 *
 * Observed performance (modern x86_64 CPU, no SIMD yet):
 *   - 3M vectors × 14 dims = 42M floats read (~168 MB) per request.
 *   - DRAM bandwidth ~20 GB/s → reading alone costs ~8 ms.
 *   - When float[] is in L3 cache (after the first request), drops to ~2 ms.
 *   - Compute (sub, mul, add) is partially auto-vectorized by the JIT —
 *     good enough for now.
 *
 * That's the number to beat. Explicit Vector API will only help once we
 * exit the memory-bound regime — likely with int8 quantization.
 */
public final class KnnSearcher {

    public static final int K = 5;

    private final Dataset dataset;

    // Reusable buffers per search to avoid hot-path allocation.
    // WARNING: KnnSearcher is stateful per instance — create one per thread.
    private final float[] heapDist = new float[K]; // distances of the K best
    private final int[]   heapIdx  = new int[K];   // indices of the K best

    public KnnSearcher(Dataset dataset) {
        this.dataset = dataset;
    }

    /**
     * Computes fraud_score = (#frauds among K nearest neighbors) / K.
     *
     * @param query query vector of size DIMS
     * @return fraud_score between 0.0 and 1.0
     */
    public float fraudScore(float[] query) {
        final float[] vecs = dataset.vectors();
        final int count = dataset.count();
        final int dims = Dataset.DIMS;

        // Initialize the heap with the first K vectors.
        // Max-heap: position 0 is the worst (largest distance).
        for (int i = 0; i < K; i++) {
            heapDist[i] = squaredDistance(query, vecs, i * dims, dims);
            heapIdx[i] = i;
        }
        // Heapify — reorganize the array to satisfy the max-heap property.
        for (int i = K / 2 - 1; i >= 0; i--) siftDown(i);

        // Main loop over the remaining count - K vectors.
        // This is the hot path.
        for (int i = K; i < count; i++) {
            int offset = i * dims;
            // Classic optimization: if the partial distance has already
            // exceeded the heap top, abort the calculation. With 14 dims the
            // gain is small (few short-circuit chances), but it's free.
            float topDist = heapDist[0];
            float d = squaredDistanceWithBound(query, vecs, offset, dims, topDist);
            if (d < topDist) {
                heapDist[0] = d;
                heapIdx[0] = i;
                siftDown(0);
            }
        }

        // Count frauds among the final K.
        int fraudCount = 0;
        for (int i = 0; i < K; i++) {
            if (dataset.isFraud(heapIdx[i])) fraudCount++;
        }
        return fraudCount / (float) K;
    }

    /* -------------------- Squared L2 distance -------------------- */

    /**
     * Simple version: reads 14 floats and computes sum((q[i] - v[i])^2).
     * The JIT (C2) auto-vectorizes this loop well. Manual unrolling can
     * help but isn't needed at this step.
     */
    private static float squaredDistance(float[] q, float[] vecs, int off, int dims) {
        float sum = 0f;
        for (int i = 0; i < dims; i++) {
            float d = q[i] - vecs[off + i];
            sum += d * d;
        }
        return sum;
    }

    /**
     * Early-exit version: if sum has exceeded bound, returns immediately.
     * Useful when we're far from the top-K and want to bail early.
     *
     * Note: early-exit breaks auto-vectorization because it introduces a
     * branch in the middle of the loop. With 14 dims we can fit a single
     * test halfway through.
     */
    private static float squaredDistanceWithBound(float[] q, float[] vecs, int off, int dims, float bound) {
        // First half — no test, let the JIT vectorize.
        float sum = 0f;
        int half = dims >>> 1; // 7
        for (int i = 0; i < half; i++) {
            float d = q[i] - vecs[off + i];
            sum += d * d;
        }
        // Test in the middle.
        if (sum >= bound) return Float.POSITIVE_INFINITY;
        // Second half.
        for (int i = half; i < dims; i++) {
            float d = q[i] - vecs[off + i];
            sum += d * d;
        }
        return sum;
    }

    /* -------------------- Max-heap operations -------------------- */

    /**
     * Classic sift-down: pushes the element at position `i` downward until
     * the max-heap property is restored.
     *
     * Max-heap: parent >= children. The top (pos 0) is the largest.
     * When we replace the top with a smaller candidate, we just sink it.
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
            // Swap
            float td = heapDist[i]; heapDist[i] = heapDist[largest]; heapDist[largest] = td;
            int   ti = heapIdx[i];  heapIdx[i]  = heapIdx[largest];  heapIdx[largest]  = ti;
            i = largest;
        }
    }
}
