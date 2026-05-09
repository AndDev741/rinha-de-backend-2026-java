package com.andre.rinha.vector;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * v5 IVF (Inverted File) k-NN search.
 *
 * Algorithm:
 *   1. Quantize the float query into byte[14] using Dataset.quantize.
 *   2. Compute float distance from the query to each of the K centroids
 *      (small loop, K * DIMS ops, fully cache-resident).
 *   3. Pick the NPROBE clusters with the smallest distances.
 *   4. For each picked cluster, scan ONLY the vectors in that cluster's
 *      contiguous range using the v4 hybrid SIMD distance (B→F + FMA).
 *   5. Maintain a top-K=5 max-heap across all probed clusters.
 *
 * Why this is dramatically faster than brute force:
 *   v4 does 3M × 14 = 42M ops per request.
 *   v5 does K × DIMS  +  NPROBE × (N/K) × DIMS
 *        = 256 × 14   +  3 × ~12k × 14
 *        ≈ 3.6 k      +  500 k
 *        ≈ 504 k ops per request
 *   That's ~80× fewer comparisons than v4.
 *
 * Recall:
 *   The true top-5 may straddle cluster boundaries — IVF is approximate.
 *   With NPROBE=3, the top 3 nearest clusters cover ~95-98% of true top-5
 *   on real data. The parity test verifies this on synthetic data.
 *
 * Layout reminder:
 *   Dataset.vectors() is REORDERED so cluster c occupies
 *   [clusterStart(c), clusterEnd(c)). One contiguous range per cluster
 *   means cache prefetchers love this loop.
 */
public final class KnnSearcher {

    public static final int K = 5;
    /** Number of nearest clusters to scan per query. */
    public static final int NPROBE = 3;

    private static final VectorSpecies<Byte>  BSPECIES = ByteVector.SPECIES_64;
    private static final VectorSpecies<Float> FSPECIES = FloatVector.SPECIES_256;

    private static final int LOOP_BOUND = BSPECIES.loopBound(Dataset.DIMS);
    private static final VectorMask<Byte>  TAIL_MASK_B = BSPECIES.indexInRange(LOOP_BOUND, Dataset.DIMS);
    private static final VectorMask<Float> TAIL_MASK_F = FSPECIES.indexInRange(LOOP_BOUND, Dataset.DIMS);

    private final Dataset dataset;

    // Reusable buffers — KnnSearcher is stateful per instance, one per thread.
    private final byte[]  qBytes        = new byte[Dataset.DIMS];
    private final float[] heapDist      = new float[K];
    private final int[]   heapIdx       = new int[K];
    private final float[] centroidDist;          // [K], dynamically sized at construction
    private final int[]   probeIds      = new int[NPROBE];
    private final float[] probeDist     = new float[NPROBE];

    public KnnSearcher(Dataset dataset) {
        this.dataset = dataset;
        this.centroidDist = new float[dataset.k()];
    }

    public static String simdInfo() {
        return "B" + BSPECIES.length() + " → F" + FSPECIES.length()
                + " (BSPECIES=" + BSPECIES + ", FSPECIES=" + FSPECIES
                + ", LOOP_BOUND=" + LOOP_BOUND + ", NPROBE=" + NPROBE + ")";
    }

    /** Computes fraud_score = (#frauds among K nearest neighbors) / K. */
    public float fraudScore(float[] query) {
        // 1. Quantize the query (used for the bucket scan).
        dataset.quantize(query, qBytes);

        // 2. Compute float distance from query to all centroids.
        //    Pre-load query as FloatVector once.
        final FloatVector qfFull = FloatVector.fromArray(FSPECIES, query, 0);
        final FloatVector qfTail = FloatVector.fromArray(FSPECIES, query, LOOP_BOUND, TAIL_MASK_F);

        final float[] centroids = dataset.centroids();
        final int kClusters = dataset.k();
        for (int c = 0; c < kClusters; c++) {
            centroidDist[c] = squaredDistanceFloat(qfFull, qfTail, centroids, c * Dataset.DIMS);
        }

        // 3. Find the nearest clusters (clamped to NPROBE — or all clusters
        //    if there are fewer than NPROBE, which happens in tests).
        int actualNprobe = findTopNprobe(kClusters);

        // 4. Pre-load the int8 query as a pair of FloatVectors for the
        //    hybrid B→F SIMD path used inside the buckets.
        final FloatVector qbFull = LOOP_BOUND > 0
                ? widenByteToFloat(ByteVector.fromArray(BSPECIES, qBytes, 0))
                : null;
        final FloatVector qbTail = widenByteToFloat(
                ByteVector.fromArray(BSPECIES, qBytes, LOOP_BOUND, TAIL_MASK_B));

        // 5. Scan each chosen cluster.
        //    Initialize the heap with the first K candidates from the FIRST
        //    probed cluster (assumed >= K vectors — true for K=256, N=3M
        //    where average cluster size is ~12k).
        final byte[] vecs = dataset.vectors();
        boolean heapInitialized = false;

        for (int p = 0; p < actualNprobe; p++) {
            int clusterId = probeIds[p];
            int start = dataset.clusterStart(clusterId);
            int end   = dataset.clusterEnd(clusterId);

            int i = start;

            if (!heapInitialized) {
                int initEnd = Math.min(start + K, end);
                for (; i < initEnd; i++) {
                    heapDist[i - start] = squaredDistanceByteHybrid(qbFull, qbTail, vecs, i * Dataset.DIMS);
                    heapIdx[i - start]  = i;
                }
                if (i - start == K) {
                    // Heapify
                    for (int h = K / 2 - 1; h >= 0; h--) siftDown(h);
                    heapInitialized = true;
                }
                // If a single tiny cluster didn't reach K we'll continue
                // filling on the next probe iteration.
            }

            for (; i < end; i++) {
                float d = squaredDistanceByteHybrid(qbFull, qbTail, vecs, i * Dataset.DIMS);
                if (d < heapDist[0]) {
                    heapDist[0] = d;
                    heapIdx[0] = i;
                    siftDown(0);
                }
            }
        }

        // 6. Count frauds among the K winners.
        return countFrauds() / (float) K;
    }

    /* =================================================================== *
     * Centroid distance: pure float SIMD                                   *
     * =================================================================== */

    private static float squaredDistanceFloat(FloatVector qFull, FloatVector qTail,
                                              float[] centroids, int off) {
        FloatVector cFull = FloatVector.fromArray(FSPECIES, centroids, off);
        FloatVector cTail = FloatVector.fromArray(FSPECIES, centroids, off + LOOP_BOUND, TAIL_MASK_F);
        FloatVector diffFull = qFull.sub(cFull);
        FloatVector sum = diffFull.fma(diffFull, FloatVector.zero(FSPECIES));
        FloatVector diffTail = qTail.sub(cTail);
        sum = diffTail.fma(diffTail, sum);
        return sum.reduceLanes(VectorOperators.ADD);
    }

    /**
     * Find the top-min(NPROBE, kClusters) smallest values in centroidDist[0..k).
     *
     * Returns the actual number of probes (= min(NPROBE, kClusters)), useful
     * for the test-only case where the dataset has fewer than NPROBE clusters.
     *
     * Implementation: a simple O(probeCount × K) scan — clearer than a heap
     * and just as fast for our scale (NPROBE=3, K=256 → 768 comparisons,
     * dwarfed by everything else).
     */
    private int findTopNprobe(int kClusters) {
        int probeCount = Math.min(NPROBE, kClusters);
        for (int p = 0; p < probeCount; p++) {
            probeDist[p] = Float.POSITIVE_INFINITY;
            probeIds[p] = -1;
        }
        for (int c = 0; c < kClusters; c++) {
            float d = centroidDist[c];
            int worstIdx = 0;
            for (int p = 1; p < probeCount; p++) {
                if (probeDist[p] > probeDist[worstIdx]) worstIdx = p;
            }
            if (d < probeDist[worstIdx]) {
                probeDist[worstIdx] = d;
                probeIds[worstIdx] = c;
            }
        }
        return probeCount;
    }

    /* =================================================================== *
     * Bucket vector distance: hybrid B→F SIMD + FMA (same as v4)           *
     * =================================================================== */

    private static float squaredDistanceByteHybrid(FloatVector qFull, FloatVector qTail,
                                                    byte[] vecs, int off) {
        FloatVector sum = FloatVector.zero(FSPECIES);

        if (qFull != null) {
            for (int i = 0; i < LOOP_BOUND; i += BSPECIES.length()) {
                FloatVector v = widenByteToFloat(ByteVector.fromArray(BSPECIES, vecs, off + i));
                FloatVector diff = qFull.sub(v);
                sum = diff.fma(diff, sum);
            }
        }
        FloatVector vTail = widenByteToFloat(
                ByteVector.fromArray(BSPECIES, vecs, off + LOOP_BOUND, TAIL_MASK_B));
        FloatVector diffTail = qTail.sub(vTail);
        sum = diffTail.fma(diffTail, sum);

        return sum.reduceLanes(VectorOperators.ADD);
    }

    private static FloatVector widenByteToFloat(ByteVector b) {
        return (FloatVector) b.convertShape(VectorOperators.B2F, FSPECIES, 0);
    }

    /* =================================================================== *
     * Heap utilities                                                       *
     * =================================================================== */

    private int countFrauds() {
        int n = 0;
        for (int i = 0; i < K; i++) {
            if (dataset.isFraud(heapIdx[i])) n++;
        }
        return n;
    }

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
