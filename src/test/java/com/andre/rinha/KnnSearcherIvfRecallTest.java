package com.andre.rinha;

import com.andre.rinha.vector.Dataset;
import com.andre.rinha.vector.KMeans;
import com.andre.rinha.vector.KnnSearcher;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * v5's central correctness test: how often does IVF k-NN agree with brute
 * force on the top-5 nearest neighbors?
 *
 * Setup:
 *   - 10 k synthetic vectors in 14 dims, with a random sentinel-style -1 in
 *     dims 5/6 to mimic real "no last transaction" cases.
 *   - Cluster the float vectors with K=16 using {@link KMeans} (smaller K
 *     than production's 256 because N is small here — keeps clusters big
 *     enough to be meaningful, otherwise too many singleton clusters skew
 *     the test).
 *   - Quantize to int8 with global min/max (same recipe as DatasetBuilder).
 *   - Reorder vectors and labels by cluster (same recipe as DatasetBuilder).
 *   - For each random query, compute top-5 with float32 brute force (ground
 *     truth) AND with v5 IVF KnnSearcher. Measure agreement.
 *
 * Acceptance:
 *   - fraud_score equality ≥ 80 % on synthetic uniform data (worst case).
 *   - perfect top-5 set match is *informational* — IVF is approximate by
 *     design and uniform data has lots of near-ties.
 *
 * On real fraud data (clearer cluster structure) we expect noticeably
 * higher agreement. The k6 benchmark is the final word.
 */
class KnnSearcherIvfRecallTest {

    private static final int DIMS = Dataset.DIMS;
    private static final int N = 10_000;
    private static final int K_CLUSTERS = 16;     // small for synthetic test
    private static final int QUERIES = 200;
    private static final int KMEANS_MAX_ITERS = 30;
    private static final long DATA_SEED = 42L;
    private static final long QUERY_SEED = 7L;
    private static final long KMEANS_SEED = 11L;

    @Test
    void ivfRecallVsBruteForce() {
        // ---- Build synthetic dataset ----
        Random rng = new Random(DATA_SEED);
        float[] vectorsF = new float[N * DIMS];
        for (int i = 0; i < vectorsF.length; i++) {
            vectorsF[i] = rng.nextFloat();
            if ((i % DIMS == 5 || i % DIMS == 6) && rng.nextFloat() < 0.30f) {
                vectorsF[i] = -1f;
            }
        }
        BitSet labelsTrue = new BitSet(N);
        for (int i = 0; i < N; i++) {
            if (rng.nextFloat() < 0.30f) labelsTrue.set(i);
        }

        // ---- k-means cluster the float vectors ----
        KMeans.Result km = KMeans.fit(vectorsF, N, DIMS, K_CLUSTERS, KMEANS_MAX_ITERS, KMEANS_SEED);

        // ---- Compute global mins/maxs and quantize ----
        float globalMin = Float.POSITIVE_INFINITY;
        float globalMax = Float.NEGATIVE_INFINITY;
        for (float v : vectorsF) {
            if (v < globalMin) globalMin = v;
            if (v > globalMax) globalMax = v;
        }
        float[] mins = new float[DIMS];
        float[] maxs = new float[DIMS];
        Arrays.fill(mins, globalMin);
        Arrays.fill(maxs, globalMax);

        byte[] vectorsI8 = new byte[N * DIMS];
        float scale = 255f / (globalMax - globalMin);
        for (int i = 0; i < N; i++) {
            int base = i * DIMS;
            for (int d = 0; d < DIMS; d++) {
                int q = Math.round((vectorsF[base + d] - globalMin) * scale) - 128;
                if (q < -128) q = -128;
                if (q >  127) q =  127;
                vectorsI8[base + d] = (byte) q;
            }
        }

        // ---- Reorder vectors+labels by cluster (mirror DatasetBuilder) ----
        int[] assignments = km.assignments();
        int[] counts = new int[K_CLUSTERS];
        for (int a : assignments) counts[a]++;
        int[] offsets = new int[K_CLUSTERS + 1];
        for (int c = 0; c < K_CLUSTERS; c++) offsets[c + 1] = offsets[c] + counts[c];

        byte[] reorderedVecs = new byte[N * DIMS];
        BitSet reorderedLabels = new BitSet(N);
        // Track reorderedIdx → originalIdx so float-ground-truth uses the same indexing
        int[] newToOld = new int[N];
        int[] writePos = offsets.clone();
        for (int oldIdx = 0; oldIdx < N; oldIdx++) {
            int c = assignments[oldIdx];
            int newIdx = writePos[c]++;
            System.arraycopy(vectorsI8, oldIdx * DIMS, reorderedVecs, newIdx * DIMS, DIMS);
            if (labelsTrue.get(oldIdx)) reorderedLabels.set(newIdx);
            newToOld[newIdx] = oldIdx;
        }

        // For float ground truth we also need the floats in the same
        // reordered layout so we compare apples to apples.
        float[] reorderedFloats = new float[N * DIMS];
        for (int newIdx = 0; newIdx < N; newIdx++) {
            int oldIdx = newToOld[newIdx];
            System.arraycopy(vectorsF, oldIdx * DIMS, reorderedFloats, newIdx * DIMS, DIMS);
        }

        Dataset ds = Dataset.fromArrays(reorderedVecs, reorderedLabels, mins, maxs,
                km.centroids(), offsets);
        KnnSearcher ivf = new KnnSearcher(ds);

        // ---- Run comparison ----
        Random qrng = new Random(QUERY_SEED);
        int perfectMatch = 0;
        int scoreEqual = 0;
        for (int q = 0; q < QUERIES; q++) {
            float[] query = new float[DIMS];
            for (int d = 0; d < DIMS; d++) {
                query[d] = qrng.nextFloat();
                if ((d == 5 || d == 6) && qrng.nextFloat() < 0.30f) query[d] = -1f;
            }

            int[] truthTop = topKFloat32(query, reorderedFloats, N);
            float truthScore = fraudFraction(truthTop, reorderedLabels);

            float ivfScore = ivf.fraudScore(query);

            if (Math.abs(truthScore - ivfScore) < 1e-6) scoreEqual++;
            // Note: we can't easily extract IVF's top-5 indices without
            // adding API. We assert on score equality which is what users see.
        }

        double scoreRate = scoreEqual / (double) QUERIES;
        System.out.printf("[ivf] fraud_score equal: %d/%d (%.1f%%) on K=%d clusters, NPROBE=%d%n",
                scoreEqual, QUERIES, scoreRate * 100, K_CLUSTERS, KnnSearcher.NPROBE);

        assertTrue(scoreRate >= 0.80,
                "IVF recall too low on synthetic uniform data: " + scoreRate
                        + " — increase NPROBE or check k-means convergence");
    }

    /* ---- helpers ---- */

    private static int[] topKFloat32(float[] q, float[] vecs, int n) {
        Float[] dists = new Float[n];
        for (int i = 0; i < n; i++) {
            float sum = 0;
            int base = i * DIMS;
            for (int d = 0; d < DIMS; d++) {
                float diff = q[d] - vecs[base + d];
                sum += diff * diff;
            }
            dists[i] = sum;
        }
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Float.compare(dists[a], dists[b]));
        int[] top = new int[KnnSearcher.K];
        for (int i = 0; i < KnnSearcher.K; i++) top[i] = indices[i];
        return top;
    }

    private static float fraudFraction(int[] topK, BitSet labels) {
        int frauds = 0;
        for (int idx : topK) if (labels.get(idx)) frauds++;
        return frauds / (float) topK.length;
    }
}
