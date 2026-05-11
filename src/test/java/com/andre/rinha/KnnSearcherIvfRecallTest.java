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
 * v7's central correctness test: int16 IVF + bbox repair must match float32
 * brute force on the top-5 nearest neighbors, modulo int16 rounding noise.
 *
 * Setup mirrors DatasetBuilder:
 *   1. Build a synthetic float32 dataset (10 k vectors, 14 dims) with some
 *      -1 sentinels in dims 5/6.
 *   2. Cluster the float vectors with KMeans (K=16 for the synthetic).
 *   3. Quantize to int16 with × 10000 scaling.
 *   4. Reorder vectors+labels by cluster (mirrors DatasetBuilder).
 *   5. Compute bbox per cluster.
 *   6. For each random query, compute top-5 via:
 *        - float32 brute force  (ground truth)
 *        - v7 IVF + bbox repair (under test)
 *      Compare fraud_score agreement.
 *
 * Acceptance:
 *   With bbox repair the v7 search is EXACT — same top-5 as brute force
 *   except for ties that fall on a 0.0001 int16 rounding boundary. We
 *   expect >= 99% fraud_score equality even on synthetic uniform data
 *   (which is the worst case for k-NN ordering stability).
 */
class KnnSearcherIvfRecallTest {

    private static final int DIMS = Dataset.DIMS;
    private static final int N = 10_000;
    private static final int K_CLUSTERS = 16;
    private static final int QUERIES = 200;
    private static final int KMEANS_MAX_ITERS = 30;
    private static final long DATA_SEED = 42L;
    private static final long QUERY_SEED = 7L;
    private static final long KMEANS_SEED = 11L;

    @Test
    void int16IvfBboxMatchesFloat32BruteForce() {
        // ---- Build synthetic float dataset ----
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

        // ---- k-means on float vectors ----
        KMeans.Result km = KMeans.fit(vectorsF, N, DIMS, K_CLUSTERS, KMEANS_MAX_ITERS, KMEANS_SEED);

        // ---- Quantize to int16 (× 10000) ----
        short[] vectorsI16 = new short[N * DIMS];
        for (int i = 0; i < N * DIMS; i++) {
            int q = Math.round(vectorsF[i] * Dataset.SCALE);
            if (q < Short.MIN_VALUE) q = Short.MIN_VALUE;
            if (q > Short.MAX_VALUE) q = Short.MAX_VALUE;
            vectorsI16[i] = (short) q;
        }

        // ---- Reorder by cluster ----
        int[] assignments = km.assignments();
        int[] counts = new int[K_CLUSTERS];
        for (int a : assignments) counts[a]++;
        int[] offsets = new int[K_CLUSTERS + 1];
        for (int c = 0; c < K_CLUSTERS; c++) offsets[c + 1] = offsets[c] + counts[c];

        short[] reorderedVecs = new short[N * DIMS];
        BitSet reorderedLabels = new BitSet(N);
        float[] reorderedFloats = new float[N * DIMS];
        int[] writePos = offsets.clone();
        for (int oldIdx = 0; oldIdx < N; oldIdx++) {
            int c = assignments[oldIdx];
            int newIdx = writePos[c]++;
            System.arraycopy(vectorsI16, oldIdx * DIMS, reorderedVecs, newIdx * DIMS, DIMS);
            System.arraycopy(vectorsF,   oldIdx * DIMS, reorderedFloats, newIdx * DIMS, DIMS);
            if (labelsTrue.get(oldIdx)) reorderedLabels.set(newIdx);
        }

        // ---- Compute per-cluster bbox ----
        short[] bboxMin = new short[K_CLUSTERS * DIMS];
        short[] bboxMax = new short[K_CLUSTERS * DIMS];
        for (int c = 0; c < K_CLUSTERS; c++) {
            int base = c * DIMS;
            for (int d = 0; d < DIMS; d++) {
                bboxMin[base + d] = Short.MAX_VALUE;
                bboxMax[base + d] = Short.MIN_VALUE;
            }
            for (int i = offsets[c]; i < offsets[c + 1]; i++) {
                int vBase = i * DIMS;
                for (int d = 0; d < DIMS; d++) {
                    short v = reorderedVecs[vBase + d];
                    if (v < bboxMin[base + d]) bboxMin[base + d] = v;
                    if (v > bboxMax[base + d]) bboxMax[base + d] = v;
                }
            }
        }

        // ---- Convert centroids to int16 too ----
        float[] cf = km.centroids();
        short[] centroidsI16 = new short[cf.length];
        for (int i = 0; i < cf.length; i++) {
            int q = Math.round(cf[i] * Dataset.SCALE);
            if (q < Short.MIN_VALUE) q = Short.MIN_VALUE;
            if (q > Short.MAX_VALUE) q = Short.MAX_VALUE;
            centroidsI16[i] = (short) q;
        }

        Dataset ds = Dataset.fromArrays(reorderedVecs, reorderedLabels,
                centroidsI16, offsets, bboxMin, bboxMax);
        KnnSearcher search = new KnnSearcher(ds);

        // ---- Compare ----
        Random qrng = new Random(QUERY_SEED);
        int scoreEqual = 0;
        for (int q = 0; q < QUERIES; q++) {
            float[] query = new float[DIMS];
            for (int d = 0; d < DIMS; d++) {
                query[d] = qrng.nextFloat();
                if ((d == 5 || d == 6) && qrng.nextFloat() < 0.30f) query[d] = -1f;
            }

            int[] truthTop = topKFloat32(query, reorderedFloats, N);
            float truthScore = fraudFraction(truthTop, reorderedLabels);

            float v7Score = search.fraudScore(query);

            if (Math.abs(truthScore - v7Score) < 1e-6) scoreEqual++;
        }

        double scoreRate = scoreEqual / (double) QUERIES;
        System.out.printf("[v7] fraud_score equal: %d/%d (%.1f%%) — K=%d clusters, bbox-repair%n",
                scoreEqual, QUERIES, scoreRate * 100, K_CLUSTERS);

        // Bbox repair is exact — should be >= 99% even on uniform synthetic data.
        assertTrue(scoreRate >= 0.99,
                "v7 int16 + IVF + bbox-repair should match float32 brute force, got "
                        + scoreRate + " agreement");
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
