package com.andre.rinha;

import com.andre.rinha.vector.Dataset;
import com.andre.rinha.vector.KnnSearcher;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * v4's key correctness test: how often does int8 k-NN agree with float32
 * ground truth on the top-5 selection?
 *
 * Why this matters:
 *   v4 swaps the dataset from float32 to int8 with per-dimension min/max
 *   scaling. The squared distance computed in int8 space is a per-dim
 *   *weighted* L2 of the original float distance — same ordering only when
 *   all dims have similar ranges. Our 14 dims mostly land in [0, 1] (some
 *   in [-1, 1]), so weights are nearly uniform, and we expect very high
 *   top-5 agreement.
 *
 * What we measure:
 *   For each random query, compute the top-5 nearest neighbors with float32
 *   brute force (ground truth) and with the v4 int8 KnnSearcher. Compare
 *   the two sets of indices. Track:
 *     - perfect_match  : both sets are identical (order ignored)
 *     - fraud_score_eq : both produce the same fraud_score (final user value)
 *
 * Acceptance:
 *   - fraud_score_eq >= 90% of queries (the user-visible output)
 *
 * The threshold reflects the SYNTHETIC test conditions: uniform random data
 * across 14 dims is the worst case for k-NN ordering — vectors are fuzzy and
 * neighbors are very close in distance, so quantization noise easily swaps
 * top-5 positions. Real fraud-detection data has clearer cluster structure
 * (frauds and legits are well separated), so we expect noticeably higher
 * agreement on the actual rinha k6 test. The 90% bar is a regression
 * detector, not a correctness guarantee — the real validation is the
 * detection_score on the k6 benchmark.
 *
 * The "perfect_match" stat is logged for visibility but not asserted —
 * top-5 set agreement is harder than score agreement (k-NN voting is robust
 * to a single swap), and we mostly care about the user-visible score.
 */
class KnnSearcherInt8ParityTest {

    private static final int DIMS = Dataset.DIMS;
    private static final int N = 10_000;          // synthetic dataset size
    private static final int QUERIES = 200;       // sample size for stats
    private static final long DATA_SEED = 42L;
    private static final long QUERY_SEED = 7L;

    @Test
    void int8AgreesWithFloat32OnTopFive() {
        // Build a synthetic dataset in float32 form.
        Random rng = new Random(DATA_SEED);
        float[] vectorsF = new float[N * DIMS];
        for (int i = 0; i < vectorsF.length; i++) {
            vectorsF[i] = rng.nextFloat();
            // Simulate the sentinel by occasionally placing -1 in dims 5/6
            // so the test exercises that quantization edge case.
            if ((i % 14 == 5 || i % 14 == 6) && rng.nextFloat() < 0.3f) {
                vectorsF[i] = -1f;
            }
        }
        BitSet labels = new BitSet(N);
        for (int i = 0; i < N; i++) {
            if (rng.nextFloat() < 0.30f) labels.set(i);
        }

        // Quantize the synthetic dataset using GLOBAL min/max — same recipe
        // as DatasetBuilder. (Per-dim min/max creates non-uniform per-dim
        // weights in the int8 distance, which breaks parity with float32.
        // See the comments in DatasetBuilder for the gory details.)
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
        for (int i = 0; i < N; i++) {
            int base = i * DIMS;
            for (int d = 0; d < DIMS; d++) {
                float range = maxs[d] - mins[d];
                float scale = (range > 0) ? 255f / range : 0f;
                int q = Math.round((vectorsF[base + d] - mins[d]) * scale) - 128;
                if (q < -128) q = -128;
                if (q >  127) q =  127;
                vectorsI8[base + d] = (byte) q;
            }
        }

        // v5 Dataset requires centroids + cluster offsets. For this test
        // (which validates int8 quantization correctness, not IVF recall) we
        // pass a trivial single-cluster partition: all N vectors live in
        // cluster 0, so the IVF search degenerates to brute force over the
        // whole dataset. That's exactly what we want for a quantization test.
        float[] trivialCentroid = new float[DIMS]; // zeros — irrelevant with 1 cluster
        int[] trivialOffsets = new int[]{ 0, N };
        Dataset int8Dataset = Dataset.fromArrays(
                vectorsI8, labels, mins, maxs, trivialCentroid, trivialOffsets);
        KnnSearcher int8Searcher = new KnnSearcher(int8Dataset);

        // Run the comparison.
        Random qrng = new Random(QUERY_SEED);
        int perfectMatches = 0;
        int fraudScoreEqual = 0;

        for (int q = 0; q < QUERIES; q++) {
            float[] query = new float[DIMS];
            for (int d = 0; d < DIMS; d++) {
                query[d] = qrng.nextFloat();
                if ((d == 5 || d == 6) && qrng.nextFloat() < 0.3f) query[d] = -1f;
            }

            // Float32 ground truth (brute force).
            int[] truthTopK = topKFloat32(query, vectorsF, labels, N);
            float truthScore = fraudFraction(truthTopK, labels);

            // Int8 (the actual v4 path).
            float int8Score = int8Searcher.fraudScore(query);

            // To compare top-K we need the int8 top-K. KnnSearcher doesn't
            // expose them, so we reconstruct: re-run the int8 distance
            // ourselves and pick top-5. (Yes, this duplicates code; the
            // alternative is to widen the public API just for tests.)
            int[] int8TopK = topKInt8(query, vectorsI8, mins, maxs, N);

            if (sameSet(truthTopK, int8TopK)) perfectMatches++;
            if (Math.abs(truthScore - int8Score) < 1e-6) fraudScoreEqual++;
        }

        double matchRate = perfectMatches / (double) QUERIES;
        double scoreRate = fraudScoreEqual / (double) QUERIES;

        System.out.printf("[parity] perfect top-5 match: %d/%d (%.1f%%)%n",
                perfectMatches, QUERIES, matchRate * 100);
        System.out.printf("[parity] fraud_score equal:   %d/%d (%.1f%%)%n",
                fraudScoreEqual, QUERIES, scoreRate * 100);

        // Don't assert on perfect set match — k-NN voting tolerates 1-2 swaps
        // so this number is informational. We care about the score below.
        assertTrue(scoreRate >= 0.90,
                "fraud_score equality rate too low on synthetic uniform data: "
                        + scoreRate + " — int8 quantization may need attention "
                        + "before trusting it on real data");
    }

    /* -------------------- ground-truth helpers -------------------- */

    private static int[] topKFloat32(float[] q, float[] vecs, BitSet labels, int n) {
        // Trivial O(N log K) using a max-heap, but simpler to write as full
        // sort here since we're only running 200 queries × 10k vectors.
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

    private static int[] topKInt8(float[] q, byte[] vecs, float[] mins, float[] maxs, int n) {
        // Quantize query.
        byte[] qb = new byte[DIMS];
        for (int d = 0; d < DIMS; d++) {
            float range = maxs[d] - mins[d];
            float scale = (range > 0) ? 255f / range : 0f;
            int x = Math.round((q[d] - mins[d]) * scale) - 128;
            if (x < -128) x = -128;
            if (x >  127) x =  127;
            qb[d] = (byte) x;
        }
        Integer[] dists = new Integer[n];
        for (int i = 0; i < n; i++) {
            int sum = 0;
            int base = i * DIMS;
            for (int d = 0; d < DIMS; d++) {
                int diff = (qb[d] & 0xFF) - 128 - ((vecs[base + d] & 0xFF) - 128);
                // Wait — both qb[d] and vecs[base+d] are stored as signed bytes.
                // In Java byte is signed, so direct sub gives the right value.
                diff = qb[d] - vecs[base + d];
                sum += diff * diff;
            }
            dists[i] = sum;
        }
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Integer.compare(dists[a], dists[b]));
        int[] top = new int[KnnSearcher.K];
        for (int i = 0; i < KnnSearcher.K; i++) top[i] = indices[i];
        return top;
    }

    private static float fraudFraction(int[] topK, BitSet labels) {
        int frauds = 0;
        for (int idx : topK) if (labels.get(idx)) frauds++;
        return frauds / (float) topK.length;
    }

    private static boolean sameSet(int[] a, int[] b) {
        if (a.length != b.length) return false;
        int[] aa = a.clone(); int[] bb = b.clone();
        Arrays.sort(aa); Arrays.sort(bb);
        return Arrays.equals(aa, bb);
    }
}
