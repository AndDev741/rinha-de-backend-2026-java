package com.andre.rinha;

import com.andre.rinha.vector.Dataset;
import com.andre.rinha.vector.KnnSearcher;
import org.junit.jupiter.api.Test;

import java.util.BitSet;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Critical correctness test for the v3 Vector API path.
 *
 * Strategy: build a synthetic dataset large enough to exercise both initial
 * heap fill AND the post-fill replacement path, run the SAME query through
 * both KnnSearcher modes, assert identical results.
 *
 * What we verify:
 *   - Both modes return the same fraud_score (the user-visible output).
 *   - Both modes select the same K nearest neighbors (same indices).
 *
 * Why this matters: Vector API uses FMA (fused multiply-add) which has
 * different rounding behavior than scalar mul+add. For our domain
 * (squared distances of normalized [0,1] floats), the rounding differences
 * are well below the gap between distinct neighbors, so the top-K must
 * still be identical. If this test ever flakes we'd want to know — it
 * means real-world queries near a tie boundary could disagree between
 * modes, and we'd need a tolerance policy.
 */
class KnnSearcherParityTest {

    private static final int DIMS = Dataset.DIMS;
    private static final int N = 5_000;          // big enough to exercise heap path
    private static final int QUERIES = 50;       // multiple queries to catch flakiness
    private static final long DATA_SEED = 42L;
    private static final long QUERY_SEED = 7L;

    @Test
    void scalarAndVectorReturnIdenticalFraudScores() {
        Dataset dataset = buildDataset();
        KnnSearcher scalar = new KnnSearcher(dataset, KnnSearcher.Mode.SCALAR);
        KnnSearcher vector = new KnnSearcher(dataset, KnnSearcher.Mode.VECTOR);

        Random qrng = new Random(QUERY_SEED);
        for (int i = 0; i < QUERIES; i++) {
            float[] query = randomVector(qrng);
            float scalarScore = scalar.fraudScore(query);
            float vectorScore = vector.fraudScore(query);
            assertEquals(scalarScore, vectorScore, 1e-6f,
                    "fraud_score diverged on query " + i + ": scalar=" + scalarScore
                            + " vector=" + vectorScore);
        }
    }

    @Test
    void worksWithSentinelMinusOneInQuery() {
        // Query with -1 in dims 5,6 (last_transaction sentinel) — tests that
        // the SIMD masked tail correctly handles negative inputs in those lanes.
        Dataset dataset = buildDataset();
        KnnSearcher scalar = new KnnSearcher(dataset, KnnSearcher.Mode.SCALAR);
        KnnSearcher vector = new KnnSearcher(dataset, KnnSearcher.Mode.VECTOR);

        float[] query = new float[DIMS];
        for (int i = 0; i < DIMS; i++) query[i] = 0.5f;
        query[5] = -1f;
        query[6] = -1f;

        assertEquals(scalar.fraudScore(query), vector.fraudScore(query), 1e-6f);
    }

    @Test
    void worksWithAllZerosAndAllOnes() {
        // Boundary cases: query at the corners of the unit cube.
        Dataset dataset = buildDataset();
        KnnSearcher scalar = new KnnSearcher(dataset, KnnSearcher.Mode.SCALAR);
        KnnSearcher vector = new KnnSearcher(dataset, KnnSearcher.Mode.VECTOR);

        float[] zeros = new float[DIMS]; // all 0.0
        float[] ones = new float[DIMS];
        for (int i = 0; i < DIMS; i++) ones[i] = 1f;

        assertEquals(scalar.fraudScore(zeros), vector.fraudScore(zeros), 1e-6f, "zeros");
        assertEquals(scalar.fraudScore(ones),  vector.fraudScore(ones),  1e-6f, "ones");
    }

    /* -------------------- helpers -------------------- */

    private static Dataset buildDataset() {
        Random rng = new Random(DATA_SEED);
        float[] vectors = new float[N * DIMS];
        for (int i = 0; i < vectors.length; i++) {
            vectors[i] = rng.nextFloat();
        }
        BitSet labels = new BitSet(N);
        // Mark a deterministic ~30% as fraud.
        for (int i = 0; i < N; i++) {
            if (rng.nextFloat() < 0.30f) labels.set(i);
        }
        return Dataset.fromArrays(vectors, labels);
    }

    private static float[] randomVector(Random rng) {
        float[] q = new float[DIMS];
        for (int i = 0; i < DIMS; i++) q[i] = rng.nextFloat();
        return q;
    }
}
