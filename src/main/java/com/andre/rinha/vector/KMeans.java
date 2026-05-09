package com.andre.rinha.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.SplittableRandom;

/**
 * Lloyd's k-means with k-means++ initialization.
 *
 * Used at build time to partition the 3M reference vectors into K clusters,
 * which v5's IVF (Inverted File) KnnSearcher uses to skip ~99% of brute-force
 * comparisons at query time.
 *
 * Performance hot path is the assignment step (each iteration): for every
 * vector, compute the squared L2 distance to all K centroids and pick the
 * smallest. We do that with FloatVector + FMA so a single iteration over
 * 3M × 256 × 14 ops takes a few seconds rather than ~30s of pure scalar.
 *
 * Convergence:
 *   - Hard cap at maxIters (default 20).
 *   - Early termination when fewer than 0.1% of assignments change between
 *     iterations (Lloyd's typically settles fast — most movement happens in
 *     the first 5-8 iters).
 *
 * Empty clusters: a cluster with zero assigned vectors is reseeded with a
 * random vector from the dataset. Rare but possible at high K.
 */
public final class KMeans {

    /** Result of a fit() call. */
    public record Result(
            float[] centroids,   // [K * dims], flat layout
            int[] assignments,   // [N] — assignment of each vector to a cluster id
            int iterations
    ) {}

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int LOOP_BOUND = SPECIES.loopBound(Dataset.DIMS);
    private static final VectorMask<Float> TAIL_MASK = SPECIES.indexInRange(LOOP_BOUND, Dataset.DIMS);

    private KMeans() {}

    /**
     * Run Lloyd's k-means.
     *
     * @param vectors  flat [n * dims] float array of all input vectors
     * @param n        number of vectors
     * @param dims     dimensions per vector (must equal Dataset.DIMS so SIMD
     *                 species line up)
     * @param k        target number of clusters
     * @param maxIters hard cap on iterations
     * @param seed     for reproducibility
     */
    public static Result fit(float[] vectors, int n, int dims, int k, int maxIters, long seed) {
        if (dims != Dataset.DIMS) {
            throw new IllegalArgumentException(
                    "KMeans uses SIMD configured for DIMS=" + Dataset.DIMS + ", got " + dims);
        }

        SplittableRandom rng = new SplittableRandom(seed);
        float[] centroids = kmeansPlusPlusInit(vectors, n, dims, k, rng);
        int[] assignments = new int[n];
        // Initial assignments: -1 means "no assignment yet" so the first iter
        // counts every assignment as a "change" (no early stop on iter 0).
        for (int i = 0; i < n; i++) assignments[i] = -1;

        int iter;
        for (iter = 0; iter < maxIters; iter++) {
            long t = System.currentTimeMillis();

            int changes = assignAll(vectors, n, dims, centroids, k, assignments);

            updateCentroids(vectors, n, dims, assignments, centroids, k, rng);

            double pct = changes * 100.0 / n;
            System.out.printf("[kmeans] iter %2d: %.2f%% reassigned (%d), %.1fs%n",
                    iter, pct, changes, (System.currentTimeMillis() - t) / 1000.0);

            // Early termination: < 0.1% movement and we've done at least 5
            // iterations (so the first few aren't tricked by lucky init).
            if (iter >= 5 && pct < 0.1) {
                System.out.printf("[kmeans] converged after %d iterations%n", iter + 1);
                iter++;
                break;
            }
        }

        return new Result(centroids, assignments, iter);
    }

    /* ===================================================================
     * K-means++ initialization
     *
     * Pick the first centroid uniformly. For each subsequent centroid c_i,
     * sample a vector with probability proportional to D(x)², where D(x) is
     * the distance from x to its nearest already-chosen centroid. This
     * spreads centroids well across the data — pure random init can produce
     * very unbalanced clusters.
     * ================================================================== */
    private static float[] kmeansPlusPlusInit(float[] vectors, int n, int dims, int k,
                                              SplittableRandom rng) {
        float[] centroids = new float[k * dims];

        // First centroid: uniform random.
        int firstIdx = rng.nextInt(n);
        System.arraycopy(vectors, firstIdx * dims, centroids, 0, dims);

        // Per-vector squared distance to the closest already-chosen centroid.
        float[] minDistSq = new float[n];
        for (int i = 0; i < n; i++) {
            minDistSq[i] = squaredDistanceFloat(vectors, i * dims, centroids, 0, dims);
        }

        for (int c = 1; c < k; c++) {
            // Total of D(x)² across all vectors.
            double total = 0;
            for (int i = 0; i < n; i++) total += minDistSq[i];

            // Sample proportional to D(x)².
            double r = rng.nextDouble() * total;
            double acc = 0;
            int pickIdx = n - 1;
            for (int i = 0; i < n; i++) {
                acc += minDistSq[i];
                if (acc >= r) { pickIdx = i; break; }
            }
            System.arraycopy(vectors, pickIdx * dims, centroids, c * dims, dims);

            // Update minDistSq with the new centroid.
            int cBase = c * dims;
            for (int i = 0; i < n; i++) {
                float d = squaredDistanceFloat(vectors, i * dims, centroids, cBase, dims);
                if (d < minDistSq[i]) minDistSq[i] = d;
            }
        }

        return centroids;
    }

    /* ===================================================================
     * Assignment step — SIMD-accelerated.
     *
     * For each vector, find the centroid with the smallest squared distance.
     * Returns the count of vectors whose assignment changed (used for
     * early-termination decision in the outer loop).
     *
     * We pre-load each vector into a pair of FloatVectors once and iterate
     * the K centroids in the inner loop — keeps the vector in registers
     * and only re-reads the centroid table (which fits in L1 for K=256).
     * ================================================================== */
    private static int assignAll(float[] vectors, int n, int dims,
                                  float[] centroids, int k, int[] assignments) {
        int changes = 0;
        for (int i = 0; i < n; i++) {
            int base = i * dims;
            FloatVector qFull = FloatVector.fromArray(SPECIES, vectors, base);
            FloatVector qTail = FloatVector.fromArray(SPECIES, vectors, base + LOOP_BOUND, TAIL_MASK);

            int bestK = 0;
            float bestDist = Float.MAX_VALUE;
            for (int c = 0; c < k; c++) {
                int cBase = c * dims;
                FloatVector cFull = FloatVector.fromArray(SPECIES, centroids, cBase);
                FloatVector cTail = FloatVector.fromArray(SPECIES, centroids, cBase + LOOP_BOUND, TAIL_MASK);

                FloatVector diffFull = qFull.sub(cFull);
                FloatVector sum = diffFull.fma(diffFull, FloatVector.zero(SPECIES));
                FloatVector diffTail = qTail.sub(cTail);
                sum = diffTail.fma(diffTail, sum);

                float dist = sum.reduceLanes(VectorOperators.ADD);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestK = c;
                }
            }

            if (assignments[i] != bestK) {
                assignments[i] = bestK;
                changes++;
            }
        }
        return changes;
    }

    /* ===================================================================
     * Update step — recompute each centroid as the mean of its assigned
     * vectors. Empty clusters are reseeded with a random vector.
     * ================================================================== */
    private static void updateCentroids(float[] vectors, int n, int dims,
                                        int[] assignments, float[] centroids, int k,
                                        SplittableRandom rng) {
        // Zero centroids and counts.
        for (int i = 0; i < centroids.length; i++) centroids[i] = 0f;
        int[] counts = new int[k];

        for (int i = 0; i < n; i++) {
            int c = assignments[i];
            counts[c]++;
            int srcBase = i * dims;
            int dstBase = c * dims;
            for (int d = 0; d < dims; d++) {
                centroids[dstBase + d] += vectors[srcBase + d];
            }
        }

        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                int base = c * dims;
                float inv = 1f / counts[c];
                for (int d = 0; d < dims; d++) centroids[base + d] *= inv;
            } else {
                // Empty cluster — seed with a random vector. Rare but happens
                // at high K when initial centroids cluster too tightly.
                int rndIdx = rng.nextInt(n);
                System.arraycopy(vectors, rndIdx * dims, centroids, c * dims, dims);
            }
        }
    }

    /* ===================================================================
     * Squared L2 distance — scalar version used during k-means++ init.
     * The init step is O(N*K) once, so we don't bother SIMD-ifying it.
     * ================================================================== */
    private static float squaredDistanceFloat(float[] a, int aOff, float[] b, int bOff, int dims) {
        float sum = 0;
        for (int d = 0; d < dims; d++) {
            float diff = a[aOff + d] - b[bOff + d];
            sum += diff * diff;
        }
        return sum;
    }
}
