package com.andre.rinha.vector;

import java.util.SplittableRandom;

/**
 * Lloyd's k-means with k-means++ initialization. Build-time only.
 *
 * v7: rewritten to be pure scalar (no Vector API), matching the runtime
 * KnnSearcher's philosophy. K-means is a one-time build step (~1-3 min
 * for 3M × 256 × 20 iters); scalar with manual unrolling is fast enough
 * and keeps the project free of incubator dependencies.
 *
 * Convergence: hard cap at maxIters; early stop when fewer than 0.1 % of
 * assignments change between iterations.
 *
 * Empty clusters: reseeded with a random vector from the input.
 */
public final class KMeans {

    public record Result(
            float[] centroids,
            int[] assignments,
            int iterations
    ) {}

    private KMeans() {}

    public static Result fit(float[] vectors, int n, int dims, int k, int maxIters, long seed) {
        if (dims != Dataset.DIMS) {
            throw new IllegalArgumentException(
                    "KMeans expects DIMS=" + Dataset.DIMS + ", got " + dims);
        }

        SplittableRandom rng = new SplittableRandom(seed);
        float[] centroids = kmeansPlusPlusInit(vectors, n, dims, k, rng);
        int[] assignments = new int[n];
        for (int i = 0; i < n; i++) assignments[i] = -1;

        int iter;
        for (iter = 0; iter < maxIters; iter++) {
            long t = System.currentTimeMillis();

            int changes = assignAll(vectors, n, centroids, k, assignments);

            updateCentroids(vectors, n, dims, assignments, centroids, k, rng);

            double pct = changes * 100.0 / n;
            System.out.printf("[kmeans] iter %2d: %.2f%% reassigned (%d), %.1fs%n",
                    iter, pct, changes, (System.currentTimeMillis() - t) / 1000.0);

            if (iter >= 5 && pct < 0.1) {
                System.out.printf("[kmeans] converged after %d iterations%n", iter + 1);
                iter++;
                break;
            }
        }

        return new Result(centroids, assignments, iter);
    }

    /* =================================================================== *
     * K-means++ initialization                                             *
     * =================================================================== */
    private static float[] kmeansPlusPlusInit(float[] vectors, int n, int dims, int k,
                                              SplittableRandom rng) {
        float[] centroids = new float[k * dims];

        int firstIdx = rng.nextInt(n);
        System.arraycopy(vectors, firstIdx * dims, centroids, 0, dims);

        float[] minDistSq = new float[n];
        for (int i = 0; i < n; i++) {
            minDistSq[i] = squaredDistance(vectors, i * dims, centroids, 0);
        }

        for (int c = 1; c < k; c++) {
            double total = 0;
            for (int i = 0; i < n; i++) total += minDistSq[i];

            double r = rng.nextDouble() * total;
            double acc = 0;
            int pickIdx = n - 1;
            for (int i = 0; i < n; i++) {
                acc += minDistSq[i];
                if (acc >= r) { pickIdx = i; break; }
            }
            System.arraycopy(vectors, pickIdx * dims, centroids, c * dims, dims);

            int cBase = c * dims;
            for (int i = 0; i < n; i++) {
                float d = squaredDistance(vectors, i * dims, centroids, cBase);
                if (d < minDistSq[i]) minDistSq[i] = d;
            }
        }

        return centroids;
    }

    /* =================================================================== *
     * Assignment step — scalar manual unrolled                             *
     *                                                                       *
     * For each vector, scan all K centroids and pick the nearest. The      *
     * inner per-centroid distance is fully unrolled across the 14 dims so  *
     * the JIT can spill q[*] into registers and stream through the         *
     * centroids array sequentially.                                        *
     * =================================================================== */
    private static int assignAll(float[] vectors, int n,
                                  float[] centroids, int k, int[] assignments) {
        int changes = 0;
        for (int i = 0; i < n; i++) {
            int b = i * Dataset.DIMS;
            float q0  = vectors[b],     q1  = vectors[b + 1],  q2  = vectors[b + 2];
            float q3  = vectors[b + 3], q4  = vectors[b + 4],  q5  = vectors[b + 5];
            float q6  = vectors[b + 6], q7  = vectors[b + 7],  q8  = vectors[b + 8];
            float q9  = vectors[b + 9], q10 = vectors[b + 10], q11 = vectors[b + 11];
            float q12 = vectors[b + 12], q13 = vectors[b + 13];

            int bestK = 0;
            float bestDist = Float.MAX_VALUE;

            for (int c = 0; c < k; c++) {
                int cb = c * Dataset.DIMS;
                float x, dist;
                x = centroids[cb     ] - q0;  dist  = x * x;
                x = centroids[cb +  1] - q1;  dist += x * x;
                x = centroids[cb +  2] - q2;  dist += x * x;
                x = centroids[cb +  3] - q3;  dist += x * x;
                x = centroids[cb +  4] - q4;  dist += x * x;
                x = centroids[cb +  5] - q5;  dist += x * x;
                x = centroids[cb +  6] - q6;  dist += x * x;
                x = centroids[cb +  7] - q7;  dist += x * x;
                x = centroids[cb +  8] - q8;  dist += x * x;
                x = centroids[cb +  9] - q9;  dist += x * x;
                x = centroids[cb + 10] - q10; dist += x * x;
                x = centroids[cb + 11] - q11; dist += x * x;
                x = centroids[cb + 12] - q12; dist += x * x;
                x = centroids[cb + 13] - q13; dist += x * x;

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

    /* =================================================================== *
     * Update step                                                          *
     * =================================================================== */
    private static void updateCentroids(float[] vectors, int n, int dims,
                                        int[] assignments, float[] centroids, int k,
                                        SplittableRandom rng) {
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
                int rndIdx = rng.nextInt(n);
                System.arraycopy(vectors, rndIdx * dims, centroids, c * dims, dims);
            }
        }
    }

    private static float squaredDistance(float[] a, int aOff, float[] b, int bOff) {
        float sum = 0;
        for (int d = 0; d < Dataset.DIMS; d++) {
            float diff = a[aOff + d] - b[bOff + d];
            sum += diff * diff;
        }
        return sum;
    }
}
