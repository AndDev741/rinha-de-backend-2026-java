package com.andre.rinha.vector;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * v4 brute-force k-NN over the int8 quantized Dataset.
 *
 * Pipeline per request:
 *   1. Quantize the float query into byte[14] using Dataset.quantize.
 *   2. Walk through all N reference vectors. For each:
 *      - load 8 bytes via ByteVector
 *      - sign-extend to FloatVector via B2F conversion (one VPMOVSXBD +
 *        one VCVTDQ2PS in machine code)
 *      - sub from preloaded float query, FMA accumulate sum + diff*diff
 *      - reduceLanes for the squared distance
 *   3. Maintain top-K via a size-5 max-heap with float distances.
 *   4. Return fraud_score = (#fraud among top-K) / K.
 *
 * Why byte storage but float compute (the "hybrid" approach):
 *
 *   We tried pure-int math (byte → int → sub → mul → add). Result on Docker
 *   was -6000 with 711 served. The pure-float v3 path served 1188-1605.
 *   Pure int8 SIMD was *slower* than v3 float32 SIMD because:
 *     - VPMULLD (int multiply) has worse throughput than VFMADD231PS (FMA)
 *     - One FMA replaces a separate mul + add
 *     - Float pipeline has more execution ports on Haswell than int multiply
 *
 *   Storing as bytes still gives us 4× memory bandwidth savings (8 bytes
 *   read per chunk vs 32 bytes for float[]). Converting on the fly to
 *   FloatVector lets us reuse the FMA-friendly v3 inner loop. We get the
 *   memory win AND keep the fast compute path.
 *
 *   Distance ordering correctness: with global min/max, converting bytes
 *   directly to floats (i.e., NOT dequantizing back to original units)
 *   produces a distance that is a constant multiple of the float32 distance.
 *   The k-NN top-5 set is identical up to rounding noise.
 *
 * SIMD widths (Mac Mini 2014 = Haswell = AVX2):
 *   ByteVector .SPECIES_64  →  8 byte lanes  (64-bit register)
 *   FloatVector.SPECIES_256 →  8 float lanes (256-bit register)
 *   B→F conversion at part=0 maps lane-for-lane (sign extension + int→float).
 */
public final class KnnSearcher {

    public static final int K = 5;

    /** 8 byte lanes per chunk — matches FSPECIES lane count for 1:1 conversion. */
    private static final VectorSpecies<Byte>  BSPECIES = ByteVector.SPECIES_64;
    /** 8 float lanes per chunk — accumulator. */
    private static final VectorSpecies<Float> FSPECIES = FloatVector.SPECIES_256;

    /** Last byte index where a full SIMD chunk fits. With DIMS=14 and 8 lanes, this is 8. */
    private static final int LOOP_BOUND = BSPECIES.loopBound(Dataset.DIMS);

    /** Mask for the tail load. Lanes 0..(DIMS-LOOP_BOUND-1) are unmasked. */
    private static final VectorMask<Byte> TAIL_MASK = BSPECIES.indexInRange(LOOP_BOUND, Dataset.DIMS);

    private final Dataset dataset;

    // Reusable buffers — KnnSearcher is stateful per instance, one per thread.
    private final byte[]  qBytes   = new byte[Dataset.DIMS];
    private final float[] heapDist = new float[K];
    private final int[]   heapIdx  = new int[K];

    public KnnSearcher(Dataset dataset) {
        this.dataset = dataset;
    }

    public static String simdInfo() {
        return "B" + BSPECIES.length() + " → F" + FSPECIES.length()
                + " (BSPECIES=" + BSPECIES + ", FSPECIES=" + FSPECIES
                + ", LOOP_BOUND=" + LOOP_BOUND + ")";
    }

    /** Computes fraud_score = (#frauds among K nearest neighbors) / K. */
    public float fraudScore(float[] query) {
        // 1. Quantize the query to int8 using the dataset's mins/maxs.
        dataset.quantize(query, qBytes);

        final byte[] vecs = dataset.vectors();
        final int count = dataset.count();

        // 2. Pre-load the query as FloatVectors. qFull is null when the SIMD
        //    width is wider than DIMS (won't happen on AVX2 with DIMS=14).
        final FloatVector qFull = LOOP_BOUND > 0
                ? widenByteToFloat(ByteVector.fromArray(BSPECIES, qBytes, 0))
                : null;
        final FloatVector qTail = widenByteToFloat(
                ByteVector.fromArray(BSPECIES, qBytes, LOOP_BOUND, TAIL_MASK));

        // 3. Initialize the heap with the first K vectors.
        for (int i = 0; i < K; i++) {
            heapDist[i] = squaredDistance(qFull, qTail, vecs, i * Dataset.DIMS);
            heapIdx[i] = i;
        }
        for (int i = K / 2 - 1; i >= 0; i--) siftDown(i);

        // 4. Hot loop. No early-exit branch — kills SIMD speculation.
        for (int i = K; i < count; i++) {
            float d = squaredDistance(qFull, qTail, vecs, i * Dataset.DIMS);
            if (d < heapDist[0]) {
                heapDist[0] = d;
                heapIdx[0] = i;
                siftDown(0);
            }
        }

        return countFrauds() / (float) K;
    }

    /**
     * Squared L2 distance (in byte-as-float space).
     *
     * For DIMS=14 on AVX2:
     *   chunk 0 — full width:   diff = qFull - widen(vecs[off..off+7])
     *   tail    — masked load:  diff = qTail - widen(vecs[off+8..off+13])
     *   FMA accumulate, reduce.
     */
    private static float squaredDistance(FloatVector qFull, FloatVector qTail,
                                         byte[] vecs, int off) {
        FloatVector sum = FloatVector.zero(FSPECIES);

        if (qFull != null) {
            for (int i = 0; i < LOOP_BOUND; i += BSPECIES.length()) {
                FloatVector v = widenByteToFloat(ByteVector.fromArray(BSPECIES, vecs, off + i));
                FloatVector diff = qFull.sub(v);
                sum = diff.fma(diff, sum);  // sum + diff * diff in one rounding step
            }
        }

        FloatVector vTail = widenByteToFloat(
                ByteVector.fromArray(BSPECIES, vecs, off + LOOP_BOUND, TAIL_MASK));
        FloatVector diffTail = qTail.sub(vTail);
        sum = diffTail.fma(diffTail, sum);

        return sum.reduceLanes(VectorOperators.ADD);
    }

    /**
     * Sign-extending widen of an 8-lane ByteVector to an 8-lane FloatVector.
     * VectorOperators.B2F first sign-extends each byte to an int, then
     * converts that int to a float — both happen as a single conversion
     * primitive on x86 (VPMOVSXBD + VCVTDQ2PS).
     */
    private static FloatVector widenByteToFloat(ByteVector b) {
        return (FloatVector) b.convertShape(VectorOperators.B2F, FSPECIES, 0);
    }

    /* -------------------- Shared utilities -------------------- */

    private int countFrauds() {
        int n = 0;
        for (int i = 0; i < K; i++) {
            if (dataset.isFraud(heapIdx[i])) n++;
        }
        return n;
    }

    /** Max-heap sift-down on float distances. */
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
