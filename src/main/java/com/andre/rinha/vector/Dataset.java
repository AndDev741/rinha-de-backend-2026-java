package com.andre.rinha.vector;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.BitSet;

/**
 * v4 dataset — int8 quantized.
 *
 * Reads three files produced by {@link com.andre.rinha.prep.DatasetBuilder}:
 *   vectors-i8.bin  →  N × 14 signed bytes (the quantized vectors)
 *   scales.bin      →  mins[14] + maxs[14] as float32 LE
 *   labels.bin      →  packed bitset, bit i = 1 if vector i is fraud
 *
 * Memory profile:
 *   vectors[]  ≈ 42 MB  (down from 168 MB in v1/v2/v3 — fits in L3 cache!)
 *   labels     ≈ 375 KB
 *   mins/maxs  = 112 B  (precomputed scaleFactor[] of 56 B for the hot path)
 *
 * Same heap-copy strategy as v1: we mmap the file then bulk-copy into a
 * heap byte[]. The reasons (no virtual call, no bounds check, JIT-friendly
 * layout) hold even more strongly for int8 because every memory operation
 * counts proportionally more when the data is small.
 *
 * Quantization invariant:
 *   For each dim d the byte b corresponds to the float
 *     f = ((b + 128) / 255) * (maxs[d] - mins[d]) + mins[d]
 *   Equivalently, the original quantization rule is
 *     b = round((f - mins[d]) * 255 / (maxs[d] - mins[d])) - 128
 *   Both the dataset and any query are quantized with the SAME mins/maxs,
 *   so squared int8 differences are a (per-dim weighted) approximation of
 *   the float32 squared L2 distance. Ordering agreement is verified by the
 *   parity test against synthetic ground truth.
 */
public final class Dataset {

    public static final int DIMS = Vectorizer.DIMS;

    /**
     * Bytes appended to the end of the {@code vectors} array beyond the real
     * data. Protects masked SIMD tail loads on the last record. 16 is enough
     * for any current SIMD width (AVX-512 is 64 bytes wide but reads at most
     * 16 bytes past LOOP_BOUND for DIMS=14).
     */
    private static final int TAIL_PAD = 16;

    private final byte[] vectors;       // size = count * DIMS
    private final BitSet labels;
    private final int count;

    // Per-dimension quantization constants. Kept tiny (14 floats each) so
    // they live in L1 alongside the hot loop's working set.
    private final float[] mins;
    private final float[] maxs;
    /** scaleFactor[d] = 255 / (maxs[d] - mins[d]). Precomputed to skip one div per quantize. */
    private final float[] scaleFactor;

    private Dataset(byte[] vectors, BitSet labels, int count,
                    float[] mins, float[] maxs, float[] scaleFactor) {
        this.vectors = vectors;
        this.labels = labels;
        this.count = count;
        this.mins = mins;
        this.maxs = maxs;
        this.scaleFactor = scaleFactor;
    }

    public int count() { return count; }
    public byte[] vectors() { return vectors; }
    public boolean isFraud(int i) { return labels.get(i); }

    public float[] mins() { return mins; }
    public float[] maxs() { return maxs; }

    /**
     * Quantize a float[14] query into a byte[14] using the dataset's scales.
     * Caller passes the destination so we don't allocate per-request.
     *
     * The math mirrors what DatasetBuilder did at build time. Out-of-range
     * inputs are clamped — this happens in practice when the live request
     * is more extreme than anything seen in the reference set.
     */
    public void quantize(float[] in, byte[] out) {
        for (int d = 0; d < DIMS; d++) {
            int q = Math.round((in[d] - mins[d]) * scaleFactor[d]) - 128;
            if (q < -128) q = -128;
            if (q >  127) q =  127;
            out[d] = (byte) q;
        }
    }

    /**
     * Build a Dataset directly from in-memory arrays. Used by tests so they
     * don't have to roundtrip through DatasetBuilder.
     *
     * The caller passes {@code vectors} of length {@code count * DIMS} (no
     * padding). We internally allocate a padded copy so SIMD tail loads at
     * the last record stay in bounds.
     */
    public static Dataset fromArrays(byte[] vectors, BitSet labels,
                                     float[] mins, float[] maxs) {
        if (vectors.length % DIMS != 0) {
            throw new IllegalArgumentException("vectors.length must be a multiple of " + DIMS);
        }
        if (mins.length != DIMS || maxs.length != DIMS) {
            throw new IllegalArgumentException("mins/maxs must each have length " + DIMS);
        }
        int count = vectors.length / DIMS;
        byte[] padded = new byte[vectors.length + TAIL_PAD];
        System.arraycopy(vectors, 0, padded, 0, vectors.length);
        return new Dataset(padded, labels, count, mins, maxs, computeScale(mins, maxs));
    }

    /**
     * Loads vectors-i8.bin + scales.bin + labels.bin from a directory.
     */
    public static Dataset load(Path dir) throws IOException {
        Path vectorsPath = dir.resolve("vectors-i8.bin");
        Path scalesPath  = dir.resolve("scales.bin");
        Path labelsPath  = dir.resolve("labels.bin");

        long vectorsSize = Files.size(vectorsPath);
        if (vectorsSize % DIMS != 0) {
            throw new IOException("vectors-i8.bin size " + vectorsSize + " is not a multiple of " + DIMS);
        }
        int count = (int) (vectorsSize / DIMS);

        // Load vectors via mmap then bulk-copy to heap byte[].
        // The +TAIL_PAD bytes are zero-padding that protects masked SIMD tail
        // loads at the very last record from running off the end of the array.
        // The Java Vector API spec says masked lanes never read out-of-bounds,
        // but a defensive 16 bytes is free insurance and makes the math obvious.
        byte[] vectors = new byte[(int) vectorsSize + TAIL_PAD];
        try (FileChannel ch = FileChannel.open(vectorsPath, StandardOpenOption.READ)) {
            MappedByteBuffer mbb = ch.map(FileChannel.MapMode.READ_ONLY, 0, vectorsSize);
            mbb.get(vectors, 0, (int) vectorsSize);
        }

        // Load scales — tiny, just read the whole file.
        byte[] scaleBytes = Files.readAllBytes(scalesPath);
        if (scaleBytes.length != DIMS * 2 * 4) {
            throw new IOException("scales.bin has unexpected size " + scaleBytes.length
                    + ", expected " + (DIMS * 2 * 4));
        }
        ByteBuffer sb = ByteBuffer.wrap(scaleBytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] mins = new float[DIMS];
        float[] maxs = new float[DIMS];
        for (int d = 0; d < DIMS; d++) mins[d] = sb.getFloat();
        for (int d = 0; d < DIMS; d++) maxs[d] = sb.getFloat();

        // Labels — same layout as v1/v2/v3.
        byte[] labelBytes = Files.readAllBytes(labelsPath);
        BitSet labels = new BitSet(count);
        for (int byteIdx = 0; byteIdx < labelBytes.length; byteIdx++) {
            int b = labelBytes[byteIdx] & 0xFF;
            for (int bit = 0; bit < 8; bit++) {
                int idx = byteIdx * 8 + bit;
                if (idx >= count) break;
                if ((b & (1 << bit)) != 0) labels.set(idx);
            }
        }

        return new Dataset(vectors, labels, count, mins, maxs, computeScale(mins, maxs));
    }

    private static float[] computeScale(float[] mins, float[] maxs) {
        float[] scale = new float[DIMS];
        for (int d = 0; d < DIMS; d++) {
            float range = maxs[d] - mins[d];
            scale[d] = (range > 0f) ? (255f / range) : 0f;
        }
        return scale;
    }
}
