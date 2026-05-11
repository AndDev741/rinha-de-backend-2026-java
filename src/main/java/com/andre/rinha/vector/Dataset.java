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
 * v7 dataset — int16 quantized + IVF + per-cluster bounding boxes.
 *
 * Reads five files produced by {@link com.andre.rinha.prep.DatasetBuilder}:
 *   vectors-i16.bin       →  N × 14 int16 LE (× 10000 scaling), reordered by cluster
 *   centroids-i16.bin     →  K × 14 int16 LE (same scaling)
 *   bbox.bin              →  K × 14 + K × 14 int16 LE (mins then maxs per cluster)
 *   cluster_offsets.bin   →  K+1 int32 LE
 *   labels.bin            →  packed bitset
 *
 * Memory profile:
 *   vectors[]      ≈  84 MB  (2 × 14 × 3M)
 *   centroids      ≈   7 KB
 *   bbox_min/max   ≈  14 KB  (fits in L1 cache → bbox repair is near-free)
 *   labels         ≈ 375 KB
 *
 * Quantization (lossless for our purposes):
 *   short(v) = round(v × 10000)
 *   For float values in [-1, 1], maps to int16 [-10000, 10000], safely
 *   inside short's range. ~0.0001 quantization step preserves ~4 decimal
 *   digits — effectively float-equivalent precision.
 *
 * Squared int16 distance is exactly the float32 squared distance × 1e8,
 * so ordering is preserved with no per-dim weighting tricks.
 */
public final class Dataset {

    public static final int DIMS = Vectorizer.DIMS;

    /** Quantization scale: float × SCALE → int16 (must match DatasetBuilder). */
    public static final int SCALE = 10_000;

    /** Tail padding bytes (in shorts) — kept for layout symmetry. */
    private static final int TAIL_PAD = 8;

    private final short[] vectors;     // length = count * DIMS + TAIL_PAD
    private final BitSet  labels;
    private final int     count;
    private final short[] centroids;   // length = k * DIMS
    private final int     k;
    private final int[]   clusterOffsets;
    private final short[] bboxMin;     // length = k * DIMS
    private final short[] bboxMax;     // length = k * DIMS

    private Dataset(short[] vectors, BitSet labels, int count,
                    short[] centroids, int k, int[] clusterOffsets,
                    short[] bboxMin, short[] bboxMax) {
        this.vectors = vectors;
        this.labels = labels;
        this.count = count;
        this.centroids = centroids;
        this.k = k;
        this.clusterOffsets = clusterOffsets;
        this.bboxMin = bboxMin;
        this.bboxMax = bboxMax;
    }

    public int       count()                  { return count; }
    public short[]   vectors()                { return vectors; }
    public boolean   isFraud(int i)           { return labels.get(i); }
    public int       k()                      { return k; }
    public short[]   centroids()              { return centroids; }
    public int       clusterStart(int c)      { return clusterOffsets[c]; }
    public int       clusterEnd(int c)        { return clusterOffsets[c + 1]; }
    public short[]   bboxMin()                { return bboxMin; }
    public short[]   bboxMax()                { return bboxMax; }

    /**
     * Quantize a float[14] query into a short[14] using the global × 10000
     * scaling. No per-dim min/max needed.
     */
    public static void quantize(float[] in, short[] out) {
        for (int d = 0; d < DIMS; d++) {
            int q = Math.round(in[d] * SCALE);
            if (q < Short.MIN_VALUE) q = Short.MIN_VALUE;
            if (q > Short.MAX_VALUE) q = Short.MAX_VALUE;
            out[d] = (short) q;
        }
    }

    /**
     * Build a Dataset directly from in-memory arrays. Used by tests so they
     * don't have to roundtrip through DatasetBuilder.
     */
    public static Dataset fromArrays(short[] vectors, BitSet labels,
                                     short[] centroids, int[] clusterOffsets,
                                     short[] bboxMin, short[] bboxMax) {
        if (vectors.length % DIMS != 0) {
            throw new IllegalArgumentException("vectors.length must be a multiple of " + DIMS);
        }
        int count = vectors.length / DIMS;
        if (centroids.length % DIMS != 0) {
            throw new IllegalArgumentException("centroids.length must be a multiple of " + DIMS);
        }
        int kClusters = centroids.length / DIMS;
        if (clusterOffsets.length != kClusters + 1) {
            throw new IllegalArgumentException("clusterOffsets.length must be K+1");
        }
        if (clusterOffsets[kClusters] != count) {
            throw new IllegalArgumentException("clusterOffsets[K] must equal count");
        }
        if (bboxMin.length != kClusters * DIMS || bboxMax.length != kClusters * DIMS) {
            throw new IllegalArgumentException("bbox arrays must each have length K*DIMS");
        }
        short[] padded = new short[vectors.length + TAIL_PAD];
        System.arraycopy(vectors, 0, padded, 0, vectors.length);
        return new Dataset(padded, labels, count, centroids, kClusters, clusterOffsets,
                bboxMin, bboxMax);
    }

    /** Loads all five v7 files from a directory. */
    public static Dataset load(Path dir) throws IOException {
        Path vectorsPath   = dir.resolve("vectors-i16.bin");
        Path centroidsPath = dir.resolve("centroids-i16.bin");
        Path bboxPath      = dir.resolve("bbox.bin");
        Path offsetsPath   = dir.resolve("cluster_offsets.bin");
        Path labelsPath    = dir.resolve("labels.bin");

        // ---- vectors ----
        long vectorsSize = Files.size(vectorsPath);
        if (vectorsSize % (DIMS * 2L) != 0) {
            throw new IOException("vectors-i16.bin size " + vectorsSize
                    + " is not a multiple of " + (DIMS * 2));
        }
        int count = (int) (vectorsSize / (DIMS * 2L));
        short[] vectors = new short[count * DIMS + TAIL_PAD];
        try (FileChannel ch = FileChannel.open(vectorsPath, StandardOpenOption.READ)) {
            MappedByteBuffer mbb = ch.map(FileChannel.MapMode.READ_ONLY, 0, vectorsSize);
            mbb.order(ByteOrder.LITTLE_ENDIAN);
            mbb.asShortBuffer().get(vectors, 0, count * DIMS);
        }

        // ---- centroids ----
        byte[] centroidBytes = Files.readAllBytes(centroidsPath);
        if (centroidBytes.length % (DIMS * 2) != 0) {
            throw new IOException("centroids-i16.bin size " + centroidBytes.length
                    + " is not a multiple of " + (DIMS * 2));
        }
        int kClusters = centroidBytes.length / (DIMS * 2);
        short[] centroids = new short[kClusters * DIMS];
        ByteBuffer.wrap(centroidBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(centroids);

        // ---- bbox ----
        byte[] bboxBytes = Files.readAllBytes(bboxPath);
        int expectedBbox = kClusters * DIMS * 2 * 2;  // K*DIMS * 2 (min,max) * 2 bytes
        if (bboxBytes.length != expectedBbox) {
            throw new IOException("bbox.bin size " + bboxBytes.length
                    + ", expected " + expectedBbox);
        }
        short[] bboxAll = new short[kClusters * DIMS * 2];
        ByteBuffer.wrap(bboxBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(bboxAll);
        short[] bboxMin = new short[kClusters * DIMS];
        short[] bboxMax = new short[kClusters * DIMS];
        System.arraycopy(bboxAll, 0,                  bboxMin, 0, kClusters * DIMS);
        System.arraycopy(bboxAll, kClusters * DIMS,   bboxMax, 0, kClusters * DIMS);

        // ---- offsets ----
        byte[] offsetBytes = Files.readAllBytes(offsetsPath);
        if (offsetBytes.length != (kClusters + 1) * 4) {
            throw new IOException("cluster_offsets.bin size " + offsetBytes.length
                    + ", expected " + ((kClusters + 1) * 4));
        }
        int[] offsets = new int[kClusters + 1];
        ByteBuffer.wrap(offsetBytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(offsets);
        if (offsets[kClusters] != count) {
            throw new IOException("cluster_offsets[K] = " + offsets[kClusters]
                    + " does not match vector count " + count);
        }

        // ---- labels ----
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

        return new Dataset(vectors, labels, count, centroids, kClusters, offsets,
                bboxMin, bboxMax);
    }
}
