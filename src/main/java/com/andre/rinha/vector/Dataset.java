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
 * v5 dataset — int8 quantized + IVF (Inverted File Index) layout.
 *
 * Reads five files produced by {@link com.andre.rinha.prep.DatasetBuilder}:
 *   vectors-i8.bin       →  N × 14 signed bytes (quantized vectors,
 *                            REORDERED so cluster 0 comes first, then 1, ...)
 *   scales.bin           →  mins[14] + maxs[14] as float32 LE
 *   labels.bin           →  packed bitset, bit i = 1 if vector i is fraud
 *                            (also reordered to match the new vector order)
 *   centroids.bin        →  K × 14 float32 LE — each cluster's centroid
 *   cluster_offsets.bin  →  K+1 int32 LE — start of each cluster in the
 *                            reordered vectors array (offsets[K] = total N)
 *
 * Memory profile:
 *   vectors[]    ≈ 42 MB   (int8 storage)
 *   labels       ≈ 375 KB
 *   centroids    ≈ 14 KB   (256 × 14 × 4 — fits comfortably in L1 cache)
 *   offsets      ≈ 1 KB
 *   mins/maxs    = 112 B
 *
 * Quantization invariant (unchanged from v4):
 *   For each dim d the byte b corresponds to the float
 *     f = ((b + 128) / 255) * (maxs[d] - mins[d]) + mins[d]
 *   Both the dataset and any query are quantized with the SAME mins/maxs,
 *   so squared int8 differences are a constant scaling of float32 distance.
 */
public final class Dataset {

    public static final int DIMS = Vectorizer.DIMS;

    private static final int TAIL_PAD = 16;

    private final byte[] vectors;
    private final BitSet labels;
    private final int count;
    private final float[] mins;
    private final float[] maxs;
    private final float[] scaleFactor;

    /** v5: cluster centroids, flat layout [K * DIMS] in float32. */
    private final float[] centroids;
    /** v5: number of clusters K. */
    private final int k;
    /**
     * v5: cluster offsets, length K+1.
     * Cluster c spans vector indices [offsets[c], offsets[c+1]).
     */
    private final int[] clusterOffsets;

    private Dataset(byte[] vectors, BitSet labels, int count,
                    float[] mins, float[] maxs, float[] scaleFactor,
                    float[] centroids, int k, int[] clusterOffsets) {
        this.vectors = vectors;
        this.labels = labels;
        this.count = count;
        this.mins = mins;
        this.maxs = maxs;
        this.scaleFactor = scaleFactor;
        this.centroids = centroids;
        this.k = k;
        this.clusterOffsets = clusterOffsets;
    }

    public int count() { return count; }
    public byte[] vectors() { return vectors; }
    public boolean isFraud(int i) { return labels.get(i); }
    public float[] mins() { return mins; }
    public float[] maxs() { return maxs; }

    public int k() { return k; }
    public float[] centroids() { return centroids; }

    /** Inclusive start index (in vector units) of cluster {@code c}. */
    public int clusterStart(int c) { return clusterOffsets[c]; }
    /** Exclusive end index (in vector units) of cluster {@code c}. */
    public int clusterEnd(int c)   { return clusterOffsets[c + 1]; }

    /** Quantize a float[14] query into a byte[14] using the dataset's scales. */
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
     */
    public static Dataset fromArrays(byte[] vectors, BitSet labels,
                                     float[] mins, float[] maxs,
                                     float[] centroids, int[] clusterOffsets) {
        if (vectors.length % DIMS != 0) {
            throw new IllegalArgumentException("vectors.length must be a multiple of " + DIMS);
        }
        if (mins.length != DIMS || maxs.length != DIMS) {
            throw new IllegalArgumentException("mins/maxs must each have length " + DIMS);
        }
        if (centroids.length % DIMS != 0) {
            throw new IllegalArgumentException("centroids.length must be a multiple of " + DIMS);
        }
        int kClusters = centroids.length / DIMS;
        if (clusterOffsets.length != kClusters + 1) {
            throw new IllegalArgumentException(
                    "clusterOffsets.length must be K+1 = " + (kClusters + 1));
        }
        int count = vectors.length / DIMS;
        if (clusterOffsets[kClusters] != count) {
            throw new IllegalArgumentException(
                    "clusterOffsets[K] must equal count: " + clusterOffsets[kClusters] + " != " + count);
        }
        byte[] padded = new byte[vectors.length + TAIL_PAD];
        System.arraycopy(vectors, 0, padded, 0, vectors.length);
        return new Dataset(padded, labels, count, mins, maxs, computeScale(mins, maxs),
                centroids, kClusters, clusterOffsets);
    }

    /** Loads all five v5 files from a directory. */
    public static Dataset load(Path dir) throws IOException {
        Path vectorsPath   = dir.resolve("vectors-i8.bin");
        Path scalesPath    = dir.resolve("scales.bin");
        Path labelsPath    = dir.resolve("labels.bin");
        Path centroidsPath = dir.resolve("centroids.bin");
        Path offsetsPath   = dir.resolve("cluster_offsets.bin");

        // ---- vectors ----
        long vectorsSize = Files.size(vectorsPath);
        if (vectorsSize % DIMS != 0) {
            throw new IOException("vectors-i8.bin size " + vectorsSize
                    + " is not a multiple of " + DIMS);
        }
        int count = (int) (vectorsSize / DIMS);
        byte[] vectors = new byte[(int) vectorsSize + TAIL_PAD];
        try (FileChannel ch = FileChannel.open(vectorsPath, StandardOpenOption.READ)) {
            MappedByteBuffer mbb = ch.map(FileChannel.MapMode.READ_ONLY, 0, vectorsSize);
            mbb.get(vectors, 0, (int) vectorsSize);
        }

        // ---- scales ----
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

        // ---- centroids ----
        byte[] centroidBytes = Files.readAllBytes(centroidsPath);
        if (centroidBytes.length % (DIMS * 4) != 0) {
            throw new IOException("centroids.bin size " + centroidBytes.length
                    + " is not a multiple of " + (DIMS * 4));
        }
        int kClusters = centroidBytes.length / (DIMS * 4);
        ByteBuffer cb = ByteBuffer.wrap(centroidBytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] centroids = new float[kClusters * DIMS];
        for (int i = 0; i < centroids.length; i++) centroids[i] = cb.getFloat();

        // ---- offsets ----
        byte[] offsetBytes = Files.readAllBytes(offsetsPath);
        if (offsetBytes.length != (kClusters + 1) * 4) {
            throw new IOException("cluster_offsets.bin size " + offsetBytes.length
                    + " expected " + ((kClusters + 1) * 4));
        }
        ByteBuffer ob = ByteBuffer.wrap(offsetBytes).order(ByteOrder.LITTLE_ENDIAN);
        int[] offsets = new int[kClusters + 1];
        for (int i = 0; i < offsets.length; i++) offsets[i] = ob.getInt();
        if (offsets[kClusters] != count) {
            throw new IOException("cluster_offsets[K] = " + offsets[kClusters]
                    + " does not match vector count " + count);
        }

        return new Dataset(vectors, labels, count, mins, maxs, computeScale(mins, maxs),
                centroids, kClusters, offsets);
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
