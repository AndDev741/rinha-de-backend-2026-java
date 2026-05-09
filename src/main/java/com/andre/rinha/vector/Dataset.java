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
 * Loads the reference dataset produced by DatasetBuilder.
 *
 * Important decision:
 *   - We read vectors.bin via mmap, but COPY into a heap-resident float[].
 *
 * Why copy:
 *   - MappedByteBuffer.getFloat() in a loop is ~3x slower than direct float[i]:
 *     extra bounds checks, virtual call, no JIT auto-vectorization.
 *   - The dataset is fixed and fits in heap (168 MB with -Xmx256m).
 *   - Simplicity > RAM savings at this stage. In step 4 (int8 quantization)
 *     we shrink to 42 MB and reconsider.
 *
 * Structure:
 *   - vectors[]   : 3M × 14 = 42M floats, linear layout (Struct of Arrays
 *                   isn't worth it for 14 dims — they all fit in one 64B
 *                   cache line along with the offset).
 *   - labels      : BitSet with bit i = 1 if vector i is fraud. ~375 KB.
 */
public final class Dataset {

    public static final int DIMS = Vectorizer.DIMS;

    private final float[] vectors;   // size = count * DIMS
    private final BitSet labels;     // bit i = fraud
    private final int count;

    private Dataset(float[] vectors, BitSet labels, int count) {
        this.vectors = vectors;
        this.labels = labels;
        this.count = count;
    }

    public int count() { return count; }
    public float[] vectors() { return vectors; }
    public boolean isFraud(int i) { return labels.get(i); }

    /**
     * Loads vectors.bin + labels.bin from a directory.
     *
     * @param dir directory containing vectors.bin and labels.bin
     */
    public static Dataset load(Path dir) throws IOException {
        Path vectorsPath = dir.resolve("vectors.bin");
        Path labelsPath  = dir.resolve("labels.bin");

        long vectorsSize = Files.size(vectorsPath);
        if (vectorsSize % (DIMS * 4L) != 0) {
            throw new IOException("vectors.bin has invalid size: " + vectorsSize);
        }
        int count = (int) (vectorsSize / (DIMS * 4L));

        // Read-only mmap. The kernel handles paging on demand.
        // Since we're going to read everything sequentially, this pre-populates
        // the OS page cache — good for the next instance opening the same file.
        float[] vectors = new float[count * DIMS];
        try (FileChannel ch = FileChannel.open(vectorsPath, StandardOpenOption.READ)) {
            MappedByteBuffer mbb = ch.map(FileChannel.MapMode.READ_ONLY, 0, vectorsSize);
            mbb.order(ByteOrder.LITTLE_ENDIAN);
            // asFloatBuffer + bulk get is the fastest path the JDK exposes.
            mbb.asFloatBuffer().get(vectors);
        }

        // Labels: read the whole file into a byte[] and build the BitSet.
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

        return new Dataset(vectors, labels, count);
    }

    /**
     * Alternative version that keeps the ByteBuffer mmap'd — useful when
     * memory gets tight (not used yet, but kept here for later testing).
     */
    @SuppressWarnings("unused")
    public static ByteBuffer mmapOnly(Path vectorsPath) throws IOException {
        try (FileChannel ch = FileChannel.open(vectorsPath, StandardOpenOption.READ)) {
            return ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size())
                    .order(ByteOrder.LITTLE_ENDIAN);
        }
    }
}
