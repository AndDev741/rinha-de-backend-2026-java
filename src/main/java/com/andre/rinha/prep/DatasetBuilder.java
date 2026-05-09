package com.andre.rinha.prep;

import java.io.BufferedInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;

/**
 * Converts references.json.gz into a flat binary format:
 *
 *   vectors.bin  →  3,000,000 × 14 × 4 bytes  (float32, little-endian)
 *                   = ~168 MB
 *   labels.bin   →  N bytes where each bit indicates fraud(1) / legit(0)
 *                   N = ceil(3,000,000 / 8) ≈ 375 KB
 *
 * Why this format:
 * - vectors.bin is mmap-friendly: the kernel maps it directly into memory
 *   with no parsing.
 * - little-endian is native for x86_64 and ARM (test box is x86_64).
 * - labels as a bitset saves 8x vs byte[] and fits entirely in L2 cache.
 *
 * How to run:
 *   mvn -q compile exec:java \
 *       -Dexec.mainClass=com.andre.rinha.prep.DatasetBuilder \
 *       -Dexec.args="/path/references.json.gz /path/output"
 *
 * Or directly:
 *   java -cp target/classes com.andre.rinha.prep.DatasetBuilder \
 *       references.json.gz /tmp/data
 *
 * Output:
 *   /tmp/data/vectors.bin
 *   /tmp/data/labels.bin
 *   /tmp/data/meta.txt   (count + fraud count, for sanity check)
 */
public final class DatasetBuilder {

    private static final int DIMS = 14;
    private static final int CHUNK = 1_000_000; // log every 1M records

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: DatasetBuilder <references.json.gz> <outputDir>");
            System.exit(1);
        }
        Path input = Path.of(args[0]);
        Path outDir = Path.of(args[1]);
        outDir.toFile().mkdirs();

        Path vectorsPath = outDir.resolve("vectors.bin");
        Path labelsPath  = outDir.resolve("labels.bin");
        Path metaPath    = outDir.resolve("meta.txt");

        long start = System.currentTimeMillis();

        // Streaming: it doesn't all fit in memory in inflated JSON form.
        // Read gzip → JSON → one entry at a time → write directly to FileChannel.
        try (InputStream raw = new FileInputStream(input.toFile());
             InputStream gz = new GZIPInputStream(new BufferedInputStream(raw, 1 << 20));
             DataOutputStream labelsOut = new DataOutputStream(new FileOutputStream(labelsPath.toFile()));
             FileChannel vectorsCh = new FileOutputStream(vectorsPath.toFile()).getChannel()) {

            // Reusable buffer to write each vector (14 floats = 56 bytes).
            ByteBuffer vecBuf = ByteBuffer.allocate(DIMS * 4).order(ByteOrder.LITTLE_ENDIAN);

            // Bitset accumulator for the labels. 8 bits per byte.
            int byteAccumulator = 0;
            int bitsInAccumulator = 0;

            JsonStream js = new JsonStream(gz);
            js.expect('[');

            long count = 0;
            long fraudCount = 0;

            while (true) {
                js.skipWs();
                int c = js.peek();
                if (c == ']') break;
                if (c == ',') { js.read(); js.skipWs(); }

                // Each entry: { "vector": [...], "label": "fraud"|"legit" }
                js.expect('{');
                float[] vec = new float[DIMS];
                boolean isFraud = false;
                int dim = 0;

                while (true) {
                    js.skipWs();
                    if (js.peek() == '}') { js.read(); break; }
                    if (js.peek() == ',') { js.read(); js.skipWs(); }

                    String key = js.readString();
                    js.skipWs(); js.expect(':'); js.skipWs();

                    if (key.equals("vector")) {
                        js.expect('[');
                        for (int i = 0; i < DIMS; i++) {
                            js.skipWs();
                            if (i > 0) { js.expect(','); js.skipWs(); }
                            vec[i] = (float) js.readDouble();
                            dim++;
                        }
                        js.skipWs();
                        js.expect(']');
                    } else if (key.equals("label")) {
                        String v = js.readString();
                        isFraud = v.equals("fraud");
                    } else {
                        js.skipValue();
                    }
                }

                if (dim != DIMS) {
                    throw new IllegalStateException("Row " + count + " has " + dim + " dimensions");
                }

                // Write vector (56 bytes) to the output channel.
                vecBuf.clear();
                for (int i = 0; i < DIMS; i++) vecBuf.putFloat(vec[i]);
                vecBuf.flip();
                vectorsCh.write(vecBuf);

                // Accumulate the label bit.
                byteAccumulator |= (isFraud ? 1 : 0) << bitsInAccumulator;
                bitsInAccumulator++;
                if (bitsInAccumulator == 8) {
                    labelsOut.write(byteAccumulator);
                    byteAccumulator = 0;
                    bitsInAccumulator = 0;
                }

                if (isFraud) fraudCount++;
                count++;
                if (count % CHUNK == 0) {
                    System.out.printf("[builder] %d processed (%.1fs)%n",
                            count, (System.currentTimeMillis() - start) / 1000.0);
                }
            }
            // Flush the last partial byte of the bitset.
            if (bitsInAccumulator > 0) {
                labelsOut.write(byteAccumulator);
            }

            js.expect(']');

            // meta.txt → count + fraud count (useful for validation)
            String meta = "count=" + count + "\nfraud=" + fraudCount + "\n";
            Files.writeString(metaPath, meta);

            System.out.printf("[builder] OK: %d vectors, %d frauds (%.2f%%) in %.1fs%n",
                    count, fraudCount, fraudCount * 100.0 / count,
                    (System.currentTimeMillis() - start) / 1000.0);
        }
    }

    /* -------------------- Minimal streaming JSON tokenizer -------------------- */

    /**
     * Streaming JSON tokenizer over an InputStream. Doesn't load everything
     * into memory. Simplified version of JsonReader, adapted for incremental
     * parsing.
     */
    private static final class JsonStream {
        private final InputStream in;
        private int peeked = -2;

        JsonStream(InputStream in) { this.in = in; }

        int peek() throws IOException {
            if (peeked == -2) peeked = in.read();
            return peeked;
        }

        int read() throws IOException {
            int c = peek();
            peeked = -2;
            return c;
        }

        void expect(int c) throws IOException {
            int got = read();
            if (got != c) {
                throw new IllegalStateException("Expected '" + (char) c + "', got '" + (got == -1 ? "EOF" : (char) got) + "'");
            }
        }

        void skipWs() throws IOException {
            while (true) {
                int c = peek();
                if (c == ' ' || c == '\n' || c == '\r' || c == '\t') read();
                else return;
            }
        }

        String readString() throws IOException {
            expect('"');
            StringBuilder sb = new StringBuilder(16);
            while (true) {
                int c = read();
                if (c == '"') break;
                if (c == '\\') {
                    int esc = read();
                    sb.append((char) esc); // dataset doesn't use unicode escapes
                } else {
                    sb.append((char) c);
                }
            }
            return sb.toString();
        }

        double readDouble() throws IOException {
            StringBuilder sb = new StringBuilder(16);
            while (true) {
                int c = peek();
                if (c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' || (c >= '0' && c <= '9')) {
                    sb.append((char) read());
                } else break;
            }
            return Double.parseDouble(sb.toString());
        }

        void skipValue() throws IOException {
            skipWs();
            int c = peek();
            switch (c) {
                case '"' -> readString();
                case 't', 'f', 'n' -> { while (Character.isLetter(peek())) read(); }
                case '{' -> {
                    int depth = 0;
                    do {
                        int x = read();
                        if (x == '{') depth++;
                        else if (x == '}') depth--;
                        else if (x == '"') { while (read() != '"'); }
                    } while (depth > 0);
                }
                case '[' -> {
                    int depth = 0;
                    do {
                        int x = read();
                        if (x == '[') depth++;
                        else if (x == ']') depth--;
                        else if (x == '"') { while (read() != '"'); }
                    } while (depth > 0);
                }
                default -> readDouble();
            }
        }
    }
}
