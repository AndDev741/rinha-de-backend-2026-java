package com.andre.rinha.json;

import java.time.Instant;
import java.util.ArrayList;

/**
 * Hand-rolled JSON parser, specialized for the POST /fraud-score payload.
 *
 * How it works:
 * - Single cursor (this.pos) advancing byte by byte over the input array.
 * - "skipWs" skips spaces and line breaks. JSON is text-friendly.
 * - "expect(c)" requires a specific character (e.g., ',', ':', '{').
 * - For nested objects (transaction, customer, etc.), we step into '{' and
 *   read key:value pairs until we find '}'.
 * - We don't strictly validate JSON — we trust the challenge's fixed format.
 *   In real production this would be dangerous, but here it's a conscious
 *   choice for speed.
 *
 * Performance: zero allocation of intermediate maps/lists. Strings are
 * created only where necessary (merchant id, mcc, known_merchants).
 *
 * NOTE: this parser is not reusable (keeps state). Create one per request.
 * It's lightweight — only 2 fields (buf and pos). The goal is to avoid
 * synchronization.
 */
public final class JsonReader {

    private final byte[] buf;
    private int pos;

    public JsonReader(byte[] buf) {
        this.buf = buf;
        this.pos = 0;
    }

    public static Payload parse(byte[] body) {
        return new JsonReader(body).readPayload();
    }

    /* -------------------- Top-level -------------------- */

    private Payload readPayload() {
        // Initialize all fields with defaults — makes it easier to handle
        // arbitrary block ordering in the JSON and last_transaction: null.
        double txAmount = 0;
        int txInstallments = 0;
        long txRequestedAt = 0;
        double customerAvgAmount = 0;
        int customerTxCount24h = 0;
        String[] customerKnownMerchants = new String[0];
        String merchantId = "";
        String merchantMcc = "";
        double merchantAvgAmount = 0;
        boolean terminalIsOnline = false;
        boolean terminalCardPresent = false;
        double terminalKmFromHome = 0;
        boolean hasLastTransaction = false;
        long lastTxTimestamp = 0;
        double lastTxKmFromCurrent = 0;

        skipWs();
        expect((byte) '{');
        skipWs();

        // Loop over top-level fields. Exit when we hit '}'.
        while (peek() != '}') {
            String key = readString();
            skipWs();
            expect((byte) ':');
            skipWs();

            switch (key) {
                case "id" -> skipString(); // we don't need the id
                case "transaction" -> {
                    enterObject();
                    while (peek() != '}') {
                        String k = readString();
                        skipWs(); expect((byte) ':'); skipWs();
                        switch (k) {
                            case "amount" -> txAmount = readDouble();
                            case "installments" -> txInstallments = (int) readLong();
                            case "requested_at" -> txRequestedAt = parseIsoToEpoch(readString());
                            default -> skipValue();
                        }
                        skipCommaOrEnd();
                    }
                    expect((byte) '}');
                }
                case "customer" -> {
                    enterObject();
                    while (peek() != '}') {
                        String k = readString();
                        skipWs(); expect((byte) ':'); skipWs();
                        switch (k) {
                            case "avg_amount" -> customerAvgAmount = readDouble();
                            case "tx_count_24h" -> customerTxCount24h = (int) readLong();
                            case "known_merchants" -> customerKnownMerchants = readStringArray();
                            default -> skipValue();
                        }
                        skipCommaOrEnd();
                    }
                    expect((byte) '}');
                }
                case "merchant" -> {
                    enterObject();
                    while (peek() != '}') {
                        String k = readString();
                        skipWs(); expect((byte) ':'); skipWs();
                        switch (k) {
                            case "id" -> merchantId = readString();
                            case "mcc" -> merchantMcc = readString();
                            case "avg_amount" -> merchantAvgAmount = readDouble();
                            default -> skipValue();
                        }
                        skipCommaOrEnd();
                    }
                    expect((byte) '}');
                }
                case "terminal" -> {
                    enterObject();
                    while (peek() != '}') {
                        String k = readString();
                        skipWs(); expect((byte) ':'); skipWs();
                        switch (k) {
                            case "is_online" -> terminalIsOnline = readBoolean();
                            case "card_present" -> terminalCardPresent = readBoolean();
                            case "km_from_home" -> terminalKmFromHome = readDouble();
                            default -> skipValue();
                        }
                        skipCommaOrEnd();
                    }
                    expect((byte) '}');
                }
                case "last_transaction" -> {
                    if (peek() == 'n') {
                        // null
                        expectLiteral("null");
                        // hasLastTransaction stays false
                    } else {
                        hasLastTransaction = true;
                        enterObject();
                        while (peek() != '}') {
                            String k = readString();
                            skipWs(); expect((byte) ':'); skipWs();
                            switch (k) {
                                case "timestamp" -> lastTxTimestamp = parseIsoToEpoch(readString());
                                case "km_from_current" -> lastTxKmFromCurrent = readDouble();
                                default -> skipValue();
                            }
                            skipCommaOrEnd();
                        }
                        expect((byte) '}');
                    }
                }
                default -> skipValue();
            }
            skipCommaOrEnd();
        }
        expect((byte) '}');

        return new Payload(
                txAmount, txInstallments, txRequestedAt,
                customerAvgAmount, customerTxCount24h, customerKnownMerchants,
                merchantId, merchantMcc, merchantAvgAmount,
                terminalIsOnline, terminalCardPresent, terminalKmFromHome,
                hasLastTransaction, lastTxTimestamp, lastTxKmFromCurrent
        );
    }

    /* -------------------- Lexer primitives -------------------- */

    private byte peek() {
        return buf[pos];
    }

    private void expect(byte c) {
        if (buf[pos] != c) {
            throw new IllegalStateException("Expected '" + (char) c + "' at " + pos + " got '" + (char) buf[pos] + "'");
        }
        pos++;
    }

    private void expectLiteral(String s) {
        for (int i = 0; i < s.length(); i++) {
            if (buf[pos++] != (byte) s.charAt(i)) {
                throw new IllegalStateException("Expected literal " + s);
            }
        }
    }

    private void skipWs() {
        while (pos < buf.length) {
            byte b = buf[pos];
            if (b == ' ' || b == '\n' || b == '\r' || b == '\t') pos++;
            else return;
        }
    }

    private void enterObject() {
        skipWs();
        expect((byte) '{');
        skipWs();
    }

    private void skipCommaOrEnd() {
        skipWs();
        if (pos < buf.length && buf[pos] == ',') {
            pos++;
            skipWs();
        }
    }

    private String readString() {
        expect((byte) '"');
        int start = pos;
        // Optimization: assumes strings have no escapes (valid for this challenge).
        while (buf[pos] != '"') pos++;
        String s = new String(buf, start, pos - start);
        pos++; // consume the closing quote
        return s;
    }

    private void skipString() {
        expect((byte) '"');
        while (buf[pos] != '"') pos++;
        pos++;
    }

    private double readDouble() {
        int start = pos;
        if (buf[pos] == '-') pos++;
        while (pos < buf.length) {
            byte b = buf[pos];
            if ((b >= '0' && b <= '9') || b == '.' || b == 'e' || b == 'E' || b == '+' || b == '-') pos++;
            else break;
        }
        // Double.parseDouble takes a String — allocation unavoidable here.
        // Could be optimized with a custom double parser, but that's later.
        return Double.parseDouble(new String(buf, start, pos - start));
    }

    private long readLong() {
        int start = pos;
        if (buf[pos] == '-') pos++;
        while (pos < buf.length && buf[pos] >= '0' && buf[pos] <= '9') pos++;
        long v = 0;
        boolean neg = buf[start] == '-';
        for (int i = neg ? start + 1 : start; i < pos; i++) {
            v = v * 10 + (buf[i] - '0');
        }
        return neg ? -v : v;
    }

    private boolean readBoolean() {
        if (buf[pos] == 't') {
            expectLiteral("true");
            return true;
        }
        expectLiteral("false");
        return false;
    }

    private String[] readStringArray() {
        expect((byte) '[');
        skipWs();
        if (peek() == ']') { pos++; return new String[0]; }
        // Low initial capacity — known_merchants rarely exceeds 5.
        ArrayList<String> tmp = new ArrayList<>(8);
        while (true) {
            skipWs();
            tmp.add(readString());
            skipWs();
            if (peek() == ']') { pos++; break; }
            expect((byte) ',');
        }
        return tmp.toArray(new String[0]);
    }

    /**
     * Skips an arbitrary JSON value (string, number, bool, null, object, array).
     * Used for fields we don't care about.
     */
    private void skipValue() {
        skipWs();
        byte b = buf[pos];
        switch (b) {
            case '"' -> skipString();
            case 't' -> expectLiteral("true");
            case 'f' -> expectLiteral("false");
            case 'n' -> expectLiteral("null");
            case '{' -> skipObject();
            case '[' -> skipArray();
            default -> {
                // number
                if (b == '-' || (b >= '0' && b <= '9')) {
                    while (pos < buf.length) {
                        byte x = buf[pos];
                        if ((x >= '0' && x <= '9') || x == '.' || x == 'e' || x == 'E' || x == '+' || x == '-') pos++;
                        else break;
                    }
                } else {
                    throw new IllegalStateException("Unexpected '" + (char) b + "' at " + pos);
                }
            }
        }
    }

    private void skipObject() {
        expect((byte) '{');
        int depth = 1;
        while (depth > 0) {
            byte b = buf[pos++];
            if (b == '"') {
                while (buf[pos] != '"') pos++;
                pos++;
            } else if (b == '{') depth++;
            else if (b == '}') depth--;
        }
    }

    private void skipArray() {
        expect((byte) '[');
        int depth = 1;
        while (depth > 0) {
            byte b = buf[pos++];
            if (b == '"') {
                while (buf[pos] != '"') pos++;
                pos++;
            } else if (b == '[') depth++;
            else if (b == ']') depth--;
        }
    }

    /**
     * Parses an ISO-8601 timestamp ("2026-03-11T20:23:35Z") to epoch seconds.
     * We use Instant.parse — allocation is unavoidable, but the call is rare
     * (2 per request) and the JDK has an internal fast path.
     */
    private static long parseIsoToEpoch(String iso) {
        return Instant.parse(iso).getEpochSecond();
    }
}
