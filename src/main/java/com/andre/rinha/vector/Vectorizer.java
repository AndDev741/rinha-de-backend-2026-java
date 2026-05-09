package com.andre.rinha.vector;

import com.andre.rinha.json.Payload;

import java.time.Instant;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Transforms a Payload into the 14 dimensions defined in REGRAS_DE_DETECCAO.md.
 *
 * Decisions:
 * - Constants from normalization.json are hardcoded as `static final`. Trades
 *   configurability for execution speed (constant folding by the JIT) and
 *   zero I/O at runtime.
 * - mcc_risk: HashMap pre-populated once. 10 entries → essentially zero cost
 *   per lookup.
 * - The main method writes directly into a float[14] passed by the caller.
 *   Allows the array to be reused (per-request, thread-local) — zero
 *   allocation on the hot path.
 */
public final class Vectorizer {

    // ----- Normalization constants -----
    private static final double MAX_AMOUNT              = 10_000d;
    private static final double MAX_INSTALLMENTS        = 12d;
    private static final double AMOUNT_VS_AVG_RATIO     = 10d;
    private static final double MAX_MINUTES             = 1440d;
    private static final double MAX_KM                  = 1000d;
    private static final double MAX_TX_COUNT_24H        = 20d;
    private static final double MAX_MERCHANT_AVG_AMOUNT = 10_000d;

    // Risk table by MCC (mcc_risk.json). Missing MCCs get the default 0.5.
    private static final Map<String, Float> MCC_RISK;
    private static final float MCC_RISK_DEFAULT = 0.5f;

    static {
        // Hardcoded — the file is fixed and has only 10 entries.
        MCC_RISK = new HashMap<>(16);
        MCC_RISK.put("5411", 0.15f);
        MCC_RISK.put("5812", 0.30f);
        MCC_RISK.put("5912", 0.20f);
        MCC_RISK.put("5944", 0.45f);
        MCC_RISK.put("7801", 0.80f);
        MCC_RISK.put("7802", 0.75f);
        MCC_RISK.put("7995", 0.85f);
        MCC_RISK.put("4511", 0.35f);
        MCC_RISK.put("5311", 0.25f);
        MCC_RISK.put("5999", 0.50f);
    }

    public static final int DIMS = 14;

    private Vectorizer() {}

    /**
     * Fills `out` (length 14) with the normalized vector.
     * Uses float (not double) to match the reference dataset and to enable
     * SIMD with FloatVector later on.
     */
    public static void vectorize(Payload p, float[] out) {
        // 0: amount
        out[0] = clamp01((float) (p.txAmount() / MAX_AMOUNT));
        // 1: installments
        out[1] = clamp01((float) (p.txInstallments() / MAX_INSTALLMENTS));
        // 2: amount_vs_avg
        // Watch out for avg_amount = 0. The rules don't say what to do; we use
        // 1.0 (saturation) as a reasonable interpretation: if the customer
        // never spent anything, any transaction looks far above average.
        float dim2 = p.customerAvgAmount() > 0
                ? clamp01((float) ((p.txAmount() / p.customerAvgAmount()) / AMOUNT_VS_AVG_RATIO))
                : 1.0f;
        out[2] = dim2;

        // 3 and 4: hour_of_day and day_of_week — derived from timestamp (UTC).
        ZonedDateTime zdt = Instant.ofEpochSecond(p.txRequestedAtEpochSeconds()).atZone(ZoneOffset.UTC);
        out[3] = (float) (zdt.getHour() / 23d);
        // DayOfWeek: MONDAY=1 ... SUNDAY=7 → map to 0..6 (Mon=0).
        out[4] = (float) ((zdt.getDayOfWeek().getValue() - 1) / 6d);

        // 5 and 6: depend on last_transaction. Sentinel -1 when absent.
        if (p.hasLastTransaction()) {
            long minutes = (p.txRequestedAtEpochSeconds() - p.lastTxTimestampEpochSeconds()) / 60;
            if (minutes < 0) minutes = 0; // robustness: out-of-order timestamps
            out[5] = clamp01((float) (minutes / MAX_MINUTES));
            out[6] = clamp01((float) (p.lastTxKmFromCurrent() / MAX_KM));
        } else {
            out[5] = -1f;
            out[6] = -1f;
        }

        // 7: km_from_home
        out[7] = clamp01((float) (p.terminalKmFromHome() / MAX_KM));
        // 8: tx_count_24h
        out[8] = clamp01((float) (p.customerTxCount24h() / MAX_TX_COUNT_24H));
        // 9 and 10: booleans
        out[9]  = p.terminalIsOnline()    ? 1f : 0f;
        out[10] = p.terminalCardPresent() ? 1f : 0f;

        // 11: unknown_merchant — 1 if merchantId is NOT in known_merchants.
        out[11] = isKnownMerchant(p.merchantId(), p.customerKnownMerchants()) ? 0f : 1f;

        // 12: mcc_risk — direct lookup, default 0.5.
        Float risk = MCC_RISK.get(p.merchantMcc());
        out[12] = risk != null ? risk : MCC_RISK_DEFAULT;

        // 13: merchant_avg_amount
        out[13] = clamp01((float) (p.merchantAvgAmount() / MAX_MERCHANT_AVG_AMOUNT));
    }

    private static float clamp01(float v) {
        if (v < 0f) return 0f;
        if (v > 1f) return 1f;
        return v;
    }

    private static boolean isKnownMerchant(String id, String[] known) {
        // Linear scan. For arrays of 0-5 elements it's faster than HashSet.
        for (String s : known) {
            if (s.equals(id)) return true;
        }
        return false;
    }
}
