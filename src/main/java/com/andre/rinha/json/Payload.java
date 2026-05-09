package com.andre.rinha.json;

/**
 * Immutable representation of the POST /fraud-score request.
 *
 * Design notes:
 * - Record (Java 16+) is the modern Java "DTO": immutable, generated
 *   equals/hashCode/toString, zero boilerplate.
 * - known_merchants comes as String[] (not List) to avoid ArrayList overhead.
 *   Since the vector only needs to know whether merchantId IS or IS NOT in
 *   the array, a linear scan over 2-5 elements is faster than a HashSet for
 *   that size.
 * - hasLastTransaction is a dedicated boolean instead of making the
 *   last_transaction_minutes field nullable. Java has no zero-overhead
 *   optional primitives, so we avoid Float/Double boxing.
 * - Dates become epochSeconds (long) at parse time — we don't keep the
 *   original string. Cheaper to compute hour_of_day and day_of_week directly
 *   from the epoch.
 */
public record Payload(
        // transaction.*
        double txAmount,
        int txInstallments,
        long txRequestedAtEpochSeconds,
        // customer.*
        double customerAvgAmount,
        int customerTxCount24h,
        String[] customerKnownMerchants,
        // merchant.*
        String merchantId,
        String merchantMcc,
        double merchantAvgAmount,
        // terminal.*
        boolean terminalIsOnline,
        boolean terminalCardPresent,
        double terminalKmFromHome,
        // last_transaction.* (may be null in the original payload)
        boolean hasLastTransaction,
        long lastTxTimestampEpochSeconds,
        double lastTxKmFromCurrent
) {
}
