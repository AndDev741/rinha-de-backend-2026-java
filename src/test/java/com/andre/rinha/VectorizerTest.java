package com.andre.rinha;

import com.andre.rinha.json.JsonReader;
import com.andre.rinha.json.Payload;
import com.andre.rinha.vector.Vectorizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Validates vectorization against the examples in REGRAS_DE_DETECCAO.md.
 *
 * The expected values come straight from the official documentation.
 */
class VectorizerTest {

    private static final String LEGIT = """
            {
              "id": "tx-1329056812",
              "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
              "customer":    { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
              "merchant":    { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
              "terminal":    { "is_online": false, "card_present": true, "km_from_home": 29.23 },
              "last_transaction": null
            }""";

    @Test
    void matchesDocumentedLegitVector() {
        Payload p = JsonReader.parse(LEGIT.getBytes());
        float[] v = new float[Vectorizer.DIMS];
        Vectorizer.vectorize(p, v);

        // Documented values:
        // [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
        float tol = 1e-3f;
        assertEquals(0.0041f, v[0],  tol, "amount");
        assertEquals(0.1667f, v[1],  tol, "installments");
        assertEquals(0.05f,   v[2],  tol, "amount_vs_avg");
        assertEquals(0.7826f, v[3],  tol, "hour_of_day (18/23)");
        // 2026-03-11 is a Wednesday: DayOfWeek = WEDNESDAY = 3 (Mon=1).
        // (3-1)/6 = 0.3333 ✓
        assertEquals(0.3333f, v[4],  tol, "day_of_week");
        assertEquals(-1f,     v[5],  0,   "minutes_since_last (sentinel)");
        assertEquals(-1f,     v[6],  0,   "km_from_last (sentinel)");
        assertEquals(0.0292f, v[7],  tol, "km_from_home");
        assertEquals(0.15f,   v[8],  tol, "tx_count_24h");
        assertEquals(0f,      v[9],  0,   "is_online");
        assertEquals(1f,      v[10], 0,   "card_present");
        assertEquals(0f,      v[11], 0,   "unknown_merchant (MERC-016 is in known)");
        assertEquals(0.15f,   v[12], tol, "mcc_risk 5411");
        assertEquals(0.006f,  v[13], tol, "merchant_avg_amount");
    }
}
