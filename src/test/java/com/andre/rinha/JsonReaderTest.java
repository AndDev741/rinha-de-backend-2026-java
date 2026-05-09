package com.andre.rinha;

import com.andre.rinha.json.JsonReader;
import com.andre.rinha.json.Payload;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class JsonReaderTest {

    private static final String LEGIT = """
            {
              "id": "tx-1329056812",
              "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
              "customer":    { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
              "merchant":    { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
              "terminal":    { "is_online": false, "card_present": true, "km_from_home": 29.23 },
              "last_transaction": null
            }""";

    private static final String FRAUD = """
            {
              "id": "tx-3330991687",
              "transaction": { "amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z" },
              "customer":    { "avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"] },
              "merchant":    { "id": "MERC-068", "mcc": "7802", "avg_amount": 54.86 },
              "terminal":    { "is_online": false, "card_present": true, "km_from_home": 952.27 },
              "last_transaction": { "timestamp": "2026-03-14T03:00:00Z", "km_from_current": 100.5 }
            }""";

    @Test
    void parsesLegitWithNullLastTransaction() {
        Payload p = JsonReader.parse(LEGIT.getBytes());
        assertEquals(41.12, p.txAmount(), 1e-6);
        assertEquals(2, p.txInstallments());
        assertEquals(82.24, p.customerAvgAmount(), 1e-6);
        assertEquals(3, p.customerTxCount24h());
        assertArrayEquals(new String[]{"MERC-003", "MERC-016"}, p.customerKnownMerchants());
        assertEquals("MERC-016", p.merchantId());
        assertEquals("5411", p.merchantMcc());
        assertEquals(60.25, p.merchantAvgAmount(), 1e-6);
        assertFalse(p.terminalIsOnline());
        assertTrue(p.terminalCardPresent());
        assertEquals(29.23, p.terminalKmFromHome(), 1e-6);
        assertFalse(p.hasLastTransaction());
    }

    @Test
    void parsesFraudWithLastTransaction() {
        Payload p = JsonReader.parse(FRAUD.getBytes());
        assertEquals(9505.97, p.txAmount(), 1e-6);
        assertEquals(10, p.txInstallments());
        assertTrue(p.hasLastTransaction());
        assertEquals(100.5, p.lastTxKmFromCurrent(), 1e-6);
    }
}
