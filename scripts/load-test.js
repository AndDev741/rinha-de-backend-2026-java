// k6 load script for rinha-fraud instrumentation runs.
//
// Constant arrival rate roughly matching the rinha test profile:
// ~800 req/s sustained for 30 s. Tail latency measurements are reliable
// once the histograms have a few tens of thousands of samples per span.

import http from 'k6/http';

const RATE = Number(__ENV.RATE || 800);
const DURATION = __ENV.DURATION || '30s';

export const options = {
  scenarios: {
    sustained: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1s',
      duration: DURATION,
      preAllocatedVUs: Math.min(50, RATE),
      maxVUs: Math.max(200, RATE),
    },
  },
  thresholds: {
    // No hard threshold — we're measuring, not gating.
    http_req_failed: ['rate<0.01'],
  },
};

const PAYLOADS = [
  '{"id":"a","transaction":{"amount":250.5,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":180.0,"tx_count_24h":4,"known_merchants":["M-007","M-002"]},"merchant":{"id":"M-007","mcc":"5411","avg_amount":220.0},"terminal":{"is_online":true,"card_present":true,"km_from_home":12.5},"last_transaction":{"timestamp":"2026-03-11T18:00:00Z","km_from_current":5.2}}',
  '{"id":"b","transaction":{"amount":4500.0,"installments":12,"requested_at":"2026-03-11T03:14:00Z"},"customer":{"avg_amount":50.0,"tx_count_24h":1,"known_merchants":["M-100"]},"merchant":{"id":"M-999","mcc":"5812","avg_amount":3200.0},"terminal":{"is_online":false,"card_present":false,"km_from_home":85.0},"last_transaction":null}',
  '{"id":"c","transaction":{"amount":15.0,"installments":1,"requested_at":"2026-03-11T12:00:00Z"},"customer":{"avg_amount":18.0,"tx_count_24h":8,"known_merchants":["M-001","M-002","M-003","M-004"]},"merchant":{"id":"M-001","mcc":"5411","avg_amount":20.0},"terminal":{"is_online":true,"card_present":true,"km_from_home":1.5},"last_transaction":{"timestamp":"2026-03-11T11:30:00Z","km_from_current":0.3}}',
  '{"id":"d","transaction":{"amount":950.0,"installments":6,"requested_at":"2026-03-11T22:45:00Z"},"customer":{"avg_amount":300.0,"tx_count_24h":2,"known_merchants":[]},"merchant":{"id":"M-555","mcc":"5732","avg_amount":850.0},"terminal":{"is_online":true,"card_present":false,"km_from_home":45.0},"last_transaction":{"timestamp":"2026-03-11T21:00:00Z","km_from_current":40.0}}',
  '{"id":"e","transaction":{"amount":75.0,"installments":2,"requested_at":"2026-03-11T09:30:00Z"},"customer":{"avg_amount":80.0,"tx_count_24h":5,"known_merchants":["M-010","M-020","M-030"]},"merchant":{"id":"M-020","mcc":"5411","avg_amount":78.0},"terminal":{"is_online":true,"card_present":true,"km_from_home":3.0},"last_transaction":null}',
  '{"id":"f","transaction":{"amount":7800.0,"installments":24,"requested_at":"2026-03-11T04:50:00Z"},"customer":{"avg_amount":120.0,"tx_count_24h":1,"known_merchants":["M-050","M-051"]},"merchant":{"id":"M-666","mcc":"7995","avg_amount":6500.0},"terminal":{"is_online":true,"card_present":false,"km_from_home":120.0},"last_transaction":null}',
  '{"id":"g","transaction":{"amount":1.5,"installments":1,"requested_at":"2026-03-08T14:15:00Z"},"customer":{"avg_amount":4.0,"tx_count_24h":3,"known_merchants":["M-300","M-301"]},"merchant":{"id":"M-300","mcc":"5499","avg_amount":2.0},"terminal":{"is_online":true,"card_present":true,"km_from_home":0.5},"last_transaction":null}',
  '{"id":"h","transaction":{"amount":420.0,"installments":3,"requested_at":"2026-03-11T16:05:00Z"},"customer":{"avg_amount":350.0,"tx_count_24h":6,"known_merchants":["M-A","M-B","M-C","M-D","M-E","M-F","M-G"]},"merchant":{"id":"M-Z","mcc":"5969","avg_amount":380.0},"terminal":{"is_online":true,"card_present":false,"km_from_home":22.0},"last_transaction":{"timestamp":"2026-03-11T15:30:00Z","km_from_current":18.0}}',
  '{"id":"i","transaction":{"amount":2100.0,"installments":6,"requested_at":"2026-03-11T13:40:00Z"},"customer":{"avg_amount":900.0,"tx_count_24h":2,"known_merchants":["M-777","M-200","M-007"]},"merchant":{"id":"M-777","mcc":"5944","avg_amount":1800.0},"terminal":{"is_online":true,"card_present":true,"km_from_home":8.0},"last_transaction":{"timestamp":"2026-03-11T12:00:00Z","km_from_current":2.1}}',
  '{"id":"j","transaction":{"amount":8500.0,"installments":18,"requested_at":"2026-03-11T01:20:00Z"},"customer":{"avg_amount":600.0,"tx_count_24h":4,"known_merchants":["M-1","M-2","M-3","M-4","M-5","M-6","M-7"]},"merchant":{"id":"M-X","mcc":"6011","avg_amount":7200.0},"terminal":{"is_online":false,"card_present":true,"km_from_home":60.0},"last_transaction":{"timestamp":"2026-03-10T23:00:00Z","km_from_current":55.0}}',
];

const PARAMS = {
  headers: { 'Content-Type': 'application/json' },
  timeout: '2s',
};

export default function () {
  const idx = __ITER % PAYLOADS.length;
  http.post('http://localhost:9999/fraud-score', PAYLOADS[idx], PARAMS);
}
