# rinha-fraud-java — submission

This branch holds the minimal stack the rinha test engine runs.

| File | Role |
|---|---|
| `docker-compose.yml` | Pulls `anddev741/rinha-fraud-java:v7` and brings up 2 API instances behind nginx, under the 1 vCPU / 350 MB budget. |
| `nginx.conf` | Round-robin between `api-1` and `api-2`, listening on `:9999`. |

The image is pre-baked: the int16 IVF dataset (84 MB) lives inside
`/data` in the container, so `/ready` answers 200 within about a second
of startup. Nothing else is needed.

To run locally:

```bash
docker compose up -d
until curl -sf http://localhost:9999/ready > /dev/null; do sleep 1; done
```

The k6 test script comes from the
[rinha repo](https://github.com/zanfranceschi/rinha-de-backend-2026):

```bash
git clone https://github.com/zanfranceschi/rinha-de-backend-2026.git /tmp/rinha
cd /tmp/rinha
k6 run test/test.js
```

---

### Where the rest lives

Source code, Dockerfile, the v1-to-v7 journey, and the design notes are
on the [`main` branch](../../tree/main). Read that one if you want to
understand how the IVF + bbox-repair k-NN works or how the image is
built.
