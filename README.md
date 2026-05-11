# Rinha de Backend 2026 — Java

> **🇬🇧 [English](#english) · 🇧🇷 [Português](#português)**

> **Branch strategy:** each major step lives on its own branch (`v1`, `v2`,
> `v3`, ...). The final submission lives on `main`. This branch is **`v7`** —
> int16 storage + bbox-repair IVF + scalar manual loop. **Docker score:
> +4084.34** (p99 82 ms, 0% failure, exact k-NN).

---

## English

In `v7` we **rebuild on top of v5** with three targeted changes inspired
by the top JVM solution on the rinha leaderboard:

1. **`int16` storage (× 10000 scaling)** instead of `int8`. Doubles the
   memory footprint (84 MB vs 42 MB) but eliminates quantization loss —
   distances in int16 space are an exact constant scaling of the float32
   distances.
2. **`bbox-repair` IVF** instead of `nprobe=3` approximation. At build time
   we compute the axis-aligned bounding box of each cluster. At query time
   we scan the **single** closest cluster, then for every other cluster
   compute a strict lower-bound distance to its bbox. Clusters that can
   provably not beat the current top-5 are skipped. This is **exact** k-NN
   with most clusters never scanned.
3. **Pure scalar manual-unrolled loop** instead of Vector API. v6 showed
   that GraalVM's experimental Vector API support hurts us, and even on
   the JIT a scalar loop with `if (dist > worst) continue` between dims
   exits cheaply for far candidates. The JIT auto-vectorizes the simple
   form when it can.

**Result: +4084 in Docker (up +1484 from v5).** Detection score hits the
+3000 ceiling (FP=0, FN=0). The remaining gap to the leaderboard top is
all in p99 (currently 82 ms, room for ~+2000 more if we can hit single-
digit milliseconds).

**Findings:**
- 🎯 **First positive Docker score: +2600.82** (was −6000 in v4).
- 🎯 **p99 dropped from 2001 ms (cutoff) to 88 ms.** A ~23× speedup
  per request, on the same hardware, same memory budget.
- ✅ **Detection is approximate but acceptable.** IVF can miss true top-5
  neighbors that straddle cluster boundaries. With NPROBE=3 we trade
  a detection_score reduction (~+1547 of max +3000) for a giant p99 win
  (+1054 of max +3000).
- ✅ **Still fits the 350 MB rinha budget** (inherited from v4).
- 📌 **Where headroom remains:** detection still has ~+1300 left if we
  tune NPROBE up; p99 has ~+2000 if we move to a native binary that kills
  JIT warmup and tightens memory layout.

### Roadmap (current)

1. ✅ **`v1`** — scalar baseline on the host JVM (final score: +2742)
2. ✅ **`v2`** — Docker stack + nginx LB + cgroup limits (final score: −6000)
3. ✅ **`v3`** — Vector API SIMD with KNN_MODE A/B (host: +2726, Docker: −6000)
4. ✅ **`v4`** — int8 storage + hybrid B→F SIMD (Docker fits 350 MB, score: −6000)
5. ✅ **`v5`** — IVF coarse k-means, NPROBE=3 approximation (Docker: +2600.82, p99 88 ms)
6. ✅ **`v6`** — GraalVM `native-image` attempt (Docker: −6000, regression vs v5)
7. ✅ **`v7` (this branch)** — int16 + bbox-repair + scalar manual loop (Docker: **+4084.34**, p99 82 ms, exact k-NN)

### Future optimizations — not yet attempted

- **HAProxy + Unix Domain Socket** instead of nginx + TCP — cheaper hop
  for localhost traffic, can shave ~50-100 ms off p99 outliers under load.
- **K=1024 + cluster splitting** (cap max cluster size at ~500) — tighter
  clusters mean faster per-bucket scans and more aggressive bbox pruning.
- **GraalVM native-image revisited.** v7's pure scalar code (no Vector API)
  is exactly what native-image handles best. Worth re-trying — might
  finally deliver the warmup-free + low-RSS win v6 couldn't.
- **Pre-baked HTTP responses** — pre-build the 6 possible response bytes
  (one per fraud_score value of 0.0/0.2/0.4/0.6/0.8/1.0) and just
  `write()` the right one. Skips StringBuilder + UTF-8 encode per request.
- **PGO (Profile-Guided Optimization)** for native-image. Collect a
  runtime profile, rebuild — could push native-image past the JIT.
- **Bake the dataset into the image** — needed for actual rinha submission.

Each branch is self-contained and lets us compare scores side-by-side as we
evolve the solution.

---

### Structure

```
rinha-de-backend-andre-java/
├── pom.xml                              # Maven, Java 25, zero runtime deps
├── Dockerfile                           # multi-stage build: Maven → JRE 25
├── docker-compose.yml                   # nginx + 2× api, with cgroup limits
├── nginx.conf                           # round-robin LB
├── src/main/java/com/andre/rinha/
│   ├── App.java                         # entry point, starts HTTP server
│   ├── http/
│   │   ├── ReadyHandler.java            # GET /ready
│   │   └── FraudHandler.java            # POST /fraud-score
│   ├── json/
│   │   ├── Payload.java                 # immutable record
│   │   └── JsonReader.java              # hand-rolled parser
│   ├── vector/
│   │   ├── Vectorizer.java              # 14 dimensions + clamp
│   │   ├── Dataset.java                 # mmap → short[] (int16 × 10000) + centroids + bbox + offsets
│   │   ├── KnnSearcher.java             # IVF + bbox-repair (exact k-NN, scalar)
│   │   └── KMeans.java                  # Lloyd's + k-means++ (build-time only)
│   └── prep/
│       └── DatasetBuilder.java          # CLI: json.gz → int16 + centroids + bbox + offsets
└── src/test/java/...
    ├── JsonReaderTest.java
    ├── VectorizerTest.java
    └── KnnSearcherIvfRecallTest.java    # v7: int16 + IVF + bbox vs float32 (exact)
```

---

### Why each piece is the way it is — FAQ

#### Why `com.sun.net.httpserver.HttpServer` and not Helidon/Netty?

It ships with the JDK, zero deps, and the entire server bring-up fits in
five lines. With a 1 vCPU baseline, the bottleneck will never be HTTP — it'll
be the k-NN. Swapping in Helidon/Netty now is premature optimization. Measure
first, decide later.

#### Why a hand-rolled JSON parser?

Jackson and Gson use reflection and allocate intermediate maps/lists to
represent the document. For a fixed ~10-field schema, that costs **~50–100 µs**
per request and ~5 MB of in-flight heap. A cursor-based hand-rolled parser
runs in **~5 µs** with zero allocation beyond the `Payload` itself.

Trade-off: hand-rolled parsers are brittle. If the schema changes, you break.
The challenge contract is fixed, so this is a conscious choice.

#### Why a heap `float[]` instead of keeping the `MappedByteBuffer`?

`MappedByteBuffer.getFloat(i)` is ~3× slower than `array[i]`:
- extra bounds checks
- virtual call (`HeapByteBuffer` vs `DirectByteBuffer`)
- the C2 JIT doesn't auto-vectorize loops over ByteBuffer, but it does over `float[]`

Cost: 168 MB of heap. Fits in `-Xmx320m`. **It does NOT fit twice in 350 MB
total** (proven in v2). That's why `v4` will tackle this with int8 quantization
(168 MB → 42 MB).

#### Why a `BitSet` for the labels?

3M booleans:
- `boolean[]` → 3 MB
- `BitSet`     → 375 KB

The difference fits entirely in L2 cache. Looking up a label during k-NN is
essentially free. `BitSet.get(i)` is a tiny bit-twiddling operation.

#### Why squared L2 distance instead of Euclidean?

`sqrt` is expensive (10–20 cycles) and **doesn't change ordering**. Since we
only compare distances against each other to find the top-5, we can skip the
square root entirely. Free win.

#### Why `ThreadLocal<KnnSearcher>` instead of one per request?

`KnnSearcher` keeps reusable `heapDist[5]` and `heapIdx[5]` arrays. Allocating
one per request creates ~80 bytes of garbage per call. At 10k req/s that's
800 KB/s of GC pressure. ThreadLocal kills it.

#### Why `warmup()` at startup?

The JVM JITs in tiers: starts interpreted, then C1 (fast, lightly optimized),
then C2 (slow, heavily optimized). The first real request hits everything
interpreted and costs 10–50× more than a "warm" request.

The challenge measures p99 from the first request, so paying ~1 second of
warmup at startup (which only counts toward `/ready`, not p99) is a great
trade. **In v2 with 0.45 vCPU per container, warmup balloons to ~37 s** —
expected, since each warmup search now competes with itself for cache.

#### Why no Vector API yet?

Per the v1 hypothesis, the C2 JIT already auto-vectorizes the simple distance
loop and we're memory-bandwidth-bound, not CPU-bound. v2 confirmed CPU is the
real wall under 1 vCPU. v3 will introduce explicit Vector API — and now we
know it'll matter.

---

### Build and run — host (v1 mode)

#### 1. Prerequisites

```bash
# Java 25 (any distro: Temurin, Zulu, Amazon Corretto…)
java -version   # must show 25+

# Maven
mvn -v
```

#### 2. Build

```bash
cd ~/andP/rinha-de-backend-andre-java
mvn -q clean package
```

Output: `target/rinha-fraud.jar`.

#### 3. Generate the binary dataset

Run **once**. Reads `references.json.gz` from the rinha repo and emits the
two files we'll `mmap` at runtime.

```bash
mkdir -p data

java -cp target/classes \
     --add-modules=jdk.incubator.vector \
     com.andre.rinha.prep.DatasetBuilder \
     ~/andP/rinha-de-backend-2026/resources/references.json.gz \
     ./data
```

Files in `data/`:
- `vectors-i8.bin` — 42 MB (int8 quantized vectors)
- `scales.bin`     — 112 B (global mins[14] + maxs[14] in float32 LE)
- `labels.bin`     — ~367 KB
- `meta.txt`       — sanity check

#### 4. Run the app on the host

```bash
DATA_DIR=./data PORT=9999 \
  java -Xmx256m \
       --add-modules=jdk.incubator.vector \
       -jar target/rinha-fraud.jar
```

#### 5. Run unit tests

```bash
mvn -q test
```

---

### Run the Docker stack — v2

This is the actual submission-shaped stack: nginx LB on `:9999`, two API
replicas, all on a bridge network.

#### 1. Make sure the dataset is generated (step 3 above)

#### 2. Build images

```bash
cd ~/andP/rinha-de-backend-andre-java
docker compose build
```

Image: `rinha-fraud:v2`, ~75 MB unpacked (Eclipse Temurin 25 JRE alpine + 25 KB JAR).

#### 3. Start

```bash
docker compose up -d
```

Three services come up:

| Service | Port | CPU | Memory | Role |
|---|---|---|---|---|
| `lb` (nginx) | host:9999 | 0.10 | 30 MB | round-robin LB |
| `api-1` | internal | 0.45 | 380 MB | Java app, instance 1 |
| `api-2` | internal | 0.45 | 380 MB | Java app, instance 2 |

> **Note on memory:** the rinha rule is **350 MB total**. We use `380 MB per
> api` (and 30 MB for nginx) — over budget. v1's heap-resident dataset
> physically can't fit twice in 350 MB. v3/v4 will solve this; v2 measures
> performance under the realistic 1-vCPU constraint while we still have
> v1's memory profile.

Wait ~40 s for warmup (yes, that's slow under 0.45 CPU):

```bash
until curl -s -f http://localhost:9999/ready > /dev/null; do sleep 1; done
echo "stack ready"
```

#### 4. Smoke / load test

From the rinha repo root (so the relative path `test/results.json` works):

```bash
cd ~/andP/rinha-de-backend-2026
k6 run test/test.js
cat test/results.json
```

#### 5. Tear down

```bash
cd ~/andP/rinha-de-backend-andre-java
docker compose down
```

---

### Baseline results — `v1` (host JVM, no limits)

> 4 threads, single instance, no resource limits

```
p99            = 1194 ms
detection      = 2819 / 3000 (0 FP, 1 FN, 0 errs out of 25k served)
final_score    = +2742
```

Detection essentially perfect. p99 paid the cost of being tested on a
generous host.

---

### Stress-test results — `v2` (Docker, 1 vCPU split 3 ways)

> nginx + 2× api, cpus split 0.10 / 0.45 / 0.45, mem relaxed for measurement

```json
{
  "p99": "2001.10ms",
  "scoring": {
    "breakdown": {
      "false_positive_detections": 0,
      "false_negative_detections": 0,
      "true_positive_detections": 594,
      "true_negative_detections": 667,
      "http_errors": 46879
    },
    "failure_rate": "97.38%",
    "p99_score": { "value": -3000, "cut_triggered": true },
    "detection_score": { "value": -3000, "cut_triggered": true },
    "final_score": -6000
  }
}
```

**What this tells us:**

- The host laptop was inflating v1 by ~10×. Reality is harsher.
- Detection logic is still perfect: among the 1 261 successfully served
  requests we got **0 FP, 0 FN**.
- Capacity ≈ 2 instances × 10 req/s = 20 req/s. The k6 test ramps to 900
  req/s — 45× our capacity → cascade of timeouts → both rinha cutoffs fire.
- To survive, **each request must drop from ~100 ms to ~2–3 ms** (≈25× speedup).
- Vector API alone gets us 30–50%. int8 quantization gets 4–8×. Coarse
  k-means partitioning gets 10–20×. We'll need a stack of these.

---

### v4 results — int8 quantization

#### Host (no limits)

| Run | served | p99 | failure | final_score |
|---|---|---|---|---|
| v3 vector | 24 424 | 1241 ms | 0% | **+2726** |
| v4 hybrid | 14 307 | 2002 ms | 56.86% | **−6000** |

v4 is *slower* on the host: the B→F SIMD widening adds compute that doesn't
pay back on a machine with abundant memory bandwidth. v3 wins here.

#### Docker (1 vCPU split, 350 MB total — rinha rules)

| Run | served | p99 | failure | final_score |
|---|---|---|---|---|
| v2 raw      | 1 261 | 2001 ms | 97.38% | −6000 |
| v3 scalar   | 1 605 | 2001 ms | 96.65% | −6000 |
| v3 vector   | 1 188 | 2001 ms | 97.54% | −6000 |
| v4 hybrid   | 668   | 2001 ms | 98.62% | −6000 |

#### What v4 actually achieved

- ✅ **Memory budget solved.** v1-v3 needed 240 MB heap per instance; v4 fits
  in 80 MB (dataset 42 MB + JVM ~30 MB). Two instances + nginx finally fit
  exactly in the 350 MB rinha rule. The Docker compose now ships honest
  limits (160 MB / api, 30 MB / nginx).
- ✅ **Detection still excellent.** 1 FP and 1 FN out of 668 served = 0.3%
  misclass on Docker. Parity test confirms ≥95% fraud_score agreement with
  float32 ground truth on synthetic uniform data (a worst case).
- ❌ **No throughput gain.** All four configs hit the same p99 cutoff floor
  under 1 vCPU. The int8 dataset reads 4× less memory but compute is
  per-request bound regardless of data size.

#### Where the bottleneck has moved

We're no longer memory-bound. We're now bound by the **brute-force loop
itself**: 3M vectors × 14 dims = 42 M ops per request. At even 1 ns/op
that's 42 ms — and we have ~2 ms budget per request to make 900 req/s
on 1 vCPU. The remaining 20× speedup must come from doing **fewer
comparisons**, not faster ones.

That's `v5`'s job (coarse k-means partitioning).

---

### v5 results — IVF k-means partitioning

K=256 clusters, NPROBE=3 (search the 3 nearest clusters per query),
~12k average vectors per cluster.

#### Cluster diagnostics (build-time)

```
[builder] cluster size: min=554 max=37662 mean=11718 empty=0
```

K-means converged at iter 19 with 0.6 % reassignments per iteration
(stopped at the 20-iter cap; could push lower with more iters but the
clusters are stable).

#### Host (no limits)

| Metric | v4 hybrid | v5 IVF | Δ |
|---|---|---|---|
| served | 14 307 | **54 059** | +278 % |
| p99 | 2002 ms | **2.26 ms** | **−885×** |
| failure | 56.86 % | 0.26 % | — |
| **final_score** | **−6000** | **+4193.11** | **+10193** |

#### Docker (1 vCPU split, 350 MB total — rinha rules)

| Run | served | p99 | failure | final_score |
|---|---|---|---|---|
| v2 raw      | 1 261  | 2001 ms | 97.38% | −6000 |
| v3 vector   | 1 188  | 2001 ms | 97.54% | −6000 |
| v4 hybrid   | 668    | 2001 ms | 98.62% | −6000 |
| **v5 IVF**  | **53 891** | **88 ms** | **0.26%** | **+2600.82** |

Detailed v5 Docker breakdown:

```json
{
  "p99": "88.32ms",
  "scoring": {
    "breakdown": {
      "false_positive_detections": 72,
      "false_negative_detections": 70,
      "true_positive_detections": 23953,
      "true_negative_detections": 29938,
      "http_errors": 0
    },
    "failure_rate": "0.26%",
    "p99_score":         { "value": 1053.95, "cut_triggered": false },
    "detection_score":   { "value": 1546.87, "cut_triggered": false },
    "final_score": 2600.82
  }
}
```

#### Why this works

- **80× fewer comparisons.** v4 brute-forces 3M × 14 = 42 M ops per request.
  v5 does 256 × 14 (centroids) + 3 × ~12k × 14 (buckets) ≈ 510 k ops. The
  inner loop time drops from ~50 ms to ~0.5 ms.
- **Same memory budget.** Centroids add 14 KB (256 × 14 × 4 B), offsets
  add 1 KB. Negligible vs the 42 MB int8 dataset.
- **Detection cost is real but bounded.** 142 misclassifications out of
  ~54 k served (0.26 %) come from queries near cluster boundaries
  whose true top-5 spans more than 3 clusters. Tuning NPROBE up trades
  p99 for recall.

#### What's left

- **Detection has +1273 of headroom** (current +1547, max 3000). NPROBE=5
  or 7 could recover most of this if the p99 budget allows.
- **p99 has +1947 of headroom** (current +1054, max 3000). v6 (GraalVM
  native) should kill JIT warmup variance and tighten layout further.
- **Combined v5+v6 target: +4500–5000.**

---

### v7 results — int16 + bbox repair + scalar manual loop

#### Build artifacts

| | v5 (int8 IVF) | v7 (int16 + bbox) |
|---|---|---|
| `vectors-i16.bin` | 42 MB (int8) | **84 MB** (int16) |
| `centroids` | 14 KB (float32) | 7 KB (int16) |
| `bbox.bin` | — | **14 KB** (new) |
| Heap per instance | ~80 MB | ~110 MB |
| Build time (k-means + reorder + bbox) | ~110 s | ~150 s |

#### Benchmarks

| Run | served | p99 | failure | FP / FN | final_score |
|---|---|---|---|---|---|
| v5 host    | 54 059 | 2.26 ms | 0.26 %  | 72 / 70 | +4193.11 |
| **v7 host** | **54 059** | **1.92 ms** | **0 %** | **0 / 0** | **+5715.76** |
| v5 Docker  | 53 891 | 88 ms   | 0.26 %  | 72 / 70 | +2600.82 |
| **v7 Docker** | **54 038** | **82 ms** | **0 %** | **0 / 0** | **+4084.34** |

#### Why the +1484 jump

| Score component | v5 Docker | v7 Docker | Δ |
|---|---|---|---|
| p99_score | +1054 | +1084 | +30 |
| detection_score | +1547 | **+3000** | **+1453** |
| **final_score** | **+2600** | **+4084** | **+1484** |

The win is essentially all in detection. v5's NPROBE=3 approximation cost
~1300 points (FP=72, FN=70 out of 54k served). v7's bbox-repair recovers
all of it — **exact k-NN, FP=0, FN=0**.

The bbox lower-bound check is O(14) per cluster — cheaper than scanning
even one vector — and rejects ~85% of clusters in this dataset. The
fraction of clusters actually scanned per query is small, so p99 doesn't
suffer despite running exact search.

#### Why v7 dropped the Vector API

v6 demonstrated that GraalVM's native-image Vector API support is
incomplete and produces a runtime regression. We also re-examined the JIT
case: on 14-dim vectors, the scalar manual loop with `if (dist > worst)
continue` between dims actually hits comparable throughput to FMA-SIMD
because (a) far candidates exit after 2-4 dims and (b) the JIT auto-
vectorizes the leading sub+mul+add chain just fine.

Bonus: dropping Vector API removes `--add-modules=jdk.incubator.vector`
from the build flags. The project no longer depends on incubator modules.

#### What's left

- **p99 has +1916 of headroom** (current +1084, max +3000). The leaderboard
  top sits at p99 = 3.7 ms with the same algorithm — closing this gap
  means UDS instead of TCP, smaller buckets (K=1024), pre-baked HTTP
  responses, possibly native-image now that Vector API is gone.
- **Detection is maxed out** at +3000.
- **Combined target with future optimizations: ~+5500-6000.**

---
---

## Português

> **Estratégia de branches:** cada passo grande vive numa branch própria
> (`v1`, `v2`, `v3`, ...). A submissão final vive em `main`. Esta branch
> é a **`v7`** — int16 + IVF com bbox-repair + loop scalar manual.
> **Score Docker: +4084.34** (p99 82 ms, 0% falha, k-NN exato).

Na `v7` **reconstruímos sobre a v5** com três mudanças cirúrgicas
inspiradas na melhor solução JVM do ranking da rinha:

1. **Storage int16 (× 10000)** em vez de int8. Dobra a memória (84 MB vs
   42 MB) mas **elimina perda de quantização** — as distâncias em int16
   são uma escala exata constante das distâncias float32.
2. **IVF com bbox-repair** em vez de NPROBE=3 aproximado. No build-time
   calculamos a bounding box axis-aligned de cada cluster. Em query
   varremos o cluster mais próximo, depois para cada outro cluster
   calculamos a distância mínima possível à sua bbox. Clusters que
   provavelmente não batem o top-5 atual são saltados. Isto é k-NN
   **exato** com a maioria dos clusters nunca varridos.
3. **Loop scalar manual unrolled** em vez de Vector API. A v6 mostrou
   que o suporte experimental da Vector API no GraalVM nos prejudica, e
   mesmo no JIT um loop scalar com `if (dist > worst) continue` entre
   dims sai cedo para candidatos longe. O JIT auto-vetoriza a forma
   simples quando consegue.

**Resultado: +4084 em Docker (subida de +1484 desde v5).** Detection score
chega ao tecto de +3000 (FP=0, FN=0). O gap que falta para o topo do
ranking é todo no p99 (82 ms agora; ~+2000 pontos disponíveis se
chegarmos a single-digit ms).

**Resultados:**
- 🎯 **Primeiro score positivo em Docker: +2600.82** (era −6000 em v4).
- 🎯 **p99 caiu de 2001 ms (cutoff) para 88 ms.** Speedup de ~23× por
  request, no mesmo hardware, mesmo orçamento de memória.
- ✅ **Detecção é aproximada mas aceitável.** IVF pode falhar top-5
  reais que cruzam fronteiras de cluster. Com NPROBE=3 trocamos
  redução do detection_score (~+1547 do máximo +3000) por um ganho
  enorme em p99 (+1054 do máximo +3000).
- ✅ **Continua a caber no orçamento de 350 MB** (herdado da v4).
- 📌 **Onde resta margem:** detection tem ~+1300 disponíveis se
  subirmos NPROBE; p99 tem ~+2000 que um binário nativo (que mata
  warmup do JIT e aperta layout) pode capturar.

### Roadmap (atual)

1. ✅ **`v1`** — baseline scalar na JVM do host (final score: +2742)
2. ✅ **`v2`** — Docker stack + nginx LB + cgroup limits (final score: −6000)
3. ✅ **`v3`** — Vector API SIMD com KNN_MODE A/B (host: +2726, Docker: −6000)
4. ✅ **`v4`** — int8 storage + híbrido B→F SIMD (Docker cabe em 350 MB, score: −6000)
5. ✅ **`v5`** — IVF coarse k-means, NPROBE=3 aproximado (Docker: +2600.82, p99 88 ms)
6. ✅ **`v6`** — tentativa GraalVM `native-image` (Docker: −6000, regressão vs v5)
7. ✅ **`v7` (esta branch)** — int16 + bbox-repair + loop scalar (Docker: **+4084.34**, p99 82 ms, k-NN exato)

### Optimizações futuras — ainda não experimentadas

- **HAProxy + Unix Domain Socket** em vez de nginx + TCP — hop mais barato
  para tráfego localhost, pode tirar ~50-100 ms de p99 outliers.
- **K=1024 + split de clusters** (cap max ~500 vetores/cluster) — clusters
  mais apertados → scans por bucket mais rápidos + pruning bbox mais
  agressivo.
- **GraalVM native-image revisitado.** O código scalar puro da v7 (sem
  Vector API) é exatamente o que o native-image melhor consegue. Vale
  re-tentar — pode finalmente entregar o ganho de warmup zero + RSS baixo
  que a v6 não conseguiu.
- **Respostas HTTP pré-prontas** — pré-construir os 6 bytes possíveis
  (um por fraud_score 0.0/0.2/0.4/0.6/0.8/1.0) e só fazer `write()` do
  certo. Salta o StringBuilder + encode UTF-8 por request.
- **PGO (Profile-Guided Optimization)** para native-image. Correr sob
  carga, coletar profile, recompilar — pode levar o native-image além do JIT.
- **Bake do dataset na imagem** — necessário para a submissão real da rinha.

---

### Estrutura

```
rinha-de-backend-andre-java/
├── pom.xml                              # Maven, Java 25, zero deps de runtime
├── Dockerfile                           # build multi-stage: Maven → JRE 25
├── docker-compose.yml                   # nginx + 2× api, com limites cgroup
├── nginx.conf                           # round-robin LB
├── src/main/java/com/andre/rinha/
│   ├── App.java                         # entry point, sobe HTTP server
│   ├── http/
│   │   ├── ReadyHandler.java            # GET /ready
│   │   └── FraudHandler.java            # POST /fraud-score
│   ├── json/
│   │   ├── Payload.java                 # record imutável
│   │   └── JsonReader.java              # parser manual
│   ├── vector/
│   │   ├── Vectorizer.java              # 14 dimensões + clamp
│   │   ├── Dataset.java                 # mmap → short[] (int16 × 10000) + centroids + bbox + offsets
│   │   ├── KnnSearcher.java             # IVF + bbox-repair (exact k-NN, scalar)
│   │   └── KMeans.java                  # Lloyd's + k-means++ (build-time only)
│   └── prep/
│       └── DatasetBuilder.java          # CLI: json.gz → int16 + centroids + bbox + offsets
└── src/test/java/...
    ├── JsonReaderTest.java
    ├── VectorizerTest.java
    └── KnnSearcherIvfRecallTest.java    # v7: int16 + IVF + bbox vs float32 (exact)
```

---

### Por que cada peça é assim — em formato de FAQ

#### Por que `com.sun.net.httpserver.HttpServer` e não Helidon/Netty?

Vem na JDK, zero deps, é tão simples que o código todo do servidor cabe em 5 linhas.
Para uma baseline com 1 vCPU, o gargalo nunca vai ser o HTTP — vai ser o k-NN.
Trocar pelo Helidon/Netty agora é otimização prematura. Vamos medir e decidir.

#### Por que parser JSON manual?

Jackson e Gson usam reflection + alocam mapas/listas para representar o documento.
Para um schema fixo de ~10 campos, isso custa **~50–100 µs** por request e ~5MB de
heap em fly. Um parser cursor-based hand-rolled fica em **~5 µs** e zero alocação
além do `Payload`.

Trade-off: parser manual é frágil. Se o schema mudar, você quebra. Para esse desafio
o schema é fixo no contrato, então é uma escolha consciente.

#### Por que `float[]` heap em vez de manter o `MappedByteBuffer`?

`MappedByteBuffer.getFloat(i)` é ~3× mais lento que `array[i]`:
- bounds checks adicionais
- chamada virtual (`HeapByteBuffer` vs `DirectByteBuffer`)
- o JIT C2 não auto-vetoriza loops sobre ByteBuffer, mas auto-vetoriza sobre `float[]`

Custo: 168 MB de heap. Cabe em `-Xmx320m`. **Não cabe duas vezes em 350 MB
totais** (provado na v2). É por isso que a `v4` vai resolver com quantização
int8 (168 MB → 42 MB).

#### Por que `BitSet` para os labels?

3M booleans:
- `boolean[]` → 3 MB
- `BitSet`     → 375 KB

A diferença cabe inteira em L2 cache. Acessar um label durante o k-NN fica praticamente
free. `BitSet.get(i)` é uma operação bit-twiddling muito barata.

#### Por que distância L2 ao quadrado em vez de euclidiana?

`sqrt` é caro (10–20 ciclos) e **não muda a ordenação**. Como só comparamos distâncias
entre si para achar os top-5, podemos pular a raiz inteiramente. Ganho grátis.

#### Por que `ThreadLocal<KnnSearcher>` em vez de criar um por request?

O `KnnSearcher` mantém os arrays `heapDist[5]` e `heapIdx[5]` reutilizáveis. Criar um
por request gera ~80 bytes de garbage por chamada. Em 10k req/s isso vira 800 KB/s
de pressão no GC. ThreadLocal mata isso de vez.

#### Por que `warmup()` no startup?

A JVM faz JIT em camadas: começa interpretado, depois C1 (rápido, pouco otimizado),
depois C2 (lento, muito otimizado). A primeira request real pega tudo interpretado
e custa 10–50× mais que uma request "quente".

A rinha mede p99 desde a primeira request, então pagar 1 segundo de warmup no
startup (que conta para o `/ready`, não para o p99) é um ótimo trade. **Na v2
com 0.45 vCPU por container, o warmup balona para ~37 s** — esperado.

#### Por que não Vector API ainda?

Pela hipótese da v1, o JIT C2 já auto-vetoriza o loop simples e estamos
memory-bandwidth-bound. A v2 confirmou que sob 1 vCPU o muro real é CPU. A
v3 vai introduzir Vector API explícita — agora sabemos que vai pesar.

---

### Como buildar e rodar — host (modo v1)

#### 1. Pré-requisitos

```bash
# Java 25 (qualquer distro: Temurin, Zulu, Amazon Corretto…)
java -version   # deve mostrar 25+

# Maven
mvn -v
```

#### 2. Build

```bash
cd ~/andP/rinha-de-backend-andre-java
mvn -q clean package
```

#### 3. Gerar o dataset binário

Roda **uma vez**:

```bash
mkdir -p data
java -cp target/classes \
     --add-modules=jdk.incubator.vector \
     com.andre.rinha.prep.DatasetBuilder \
     ~/andP/rinha-de-backend-2026/resources/references.json.gz \
     ./data
```

#### 4. Rodar a aplicação no host

```bash
DATA_DIR=./data PORT=9999 \
  java -Xmx256m \
       --add-modules=jdk.incubator.vector \
       -jar target/rinha-fraud.jar
```

#### 5. Rodar testes unitários

```bash
mvn -q test
```

---

### Rodar o stack Docker — v2

#### 1. Garantir o dataset gerado (passo 3 acima)

#### 2. Build das imagens

```bash
cd ~/andP/rinha-de-backend-andre-java
docker compose build
```

#### 3. Subir

```bash
docker compose up -d
```

| Serviço | Porta | CPU | Memória | Função |
|---|---|---|---|---|
| `lb` (nginx) | host:9999 | 0.10 | 30 MB | LB round-robin |
| `api-1` | interno | 0.45 | 380 MB | App Java, instância 1 |
| `api-2` | interno | 0.45 | 380 MB | App Java, instância 2 |

> **Nota sobre memória:** a regra da rinha é **350 MB total**. Estamos a
> usar **380 MB por api** (+30 MB nginx) — fora do orçamento. O dataset em
> heap da v1 fisicamente não cabe duas vezes em 350 MB. v3/v4 vão resolver;
> a v2 mede performance debaixo da restrição realista de 1 vCPU enquanto
> ainda temos o perfil de memória da v1.

Espera ~40s pelo warmup (sim, é lento sob 0.45 CPU):

```bash
until curl -s -f http://localhost:9999/ready > /dev/null; do sleep 1; done
echo "stack pronto"
```

#### 4. Smoke / load test

A partir da raiz do repo da rinha:

```bash
cd ~/andP/rinha-de-backend-2026
k6 run test/test.js
cat test/results.json
```

#### 5. Derrubar

```bash
cd ~/andP/rinha-de-backend-andre-java
docker compose down
```

---

### Resultados da Baseline — `v1` (JVM no host, sem limites)

```
p99            = 1194 ms
detection      = 2819 / 3000 (0 FP, 1 FN, 0 errs em 25k servidos)
final_score    = +2742
```

Detecção praticamente perfeita. O p99 pagou o custo de ser testado num host
generoso.

---

### Stress-test — `v2` (Docker, 1 vCPU dividida 3 vias)

```json
{
  "p99": "2001.10ms",
  "scoring": {
    "breakdown": {
      "false_positive_detections": 0,
      "false_negative_detections": 0,
      "true_positive_detections": 594,
      "true_negative_detections": 667,
      "http_errors": 46879
    },
    "failure_rate": "97.38%",
    "p99_score": { "value": -3000, "cut_triggered": true },
    "detection_score": { "value": -3000, "cut_triggered": true },
    "final_score": -6000
  }
}
```

**O que isto nos diz:**

- O laptop estava a inflar a v1 por ~10×. A realidade é mais dura.
- A lógica de detecção continua perfeita: dos 1 261 requests servidos com
  sucesso tivemos **0 FP, 0 FN**.
- Capacidade ≈ 2 instâncias × 10 req/s = 20 req/s. O teste do k6 sobe até
  900 req/s — 45× a nossa capacidade → cascata de timeouts → ambos os
  cortes da rinha disparam.
- Para sobreviver, **cada request tem de cair de ~100 ms para ~2–3 ms**
  (≈25× de speedup).
- Vector API sozinha dá 30–50%. Quantização int8 dá 4–8×. Partição coarse
  com k-means dá 10–20×. Vamos precisar de uma combinação.

---

### Resultados da v4 — quantização int8

#### Host (sem limites)

| Run | servidos | p99 | falha | final_score |
|---|---|---|---|---|
| v3 vector | 24 424 | 1241 ms | 0% | **+2726** |
| v4 hybrid | 14 307 | 2002 ms | 56.86% | **−6000** |

A v4 fica *mais lenta* no host: o widening SIMD B→F adiciona compute que
não compensa numa máquina com bandwidth de memória abundante. v3 ganha
aqui.

#### Docker (1 vCPU dividida, 350 MB total — regras da rinha)

| Run | servidos | p99 | falha | final_score |
|---|---|---|---|---|
| v2 raw      | 1 261 | 2001 ms | 97.38% | −6000 |
| v3 scalar   | 1 605 | 2001 ms | 96.65% | −6000 |
| v3 vector   | 1 188 | 2001 ms | 97.54% | −6000 |
| v4 hybrid   | 668   | 2001 ms | 98.62% | −6000 |

#### O que a v4 conseguiu de facto

- ✅ **Orçamento de memória resolvido.** v1-v3 precisavam de 240 MB de
  heap por instância; v4 cabe em 80 MB (dataset 42 MB + JVM ~30 MB). Duas
  instâncias + nginx finalmente cabem **exatamente** nos 350 MB. O
  docker-compose agora declara limites honestos (160 MB / api, 30 MB / nginx).
- ✅ **Detecção continua excelente.** 1 FP e 1 FN em 668 servidos = 0.3%
  de misclass em Docker. Teste de paridade confirma ≥95% de concordância
  no fraud_score contra ground truth float32 em dados uniformes sintéticos.
- ❌ **Sem ganho de throughput.** As quatro configurações batem o mesmo
  piso de p99 sob 1 vCPU. O dataset int8 lê 4× menos memória mas o
  compute é per-request bound, independentemente do tamanho dos dados.

#### Para onde se mudou o gargalo

Já não estamos memory-bound. Estamos agora bound pelo **brute force em
si**: 3M vetores × 14 dims = 42M ops por request. A 1 ns/op isso dá 42 ms
— e temos ~2 ms de orçamento por request para fazer 900 req/s em 1 vCPU.
O speedup de 20× restante tem de vir de **menos comparações**, não de
comparações mais rápidas.

Esse é o trabalho da `v5` (partição coarse com k-means).

---

### Resultados da v5 — IVF k-means

K=256 clusters, NPROBE=3 (busca os 3 mais próximos), ~12k vetores médios
por cluster.

#### Diagnóstico de clusters (build-time)

```
[builder] cluster size: min=554 max=37662 mean=11718 empty=0
```

K-means convergiu na iter 19 com 0.6% reassignments por iteração
(parou no cap de 20 iters; podia ir mais baixo com mais iters mas os
clusters já estão estáveis).

#### Host (sem limites)

| Métrica | v4 hybrid | v5 IVF | Δ |
|---|---|---|---|
| servidos | 14 307 | **54 059** | +278 % |
| p99 | 2002 ms | **2.26 ms** | **−885×** |
| falha | 56.86 % | 0.26 % | — |
| **final_score** | **−6000** | **+4193.11** | **+10193** |

#### Docker (1 vCPU dividida, 350 MB total — regras da rinha)

| Run | servidos | p99 | falha | final_score |
|---|---|---|---|---|
| v2 raw      | 1 261  | 2001 ms | 97.38% | −6000 |
| v3 vector   | 1 188  | 2001 ms | 97.54% | −6000 |
| v4 hybrid   | 668    | 2001 ms | 98.62% | −6000 |
| **v5 IVF**  | **53 891** | **88 ms** | **0.26%** | **+2600.82** |

Breakdown detalhado da v5 Docker:

```json
{
  "p99": "88.32ms",
  "scoring": {
    "breakdown": {
      "false_positive_detections": 72,
      "false_negative_detections": 70,
      "true_positive_detections": 23953,
      "true_negative_detections": 29938,
      "http_errors": 0
    },
    "failure_rate": "0.26%",
    "p99_score":         { "value": 1053.95, "cut_triggered": false },
    "detection_score":   { "value": 1546.87, "cut_triggered": false },
    "final_score": 2600.82
  }
}
```

#### Por que funciona

- **80× menos comparações.** v4 brute-force faz 3M × 14 = 42M ops por
  request. v5 faz 256 × 14 (centróides) + 3 × ~12k × 14 (buckets)
  ≈ 510k ops. O inner loop cai de ~50 ms para ~0.5 ms.
- **Mesmo orçamento de memória.** Centróides adicionam 14 KB
  (256 × 14 × 4 B), offsets adicionam 1 KB. Negligível vs os 42 MB
  do dataset int8.
- **Custo de detecção é real mas limitado.** 142 misclassifications em
  ~54 k servidos (0.26%) vêm de queries perto de fronteiras de
  cluster cujo top-5 real cruza mais de 3 clusters. Tunar NPROBE
  para cima troca p99 por recall.

#### O que sobra

- **Detection tem +1273 de margem** (atual +1547, máximo 3000).
  NPROBE=5 ou 7 pode recuperar a maior parte se p99 permitir.
- **p99 tem +1947 de margem** (atual +1054, máximo 3000). v6
  (GraalVM nativo) deve matar variância do JIT warmup e apertar
  layout.
- **Alvo combinado v5+v6: +4500–5000.**

---

### Resultados da v7 — int16 + bbox repair + loop scalar manual

#### Artefactos do build

| | v5 (int8 IVF) | v7 (int16 + bbox) |
|---|---|---|
| `vectors-i16.bin` | 42 MB (int8) | **84 MB** (int16) |
| `centroids` | 14 KB (float32) | 7 KB (int16) |
| `bbox.bin` | — | **14 KB** (novo) |
| Heap por instância | ~80 MB | ~110 MB |
| Build time (k-means + reorder + bbox) | ~110 s | ~150 s |

#### Benchmarks

| Run | servidos | p99 | falha | FP / FN | final_score |
|---|---|---|---|---|---|
| v5 host    | 54 059 | 2.26 ms | 0.26 %  | 72 / 70 | +4193.11 |
| **v7 host** | **54 059** | **1.92 ms** | **0 %** | **0 / 0** | **+5715.76** |
| v5 Docker  | 53 891 | 88 ms   | 0.26 %  | 72 / 70 | +2600.82 |
| **v7 Docker** | **54 038** | **82 ms** | **0 %** | **0 / 0** | **+4084.34** |

#### Por que o salto de +1484

| Componente | v5 Docker | v7 Docker | Δ |
|---|---|---|---|
| p99_score | +1054 | +1084 | +30 |
| detection_score | +1547 | **+3000** | **+1453** |
| **final_score** | **+2600** | **+4084** | **+1484** |

A vitória é essencialmente toda em detection. A aproximação NPROBE=3 da
v5 custou ~1300 pontos (FP=72, FN=70 em 54k servidos). O bbox-repair da
v7 recupera tudo — **k-NN exato, FP=0, FN=0**.

A verificação de bbox lower-bound é O(14) por cluster — mais barato que
varrer um único vetor — e rejeita ~85% dos clusters neste dataset. A
fração de clusters de facto varridos por query é pequena, então o p99
não sofre apesar de fazermos busca exata.

#### Por que a v7 abandonou a Vector API

A v6 demonstrou que o suporte da Vector API no native-image é incompleto
e produz regressão em runtime. Re-examinámos também o caso JIT: em
vetores de 14 dims, o loop scalar manual com `if (dist > worst)
continue` entre dims atinge throughput comparável ao FMA-SIMD porque
(a) candidatos longe saem após 2-4 dims e (b) o JIT auto-vetoriza a
sequência sub+mul+add inicial sem problemas.

Bónus: dropar Vector API remove `--add-modules=jdk.incubator.vector` das
flags de build. O projeto já não depende de módulos incubator.

#### O que sobra

- **p99 tem +1916 de margem** (atual +1084, máximo +3000). O topo do
  ranking está em p99 = 3.7 ms com o mesmo algoritmo — fechar o gap
  significa UDS em vez de TCP, buckets mais pequenos (K=1024), respostas
  HTTP pré-prontas, possivelmente native-image agora que Vector API
  saiu.
- **Detection está no máximo** em +3000.
- **Alvo combinado com optimizações futuras: ~+5500-6000.**
