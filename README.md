# Rinha de Backend 2026 — Java

> **🇬🇧 [English](#english) · 🇧🇷 [Português](#português)**

> **Branch strategy:** each major step lives on its own branch (`v1`, `v2`,
> `v3`, ...). The final submission lives on `main`. This branch is **`v6`** —
> GraalVM `native-image` build. **Surprising regression:** native AOT is
> ~4× slower than the JIT for our SIMD-heavy code on this hardware,
> because GraalVM 25's Vector API native support is still experimental
> and doesn't accelerate our `convertShape` + FMA pipeline. **`v5` remains
> the champion at +2600.82 Docker.**

---

## English

In `v6` we tried to compile the v5 code into a GraalVM native binary.
The hypothesis was: kill JIT warmup variance, shrink RSS, push p99 lower.
The reality on AVX2 hardware with the current GraalVM 25 release: **the
native build is ~4× slower than the JIT for SIMD-heavy code**.

Root cause: GraalVM 25's `native-image` advertises experimental Vector
API support (`-H:+VectorAPISupport`) but only covers "load, store, basic
arithmetic, reduce, compare, blend". Our hot path uses `convertShape(B2F)`
(byte-to-float widening) and `FloatVector.fma` — neither is on that list.
At runtime they fall back to a slow path (likely scalar emulation through
the AOT-compiled bridge), which costs more than the JIT's well-tuned
C2 codegen.

We verified this is not a configuration accident: building both with
and without `VectorAPISupport` gives the same regression. The
native binary fundamentally can't match what HotSpot does for our
specific SIMD calls today.

**`v5` (JVM with explicit Vector API) is the best version we have.**
Below is a record of v6's measurements and the future-work list it
unlocked.

In `v5` we replaced brute force with **IVF (Inverted File Index)**: a
coarse k-means partition over the 3M reference vectors into K=256
clusters. At query time we (1) compute distance to all K centroids,
(2) pick the NPROBE=3 nearest clusters, and (3) brute-force the v4
hybrid SIMD distance only inside those clusters. The brute-force loop
drops from 3M × to ~36k × — a roughly 80× reduction in inner-loop work.

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
5. ✅ **`v5`** — IVF coarse k-means (Docker: **+2600.82**, p99 88 ms) ← champion
6. ✅ **`v6` (this branch)** — GraalVM `native-image` (Docker: −6000, regression vs v5)

### Future optimizations — not yet attempted

- **PGO (Profile-Guided Optimization)** for native-image. Run the binary
  under a representative load with `-H:Profile`, then rebuild using the
  collected profile. Native-image AOT compilers traditionally give up
  10–30 % perf vs JIT in part because they lack runtime profiling;
  PGO closes that gap. Cost: a training run + a second build (~2× build
  time). Could plausibly flip v6 from −6000 to a positive Docker score.
- **Bake the dataset into the image.** Currently `vectors-i8.bin` lives
  in a host volume mounted at `/data`. The rinha submission engine
  doesn't supply data — we have to embed it. Bake means running
  `DatasetBuilder` during `docker build` and `COPY`-ing the binary
  artifacts into the image. Adds ~42 MB to the image but removes the
  external dependency. Done as part of the final submission preparation.
- **NPROBE tuning.** v5 ran NPROBE=3 (recall ~95 %). Bumping to 5 or 7
  would recover detection_score (currently +1547 of max +3000) at the
  cost of p99 (currently +1054). With v5's 88 ms p99, doubling to 176 ms
  is still well under the 2 s cutoff. Net could be +200 to +500 score.
- **More k-means iterations.** v5 stopped at iter 19 with 0.6 %
  reassignments. Pushing to 50 iters would tighten cluster fit and
  recover a small slice of detection.
- **Switch to GraalVM EE (Oracle).** GraalVM EE adds a few percent perf
  and slightly better Vector API support. Requires Oracle license
  acceptance (free for non-commercial).
- **Hand-tuned scalar fallback for native-image.** Write a non-Vector-API
  version of the inner loop and let C-compiler-style optimizations do
  the SIMD. May produce better native code than Vector API does today.

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
│   │   ├── Dataset.java                 # mmap → byte[] (int8) + scales + centroids + offsets
│   │   ├── KnnSearcher.java             # IVF: centroid search + bucket scan
│   │   └── KMeans.java                  # Lloyd's + k-means++ (build-time only)
│   └── prep/
│       └── DatasetBuilder.java          # CLI: json.gz → int8 + scales + centroids + offsets
└── src/test/java/...
    ├── JsonReaderTest.java
    ├── VectorizerTest.java
    ├── KnnSearcherInt8ParityTest.java   # v4: int8 vs float32 ground truth
    └── KnnSearcherIvfRecallTest.java    # v5: IVF recall vs brute-force
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

### v6 results — GraalVM native-image (REGRESSION)

#### Build artifacts

| | v5 (JVM) | v6 (native) |
|---|---|---|
| Image size (unpacked) | 305 MB | **138 MB** (−55 %) |
| Image size (pulled) | 75.5 MB | 34.6 MB (−54 %) |
| Container startup | ~1 s | ~3 s |
| Build time | ~2 s | **~3 min** (native compile) |

#### Benchmarks

| Run | Setup | served | p99 | failure | final_score |
|---|---|---|---|---|---|
| v5 host       | JVM, 4 threads, no limits     | 54 059 | 2.26 ms | 0.26 %  | **+4193.11** |
| v5 Docker     | JVM, rinha rules              | 53 891 | 88 ms   | 0.26 %  | **+2600.82** |
| v6 host       | native, 4 threads, host net   | 14 792 | 2002 ms | 3.11 %  | **−3173.13** |
| v6 Docker     | native, rinha rules (85 MB)   |    597 | 2001 ms | 98.79 % | **−6000.00** |
| v6 Docker (no VectorAPI flag) | native, no `-H:+VectorAPISupport` | 1 076 | 2001 ms | 97.77 % | **−6000.00** |

#### Diagnosis

v6's per-request latency was much higher than v5's at the same load.
Single requests under low concurrency: 35–80 ms (similar to v5). Under
the 900 req/s ramp: every request times out. That's a throughput
regression at sustained load, not a peak-latency regression.

The most likely culprit is the **Vector API native-image support gap**.
GraalVM 25 documents that only basic SIMD ops compile to native
instructions; `convertShape(B2F)` and `FloatVector.fma` (our two
hottest ops) are not on the supported list. They run via a slower
fallback path that the JIT doesn't need.

We tried both with and without `-H:+VectorAPISupport`. Without the
flag is marginally better (1076 vs 597 served), suggesting the
"experimental" support is actively *worse* than Java's standard
scalar emulation for our specific calls. Either way both hit −6000.

#### What v6 successfully delivered

- ✅ **Native binary builds** in 45 s (no reflection/resource config needed).
- ✅ **Static analysis passes** — our hand-rolled HTTP + JSON + Vector
  API code base is fully native-image-compatible without any extra config.
- ✅ **Image is 4.6× smaller** than the v5 JRE image (138 MB vs 305 MB).
- ✅ **Memory budget headroom** — would fit comfortably under the rinha
  cap if performance were competitive.

#### What it didn't

- ❌ **Throughput dropped 4× at sustained load.** The advertised native
  AOT speedup didn't materialize for SIMD-heavy code on AVX2 + GraalVM 25.

#### What we keep

The `v5` JVM build remains the deployment target. The `v6` infrastructure
(native-image profile in pom.xml, multi-stage Dockerfile, distroless-style
runtime) is preserved on this branch as a starting point for future
experiments — particularly **PGO** and **GraalVM EE**, both of which could
plausibly close the gap.

---
---

## Português

> **Estratégia de branches:** cada passo grande vive numa branch própria
> (`v1`, `v2`, `v3`, ...). A submissão final vive em `main`. Esta branch
> é a **`v6`** — build com GraalVM `native-image`. **Regressão
> surpreendente:** AOT nativo é ~4× mais lento que o JIT para o nosso
> código SIMD-heavy neste hardware, porque o suporte a Vector API do
> GraalVM 25 ainda é experimental e não acelera o pipeline `convertShape`
> + FMA. **A `v5` continua a ser a campeã com +2600.82 em Docker.**

Na `v6` tentámos compilar o código da v5 num binário nativo GraalVM. A
hipótese: matar variância do warmup do JIT, encolher RSS, baixar p99.
A realidade em hardware AVX2 com a release atual do GraalVM 25: **o
build nativo é ~4× mais lento que o JIT para código SIMD-heavy.**

Causa raiz: o `native-image` do GraalVM 25 anuncia suporte experimental
a Vector API (`-H:+VectorAPISupport`) mas só cobre "load, store,
basic arithmetic, reduce, compare, blend". O nosso hot path usa
`convertShape(B2F)` (widening byte→float) e `FloatVector.fma` —
nenhum dos dois está nessa lista. Em runtime caem para um caminho
lento (provavelmente emulação scalar via ponte AOT-compilada), que
custa mais que o codegen do C2 do JIT.

Verificámos que isto não é acidente de configuração: builds com e
sem `VectorAPISupport` deram a mesma regressão. O binário nativo
fundamentalmente não consegue igualar o que o HotSpot faz para os
nossos chamadas SIMD específicas hoje.

**A `v5` (JVM com Vector API explícita) é a melhor versão que temos.**
Em baixo está o registo das medições da v6 e a lista de trabalho
futuro que ela desbloqueou.

Na `v5` substituímos o brute force pelo **IVF (Inverted File Index)**:
uma partição coarse via k-means dos 3M vetores de referência em K=256
clusters. Em query time (1) calculamos distância da query aos K centróides,
(2) escolhemos os NPROBE=3 clusters mais próximos, e (3) fazemos brute
force SIMD híbrido (do v4) só dentro desses clusters. O loop de brute
force cai de 3M × para ~36k × — uma redução de ~80× no trabalho do
inner loop.

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
5. ✅ **`v5`** — IVF coarse k-means (Docker: **+2600.82**, p99 88 ms) ← campeã
6. ✅ **`v6` (esta branch)** — GraalVM `native-image` (Docker: −6000, regressão vs v5)

### Optimizações futuras — ainda não experimentadas

- **PGO (Profile-Guided Optimization)** para native-image. Correr o
  binário sob carga representativa com `-H:Profile`, depois recompilar
  usando o profile. Os AOT compilers tradicionalmente cedem 10–30 % ao
  JIT em parte por falta de profiling em runtime; PGO fecha esse gap.
  Custo: um run de treino + segundo build (~2× tempo de build). Pode
  plausivelmente virar a v6 de −6000 para um score positivo em Docker.
- **Bake do dataset na imagem.** Atualmente `vectors-i8.bin` vive num
  volume montado em `/data`. O engine de submissão da rinha não fornece
  dados — temos de embebê-los. Bake significa correr o `DatasetBuilder`
  durante `docker build` e fazer `COPY` dos artefactos para a imagem.
  Adiciona ~42 MB à imagem mas remove a dependência externa. Faz parte
  da preparação da submissão final.
- **Tuning de NPROBE.** A v5 corre NPROBE=3 (recall ~95%). Subir para
  5 ou 7 recuperaria detection_score (atual +1547 do máximo +3000) ao
  custo de p99 (atual +1054). Com p99 da v5 a 88 ms, dobrar para 176 ms
  ainda fica bem abaixo do corte de 2 s. Líquido pode ser +200 a +500
  no score.
- **Mais iterações de k-means.** A v5 parou na iter 19 com 0.6% de
  reassignments. Ir até 50 iters apertaria o ajuste dos clusters e
  recuperaria uma fatia pequena de detection.
- **Mudar para GraalVM EE (Oracle).** GraalVM EE adiciona alguns por
  cento de perf e suporte ligeiramente melhor a Vector API. Requer
  aceitar licença Oracle (gratuita para uso não-comercial).
- **Fallback scalar à mão para native-image.** Escrever uma versão
  do inner loop sem Vector API e deixar otimizações estilo C
  fazerem o SIMD. Pode produzir melhor código nativo do que a Vector
  API hoje.

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
│   │   ├── Dataset.java                 # mmap → byte[] (int8) + scales + centroids + offsets
│   │   ├── KnnSearcher.java             # IVF: centroid search + bucket scan
│   │   └── KMeans.java                  # Lloyd's + k-means++ (build-time only)
│   └── prep/
│       └── DatasetBuilder.java          # CLI: json.gz → int8 + scales + centroids + offsets
└── src/test/java/...
    ├── JsonReaderTest.java
    ├── VectorizerTest.java
    ├── KnnSearcherInt8ParityTest.java   # v4: int8 vs float32 ground truth
    └── KnnSearcherIvfRecallTest.java    # v5: IVF recall vs brute-force
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

### Resultados da v6 — GraalVM native-image (REGRESSÃO)

#### Artefactos do build

| | v5 (JVM) | v6 (nativo) |
|---|---|---|
| Tamanho da imagem (unpacked) | 305 MB | **138 MB** (−55%) |
| Tamanho da imagem (pulled) | 75.5 MB | 34.6 MB (−54%) |
| Startup do container | ~1 s | ~3 s |
| Tempo de build | ~2 s | **~3 min** (compilação nativa) |

#### Benchmarks

| Run | Setup | servidos | p99 | falha | final_score |
|---|---|---|---|---|---|
| v5 host       | JVM, 4 threads, sem limites    | 54 059 | 2.26 ms | 0.26 %  | **+4193.11** |
| v5 Docker     | JVM, regras da rinha           | 53 891 | 88 ms   | 0.26 %  | **+2600.82** |
| v6 host       | nativo, 4 threads, host net    | 14 792 | 2002 ms | 3.11 %  | **−3173.13** |
| v6 Docker     | nativo, regras da rinha (85 MB) |    597 | 2001 ms | 98.79 % | **−6000.00** |
| v6 Docker (sem flag VectorAPI) | nativo, sem `-H:+VectorAPISupport` | 1 076 | 2001 ms | 97.77 % | **−6000.00** |

#### Diagnóstico

A latência por request da v6 sob carga é muito maior que a v5 ao mesmo
load. Requests isolados em baixa concorrência: 35–80 ms (igual à v5).
Sob a rampa de 900 req/s: cada request dá timeout. É uma regressão
de throughput sob carga sustentada, não de latência peak.

A causa mais provável é o **gap no suporte a Vector API do native-image**.
GraalVM 25 documenta que só ops básicas SIMD compilam para instruções
nativas; `convertShape(B2F)` e `FloatVector.fma` (as nossas duas ops
mais quentes) não estão na lista suportada. Correm via um caminho
fallback mais lento que o JIT não precisa.

Tentámos com e sem `-H:+VectorAPISupport`. Sem flag é marginalmente
melhor (1076 vs 597 servidos), sugerindo que o suporte "experimental"
é ativamente *pior* que a emulação scalar standard do Java para as
nossas chamadas específicas. De qualquer forma, ambos batem −6000.

#### O que a v6 entregou com sucesso

- ✅ **Binário nativo compila** em 45 s (sem config para reflection/recursos).
- ✅ **Análise estática passa** — o nosso código (HTTP+JSON+Vector API
  hand-rolled) é totalmente compatível com native-image sem config extra.
- ✅ **Imagem 4.6× menor** que a imagem JRE da v5 (138 MB vs 305 MB).
- ✅ **Margem no orçamento de memória** — caberia confortavelmente sob o
  cap da rinha se a performance fosse competitiva.

#### O que não entregou

- ❌ **Throughput caiu 4× em carga sustentada.** O speedup AOT nativo
  anunciado não materializou para código SIMD-heavy em AVX2 + GraalVM 25.

#### O que mantemos

O build JVM da `v5` continua a ser o alvo de deployment. A infraestrutura
da `v6` (profile native-image no pom.xml, Dockerfile multi-stage, runtime
estilo distroless) fica preservada nesta branch como ponto de partida
para experiências futuras — particularmente **PGO** e **GraalVM EE**,
que poderiam plausivelmente fechar o gap.
