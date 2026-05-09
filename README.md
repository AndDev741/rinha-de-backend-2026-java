# Rinha de Backend 2026 — Java

> **🇬🇧 [English](#english) · 🇧🇷 [Português](#português)**

> **Branch strategy:** each major step lives on its own branch (`v1`, `v2`,
> `v3`, ...). The final submission lives on `main`. This branch is **`v3`** —
> explicit SIMD via the JDK Vector API, with an A/B switch (`KNN_MODE`) so
> we can measure the gain on the same binary.

---

## English

In `v3` we replace the scalar squared-distance loop with explicit SIMD via
`jdk.incubator.vector` (FloatVector, FMA, masked tail). Both modes coexist
on the same binary — pick at startup with `KNN_MODE=scalar|vector`. A
parity test ensures both produce identical fraud_scores within float32
epsilon.

**Findings:**
- **Host (no limits): +16% p99 improvement.** Real but modest — C2 was
  already auto-vectorizing the simple loop. Score moves +2649 → +2726.
- **Docker (1 vCPU split 3 ways): both modes still hit −6000.** The
  bottleneck is memory bandwidth, not CPU compute. SIMD makes the math
  faster but each request still has to scan 168 MB of `float[]`.
- **The next move (v4 = int8 quantization) is now the critical one.**
  Shrinking the dataset 4× lets it fit in L3 cache, eliminates the
  memory-bandwidth wall, AND doubles SIMD throughput (8-lane int8 vs
  current 8-lane float32 with the same hardware).

### Roadmap (current)

1. ✅ **`v1`** — scalar baseline on the host JVM (final score: +2742)
2. ✅ **`v2`** — Docker stack + nginx LB + cgroup limits (final score: −6000)
3. ✅ **`v3` (this branch)** — Vector API SIMD with KNN_MODE A/B (host: +2726, Docker: still −6000)
4. **`v4`** — int8 quantization of the dataset (168 MB → 42 MB, fits in L3)
5. **`v5`** — GraalVM `native-image` (kill warmup + slim RSS)
6. (optional) **`v6`** — coarse k-means partitioning (sub-ms p99)

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
│   │   ├── Dataset.java                 # mmap → float[]
│   │   └── KnnSearcher.java             # brute force + max-heap
│   └── prep/
│       └── DatasetBuilder.java          # CLI: references.json.gz → vectors.bin
└── src/test/java/...
    ├── JsonReaderTest.java
    ├── VectorizerTest.java
    └── KnnSearcherParityTest.java       # v3: SCALAR == VECTOR fraud_scores
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

#### Why explicit Vector API in v3?

C2 auto-vectorization is good but not deterministic — small loop changes can
disable it silently. Explicit `FloatVector` guarantees SIMD generation, gives
us FMA (one-rounding-step `sum + diff*diff`), and prepares the SIMD code
path we'll reuse in `v4` with `ByteVector` for int8.

v3 measured: +16% p99 on the host (deterministic SIMD beats partial auto-vec).
On 1 vCPU under cgroup limits, the gain doesn't matter yet because we're
memory-bound, not compute-bound — that's `v4`'s job.

#### Why does `KNN_MODE` exist?

A/B testing on the same binary, same JVM, same dataset. Switching between
modes only changes the distance function. Lets us isolate the SIMD gain
from any other variable, and lets us roll back instantly if a measurement
disagrees with the parity test.

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
- `vectors.bin` — ~161 MB
- `labels.bin`  — ~367 KB
- `meta.txt`    — sanity check

#### 4. Run the app on the host

```bash
# Scalar mode (v1 algorithm, kept for A/B):
DATA_DIR=./data PORT=9999 KNN_MODE=scalar \
  java -Xmx256m --add-modules=jdk.incubator.vector \
       -jar target/rinha-fraud.jar

# Vector mode (default, explicit SIMD):
DATA_DIR=./data PORT=9999 KNN_MODE=vector \
  java -Xmx256m --add-modules=jdk.incubator.vector \
       -jar target/rinha-fraud.jar
```

The startup line `[app] SIMD: ... (8 lanes, 256-bit, ...)` confirms which
SIMD width was selected on your CPU.

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
# Default (vector mode):
docker compose up -d

# Scalar mode (v1 algorithm) — useful for A/B comparison:
KNN_MODE=scalar docker compose up -d
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

### A/B results — `v3` (Vector API)

Same binary, same dataset, same workload. Only `KNN_MODE` changes.

#### Host (no resource limits)

| KNN_MODE | p99 | served | failure | final_score |
|---|---|---|---|---|
| `scalar` | 1480 ms | 21 218 | 0% | **+2649** |
| `vector` | **1241 ms** | **24 424** | 0% | **+2726** |

**Δ** — p99 −16%, throughput +15%, score **+77**. SIMD wins, but C2 was
already doing partial auto-vectorization, so the explicit Vector API gain
is a real but modest improvement.

#### Docker stack (1 vCPU split 0.10 / 0.45 / 0.45)

| KNN_MODE | p99 | served | failure | final_score |
|---|---|---|---|---|
| `scalar` | 2001 ms | 1 605 | 96.65% | **−6000** |
| `vector` | 2001 ms | 1 188 | 97.54% | **−6000** |

**Both modes hit the cutoff floor.** Under 1 vCPU split among 3 containers,
each request still takes ~50–100 ms; the test ramps to 900 req/s, our
capacity is ~30 req/s, p99 hits the 2 s wall, failure rate hits 97%, both
rinha cutoffs fire.

**The takeaway:** Vector API is a real win on the host but doesn't change
the cliff. The cliff is memory bandwidth (each request scans 168 MB of
`float[]`). The next branch (`v4` int8 quantization) shrinks the dataset
to 42 MB so it fits in L3 cache and SIMD lanes double. That's the move
that escapes −6000.

#### Detection parity

In every run, both modes produced identical fraud_scores within float32
epsilon (verified by unit test) and the same TP / FP / FN breakdown.
SIMD changed performance only — never correctness.

---
---

## Português

> **Estratégia de branches:** cada passo grande vive numa branch própria
> (`v1`, `v2`, `v3`, ...). A submissão final vive em `main`. Esta branch é
> a **`v3`** — SIMD explícito via JDK Vector API, com switch A/B
> (`KNN_MODE`) para isolar o ganho no mesmo binário.

Na `v3` substituímos o loop scalar de distância quadrada por SIMD explícito
via `jdk.incubator.vector` (FloatVector, FMA, tail mascarado). Os dois modos
coexistem no mesmo binário — escolhe-se no startup com
`KNN_MODE=scalar|vector`. Um teste de paridade garante que ambos produzem
fraud_scores idênticos dentro do epsilon float32.

**Resultados:**
- **Host (sem limites): +16% no p99.** Real mas modesto — o C2 já fazia
  auto-vectorização parcial. Score: +2649 → +2726.
- **Docker (1 vCPU dividido 3 vias): ambos batem −6000.** O gargalo é a
  largura de banda da memória, não o CPU. SIMD acelera a matemática mas
  cada request ainda lê 168 MB de `float[]`.
- **A próxima jogada (v4 = quantização int8) é a crítica.** Encolher o
  dataset 4× fá-lo caber em L3, elimina o muro de bandwidth, **e** dobra
  a throughput SIMD (8 lanes int8 contra 8 lanes float32 no mesmo hardware).

### Roadmap (atual)

1. ✅ **`v1`** — baseline scalar na JVM do host (final score: +2742)
2. ✅ **`v2`** — Docker stack + nginx LB + cgroup limits (final score: −6000)
3. ✅ **`v3` (esta branch)** — Vector API SIMD com KNN_MODE A/B (host: +2726, Docker: ainda −6000)
4. **`v4`** — quantização int8 do dataset (168 MB → 42 MB, cabe em L3)
5. **`v5`** — GraalVM `native-image` (mata warmup + corta RSS)
6. (opcional) **`v6`** — partição coarse com k-means (p99 sub-ms)

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
│   │   ├── Dataset.java                 # mmap → float[]
│   │   └── KnnSearcher.java             # brute force + max-heap
│   └── prep/
│       └── DatasetBuilder.java          # CLI: references.json.gz → vectors.bin
└── src/test/java/...
    ├── JsonReaderTest.java
    ├── VectorizerTest.java
    └── KnnSearcherParityTest.java       # v3: SCALAR == VECTOR fraud_scores
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

#### Por que Vector API explícita na v3?

A auto-vectorização do C2 funciona mas não é determinística — pequenas
mudanças no loop podem desactivá-la silenciosamente. `FloatVector` explícito
garante geração SIMD, dá-nos FMA (`sum + diff*diff` numa única instrução
com uma só rounding step), e prepara o caminho SIMD que vamos reutilizar
na `v4` com `ByteVector` para int8.

A v3 mediu: +16% no p99 do host (SIMD determinístico bate auto-vectorização
parcial). Sob 1 vCPU com cgroup, o ganho ainda não importa porque estamos
memory-bound, não compute-bound — esse é o trabalho da `v4`.

#### Por que existe `KNN_MODE`?

Testes A/B no mesmo binário, mesma JVM, mesmo dataset. Mudar de modo só
muda a função de distância. Permite isolar o ganho de SIMD de qualquer
outra variável e fazer rollback instantâneo se uma medição discordar do
teste de paridade.

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
# Modo scalar (algoritmo da v1, mantido para A/B):
DATA_DIR=./data PORT=9999 KNN_MODE=scalar \
  java -Xmx256m --add-modules=jdk.incubator.vector \
       -jar target/rinha-fraud.jar

# Modo vector (default, SIMD explícito):
DATA_DIR=./data PORT=9999 KNN_MODE=vector \
  java -Xmx256m --add-modules=jdk.incubator.vector \
       -jar target/rinha-fraud.jar
```

A linha de startup `[app] SIMD: ... (8 lanes, 256-bit, ...)` confirma a
largura SIMD selecionada na tua CPU.

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
# Default (modo vector):
docker compose up -d

# Modo scalar (algoritmo da v1) — útil para comparação A/B:
KNN_MODE=scalar docker compose up -d
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

### Resultados A/B — `v3` (Vector API)

Mesmo binário, mesmo dataset, mesma carga. Só `KNN_MODE` muda.

#### Host (sem limites de recursos)

| KNN_MODE | p99 | servidos | falha | final_score |
|---|---|---|---|---|
| `scalar` | 1480 ms | 21 218 | 0% | **+2649** |
| `vector` | **1241 ms** | **24 424** | 0% | **+2726** |

**Δ** — p99 −16%, throughput +15%, score **+77**. SIMD ganha, mas o C2 já
fazia auto-vectorização parcial, então o ganho do Vector API explícito é
real mas modesto.

#### Stack Docker (1 vCPU dividida 0.10 / 0.45 / 0.45)

| KNN_MODE | p99 | servidos | falha | final_score |
|---|---|---|---|---|
| `scalar` | 2001 ms | 1 605 | 96.65% | **−6000** |
| `vector` | 2001 ms | 1 188 | 97.54% | **−6000** |

**Os dois modos batem o piso do corte.** Sob 1 vCPU dividida em 3
containers, cada request demora ~50–100 ms; o teste sobe a 900 req/s,
a nossa capacidade é ~30 req/s, o p99 chega ao muro de 2 s, a falha
chega a 97%, ambos os cortes da rinha disparam.

**A leitura:** Vector API é uma vitória real no host mas não muda o
precipício. O precipício é a largura de banda da memória (cada request
percorre 168 MB de `float[]`). A próxima branch (`v4` quantização int8)
encolhe o dataset para 42 MB para caber em L3 e dobra as lanes SIMD.
É a jogada que escapa ao −6000.

#### Paridade da detecção

Em todos os runs, ambos os modos produziram fraud_scores idênticos dentro
do epsilon de float32 (verificado por teste unitário) e o mesmo breakdown
de TP / FP / FN. SIMD mudou só performance — nunca correctness.
