# Rinha de Backend 2026 — Java

> **🇬🇧 [English](#english) · 🇧🇷 [Português](#português)**

> **Branch strategy:** each major step lives on its own branch (`v1`, `v2`,
> `v3`, ...). The final submission lives on `main`. This branch is **`v2`** —
> the realistic-constraints stack: Docker Compose, nginx LB, two API replicas,
> with the rinha CPU/memory budget.

---

## English

In `v2` we package the v1 code into a real submission-shaped stack and run
the official k6 test under the actual rinha environment (1 vCPU + 350 MB
total). We add **no algorithmic changes**: the goal is to learn what the
realistic constraints look like.

**Spoiler:** v1's heap-resident dataset doesn't fit in two containers under
350 MB. Even with relaxed memory we hit a 1-vCPU wall: each request is
~50–200 ms, the test ramps to 900 req/s, the p99 cutoff fires. Score: **−6000**.
That's the lesson — and it's what `v3` onward will fix with real numbers.

### Roadmap (revised in v2)

1. ✅ **`v1`** — scalar baseline, no infra, run on host JVM (final score: +2742)
2. ✅ **`v2` (this branch)** — Docker stack + nginx LB + cgroup limits (final score: −6000)
3. **`v3`** — SIMD via `jdk.incubator.vector` (Vector API)
4. **`v4`** — int8 quantization of the dataset (168 MB → 42 MB)
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
    └── VectorizerTest.java
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
- `vectors.bin` — ~161 MB
- `labels.bin`  — ~367 KB
- `meta.txt`    — sanity check

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
---

## Português

> **Estratégia de branches:** cada passo grande vive numa branch própria
> (`v1`, `v2`, `v3`, ...). A submissão final vive em `main`. Esta branch é
> a **`v2`** — o stack com restrições reais: Docker Compose, nginx LB, duas
> réplicas da API, dentro do orçamento de CPU/memória da rinha.

Na `v2` empacotamos o código da v1 num stack com forma de submissão real e
rodamos o teste oficial do k6 dentro do ambiente real da rinha (1 vCPU +
350 MB no total). **Sem mudanças algorítmicas**: o objectivo é aprender
como são as restrições de verdade.

**Spoiler:** o dataset em heap da v1 não cabe em duas instâncias dentro de
350 MB. Mesmo afrouxando a memória, batemos no muro de 1 vCPU: cada request
demora ~50–200 ms, o teste sobe para 900 req/s, o corte de p99 dispara.
Score: **−6000**. Essa é a lição — e é o que `v3` em diante vai resolver
com números reais.

### Roadmap (revisto na v2)

1. ✅ **`v1`** — baseline scalar, sem infra, JVM no host (final score: +2742)
2. ✅ **`v2` (esta branch)** — Docker stack + nginx LB + cgroup limits (final score: −6000)
3. **`v3`** — SIMD via `jdk.incubator.vector` (Vector API)
4. **`v4`** — quantização int8 do dataset (168 MB → 42 MB)
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
    └── VectorizerTest.java
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
