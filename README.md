# Rinha de Backend 2026 — Java

> **🇬🇧 [English](#english) · 🇧🇷 [Português](#português)**

> **Branch strategy:** each major optimization step lives on its own branch
> (`v1`, `v2`, `v3`, ...). The final submission lives on `main`. This branch
> is **`v1`** — the baseline.

---

## English

Step-1 skeleton of our plan: JDK HTTP server + hand-rolled JSON parser +
heap-resident `float[]` dataset (loaded via mmap). **No Spring, no Jackson,
no Helidon.**

The goal of this step is a measurable baseline. From here we'll layer:

1. ✅ **(step 1 — this branch, `v1`)** scalar baseline, stock JVM
2. SIMD via `jdk.incubator.vector` (Vector API) — branch `v2`
3. Native compilation with GraalVM `native-image` — branch `v3`
4. int8 quantization (if memory gets tight) — branch `v4`
5. Optionally coarse partitioning (k-means) — branch `v5`

Each branch is self-contained and lets us compare scores side-by-side as we
evolve the solution.

---

### Structure

```
rinha-de-backend-andre-java/
├── pom.xml                              # Maven, Java 21, zero runtime deps
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

Cost: 168 MB of heap. Fits in `-Xmx256m`. It'll get tight when we run two
instances in 350 MB total — that's where int8 quantization comes in (step 4:
168 MB → 42 MB).

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
trade.

#### Why no Vector API yet?

The C2 JIT already auto-vectorizes the simple distance loop. Explicit Vector
API will only help once we leave the memory-bound regime. With 168 MB read
sequentially, the bottleneck is RAM bandwidth (~20 GB/s on the test box),
not CPU.

int8 quantization will shrink the footprint 4× and shift the regime.

---

### Build and run

#### 1. Prerequisites

```bash
# Java 21 (any distro: Temurin, Zulu, Amazon Corretto…)
java -version   # must show 21+

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

Expected output:
```
[builder] 1000000 processed (12.4s)
[builder] 2000000 processed (24.7s)
[builder] 3000000 processed (37.1s)
[builder] OK: 3000000 vectors, 999406 frauds (33.31%) in 37.1s
```

Files in `data/`:
- `vectors.bin` — ~161 MB
- `labels.bin`  — ~367 KB
- `meta.txt`    — sanity check

#### 4. Run the app

```bash
DATA_DIR=./data PORT=9999 \
  java -Xmx256m \
       --add-modules=jdk.incubator.vector \
       -jar target/rinha-fraud.jar
```

Expected output:
```
[app] loading dataset from /home/.../data
[app] dataset loaded: 3000000 vectors in 207 ms
[app] warmup complete
[app] listening on :9999 with 4 threads
```

#### 5. Smoke test

```bash
# Health
curl -i http://localhost:9999/ready

# Fraud score (legit example from official docs)
curl -s -X POST http://localhost:9999/fraud-score \
     -H 'Content-Type: application/json' \
     -d '{
       "id": "tx-1329056812",
       "transaction":      { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
       "customer":         { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
       "merchant":         { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
       "terminal":         { "is_online": false, "card_present": true, "km_from_home": 29.23 },
       "last_transaction": null
     }'
# Expected: {"approved":true,"fraud_score":0.0}
```

#### 6. Run unit tests

```bash
mvn -q test
```

Just enough to validate parser and vectorization against the example in the
official docs.

---

### Roadmap

| # | Branch | What | For |
|---|---|---|---|
| 1 | ✅ `v1` (this) | scalar baseline | establish the number to beat |
| 2 | `v2` | Vector API in `KnnSearcher` | shave 30–50% off p99 |
| 3 | `v3` | GraalVM `native-image` | kill warmup + cut RSS to ~50 MB |
| 4 | `v4` | int8 dataset quantization | fit 2 instances comfortably in 350 MB |
| 5 | `v5` | Docker Compose + nginx LB | actual submission |
| 6 | (optional) | coarse k-means partitioning | sub-ms p99 if we still want it |

Once we have a measured baseline, we decide which optimization is worth the
effort based on numbers, not gut feel.

---

### Baseline results — `v1`

> 4 threads, single instance, no resource limits

```json
{
  "expected": {
    "total": 54100,
    "fraud_count": 24058,
    "legit_count": 30042,
    "fraud_rate": 0.4447,
    "legit_rate": 0.5553,
    "edge_case_count": 797,
    "edge_case_rate": 0.0147
  },
  "p99": "1194.22ms",
  "scoring": {
    "breakdown": {
      "false_positive_detections": 0,
      "false_negative_detections": 1,
      "true_positive_detections": 11138,
      "true_negative_detections": 14043,
      "http_errors": 0
    },
    "failure_rate": "0%",
    "weighted_errors_E": 3,
    "error_rate_epsilon": 0.000119,
    "p99_score": {
      "value": -77.08,
      "cut_triggered": false
    },
    "detection_score": {
      "value": 2819.38,
      "rate_component": 3000,
      "absolute_penalty": -180.62,
      "cut_triggered": false
    },
    "final_score": 2742.3
  }
}
```

Detection is essentially perfect. Latency is the entire gap between us and
the top of the leaderboard — that's what `v2` onward will attack.

---
---

## Português

> **Estratégia de branches:** cada otimização grande vive numa branch própria
> (`v1`, `v2`, `v3`, ...). A submissão final vive em `main`. Esta branch é a
> **`v1`** — a baseline.

Esqueleto **Passo 1** do nosso plano: HTTP server da JDK + parser JSON manual + dataset
em `float[]` heap-resident (carregado via mmap). **Sem Spring, sem Jackson, sem Helidon.**

O objetivo deste passo é ter uma baseline mensurável para depois aplicarmos:

1. ✅ **(passo 1 — esta branch, `v1`)** baseline scalar, JVM padrão
2. SIMD via `jdk.incubator.vector` (Vector API) — branch `v2`
3. Compilação nativa com GraalVM `native-image` — branch `v3`
4. Quantização int8 (se a memória apertar) — branch `v4`
5. Eventualmente partição coarse (k-means) — branch `v5`

---

### Estrutura

```
rinha-de-backend-andre-java/
├── pom.xml                              # Maven, Java 21, zero deps de runtime
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

Custo: 168 MB de heap. Cabe em `-Xmx256m`. Vai apertar quando rodarmos 2 instâncias
em 350 MB total — aí entra a quantização int8 (passo 4: 168 MB → 42 MB).

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
startup (que conta para o `/ready`, não para o p99) é um ótimo trade.

#### Por que não Vector API ainda?

O JIT C2 já auto-vetoriza o loop simples de distância. Vector API explícita só vai
ajudar quando sairmos do regime memory-bound. Para 168 MB acessados sequencialmente,
o gargalo é a banda de RAM (~20 GB/s na máquina-alvo), não o CPU.

Quantização int8 vai diminuir o footprint em 4× e aí o regime muda.

---

### Como buildar e rodar

#### 1. Pré-requisitos

```bash
# Java 21 (qualquer distro: Temurin, Zulu, Amazon Corretto…)
java -version   # deve mostrar 21+

# Maven
mvn -v
```

#### 2. Build

```bash
cd ~/andP/rinha-de-backend-andre-java
mvn -q clean package
```

Saída: `target/rinha-fraud.jar`.

#### 3. Gerar o dataset binário

Roda **uma vez**. Pega o `references.json.gz` do repo da rinha e cospe os dois
ficheiros que vamos `mmap` em runtime.

```bash
mkdir -p data

java -cp target/classes \
     --add-modules=jdk.incubator.vector \
     com.andre.rinha.prep.DatasetBuilder \
     ~/andP/rinha-de-backend-2026/resources/references.json.gz \
     ./data
```

Logs esperados:
```
[builder] 1000000 processed (12.4s)
[builder] 2000000 processed (24.7s)
[builder] 3000000 processed (37.1s)
[builder] OK: 3000000 vectors, 999406 frauds (33.31%) in 37.1s
```

Saída em `data/`:
- `vectors.bin` — ~161 MB
- `labels.bin`  — ~367 KB
- `meta.txt`    — sanity check

#### 4. Rodar a aplicação

```bash
DATA_DIR=./data PORT=9999 \
  java -Xmx256m \
       --add-modules=jdk.incubator.vector \
       -jar target/rinha-fraud.jar
```

Logs esperados:
```
[app] loading dataset from /home/.../data
[app] dataset loaded: 3000000 vectors in 207 ms
[app] warmup complete
[app] listening on :9999 with 4 threads
```

#### 5. Smoke test

```bash
# Health
curl -i http://localhost:9999/ready

# Fraud score (caso legítimo da doc oficial)
curl -s -X POST http://localhost:9999/fraud-score \
     -H 'Content-Type: application/json' \
     -d '{
       "id": "tx-1329056812",
       "transaction":      { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
       "customer":         { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
       "merchant":         { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
       "terminal":         { "is_online": false, "card_present": true, "km_from_home": 29.23 },
       "last_transaction": null
     }'
# Esperado: {"approved":true,"fraud_score":0.0}
```

#### 6. Rodar testes unitários

```bash
mvn -q test
```

São poucos por enquanto — apenas o suficiente para validar parser e vetorização
contra o exemplo da documentação oficial.

---

### Próximos passos (em ordem)

| # | Branch | O quê | Para quê |
|---|---|---|---|
| 1 | ✅ `v1` (este) | baseline scalar | estabelecer o número a bater |
| 2 | `v2` | Vector API no `KnnSearcher` | tirar 30–50% do p99 |
| 3 | `v3` | GraalVM `native-image` | matar warmup + cortar RSS para ~50MB |
| 4 | `v4` | quantização int8 do dataset | caber 2 instâncias confortavelmente |
| 5 | `v5` | Docker compose + nginx LB | submissão real |
| 6 | (opcional) | k-means coarse | sub-ms p99 se ainda quisermos |

Quando estivermos com baseline medida, decidimos qual otimização vale o esforço
com base nos números, não no instinto.

---

### Resultados da Baseline — `v1`

> Com 4 threads, instância única, sem limite de recursos

```json
{
  "expected": {
    "total": 54100,
    "fraud_count": 24058,
    "legit_count": 30042,
    "fraud_rate": 0.4447,
    "legit_rate": 0.5553,
    "edge_case_count": 797,
    "edge_case_rate": 0.0147
  },
  "p99": "1194.22ms",
  "scoring": {
    "breakdown": {
      "false_positive_detections": 0,
      "false_negative_detections": 1,
      "true_positive_detections": 11138,
      "true_negative_detections": 14043,
      "http_errors": 0
    },
    "failure_rate": "0%",
    "weighted_errors_E": 3,
    "error_rate_epsilon": 0.000119,
    "p99_score": {
      "value": -77.08,
      "cut_triggered": false
    },
    "detection_score": {
      "value": 2819.38,
      "rate_component": 3000,
      "absolute_penalty": -180.62,
      "cut_triggered": false
    },
    "final_score": 2742.3
  }
}
```

Detecção praticamente perfeita. A latência é toda a distância entre nós e o
topo do ranking — é o que o `v2` em diante vai atacar.
