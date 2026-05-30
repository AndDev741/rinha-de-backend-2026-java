# Rinha de Backend 2026 — Java

> **🇬🇧 [English](#english) · 🇧🇷 [Português](#português)**

---

## English

A Java solution to the Rinha de Backend 2026 fraud-detection challenge. Given
a credit-card transaction, the API returns whether to approve it by finding
the 5 most similar transactions in a 3 million-vector reference set, then
voting on the fraud labels.

The whole thing runs inside the challenge's hard limits: **1 vCPU and 350 MB
of memory split across 2 API instances and 1 load balancer**.

No Spring, no Jackson, no Helidon. JDK 25, a single-threaded non-blocking NIO
HTTP server ([microhttp](https://github.com/ebarlas/microhttp), ~500 LOC, zero
deps), a hand-rolled JSON parser, and a from-scratch IVF k-NN with bounding-box
pruning. ~1500 lines of Java total.

### Score under rinha rules

| Metric | Value |
|---|---|
| Final score | **+4056.85** |
| p99 latency | 87.73 ms |
| Failure rate | 0 % |
| False positives | 0 / 54 054 |
| False negatives | 0 / 54 054 |
| Detection score | +3000 (maximum) |
| p99 score | +1056.85 |

Image: `anddev741/rinha-fraud-java:v8.1` on Docker Hub.
Submission commit: [`55d7c93`](../../tree/submission).

> **`main` is now v9** (microhttp NIO event loop — see the journey below). The
> score above is the last *confirmed* rinha result (v8.1); v9's own rinha number
> is pending its preview run.

The k-NN is exact: same top-5 as float32 brute force, but with most of the
dataset never scanned. v8 moved compilation from JVM JIT to GraalVM
`native-image`, killed per-request allocations on the parse / vectorize /
respond pipeline, and added a startup pre-touch of the dataset so the
first request never page-faults. Under the rinha rules (1 vCPU, 350 MB,
2 instances behind nginx) every answered request is classified correctly,
and no requests time out.

This repo is also a learning trail. Each branch (`v1` through `v9`) is
a working snapshot of one optimization, with measured scores. The journey
includes documented regressions — v6 (Vector API + native-image),
v7.1 (virtual threads), v8 (heap pinning + sort overhead). Each one
taught something concrete; the rinha section at the bottom of this README
tells those stories.

---

### Quick start

You need Docker, Docker Compose, and [k6](https://k6.io) for the load test.
The dataset is built inside the image, so there is no Java step on the host.

Two compose files are available (both run the v9 microhttp server):
- `docker-compose.yml` — the **JVM stack** (`Dockerfile`)
- `docker-compose.native.yaml` — the **native-image stack**
  (`Dockerfile.native`, ~6 min first build because `native-image` is slow)

```bash
# 1. Build the image. First run fetches references.json.gz and runs
#    DatasetBuilder, ~3 min for JVM / ~6 min for native. Cached after.
docker compose -f docker-compose.native.yaml build

# 2. Start the stack
docker compose -f docker-compose.native.yaml up -d

# 3. Wait for /ready and run the official test
until curl -sf http://localhost:9999/ready > /dev/null; do sleep 1; done
git clone https://github.com/zanfranceschi/rinha-de-backend-2026.git /tmp/rinha
cd /tmp/rinha
k6 run test/test.js
cat test/results.json
```

The `submission` branch ships a `docker-compose.yml` that pulls the
pre-built `anddev741/rinha-fraud-java:v8.1` image from Docker Hub, so the
rinha test engine doesn't have to rebuild.

### How the search works

The algorithm is **IVF (Inverted File) with bounding-box repair**.

At build time, `DatasetBuilder` runs k-means with K=256 clusters over the
3M float vectors. It quantizes every vector to int16 (multiplied by 10 000,
which preserves 4 decimal digits of precision) and reorders the array so
cluster 0 occupies the first bytes, cluster 1 the next, and so on. Each
cluster also gets a 14-dimensional bounding box: the smallest and largest
value of each dimension across the cluster's vectors.

At query time, the search does five things:

1. Quantize the float query to int16, same scaling as the dataset.
2. Compute the squared distance from the query to all 256 centroids. Pick
   the closest cluster.
3. Scan that single cluster, building a top-5 max-heap.
4. For every other cluster, compute the lower-bound distance from the query
   to the cluster's bounding box. If the lower bound already beats the worst
   of the current top-5, skip the cluster. Otherwise scan it.
5. Count how many of the final top-5 are fraud, divide by 5.

The bbox lower bound is one of those tricks that feels too good to be true
until you derive it. For each dimension `d`, the minimum possible distance
from query value `q[d]` to any vector in cluster `c` on that dimension is
zero if `bbox_min[c][d] ≤ q[d] ≤ bbox_max[c][d]`, otherwise it is the
distance to the nearer boundary. Sum the squares and you get a strict lower
bound on `distance(q, any_vector_in_c)`. If that bound exceeds your current
worst top-5, no vector in `c` can win, and you skip 12k vectors with a
single 14-dim test.

In practice the first cluster scan produces a tight top-5 and bbox rejects
the remaining 200+ clusters. The search is exact (no recall loss) but
behaves like a small fraction of brute force.

---

### Why these choices and not others

#### Why no Spring / Helidon / Netty?

The challenge runs on 1 vCPU. A full framework buys nothing here and costs RSS.
v1–v8.1 used the JDK's `com.sun.net.httpserver`, but its thread-per-request model
contends with itself on the 0.45 CPU each container gets, inflating the p99 tail.
v9 switched to microhttp — a ~500-line single-threaded non-blocking NIO event
loop (still not a framework, still zero heavy deps): one event-loop thread runs
flat-out with no self-contention, which is exactly what a fractional CPU wants.
Netty or Vert.x would add the same thread contention plus a far larger surface.

#### Why a hand-rolled JSON parser?

Jackson uses reflection and allocates intermediate maps to represent the
document. For a fixed 10-field schema, that costs around 50 to 100 µs per
request and a few MB of in-flight heap. A cursor-based parser specialized
for this exact payload runs in about 5 µs with no allocation beyond the
`Payload` record itself.

The trade-off is brittleness. If the payload schema ever changes, the
parser breaks. The challenge contract is fixed, so this is a conscious
choice.

#### Why int16 instead of int8?

We tried int8 in v4 with per-dimension min/max scaling. The math turns the
quantized distance into a per-dim weighted L2 in float space, where
booleans (range 1) get one weight and the sentinel dim (range 2) gets a
quarter weight. Synthetic top-5 agreement crashed to 14 %.

We then tried int8 with a single global scale (v5). That preserved ordering
but limited resolution to ~1/128 per dim, and the `nprobe=3` IVF
approximation cost about 1300 detection points in real runs.

int16 with `× 10 000` scaling fits in `short` (±32 767) with margin, gives
~1/10 000 resolution per dim, and lets us run exact k-NN with bbox repair.
Memory cost is 84 MB instead of 42 MB. Still inside the budget.

#### Why no Vector API?

Tried three times: v3, v6, and once during the v8 cycle. JIT (v3) got a
small win in unconstrained microbenchmarks but no improvement under
cgroup limits. GraalVM `native-image` v6 and v8 both regressed
catastrophically — `ShortVector.convertShape(S2I)` is not escape-analyzed
by C2 or by Substrate VM (the AOT compiler under native-image), so each
call allocates ~787 KB of wrapper objects. The 800 RPS benchmark went from
44 ms avg to over 400 ms with 67 % of requests failing.

The current scalar manual-unrolled loop, with `if (dist > worst) continue`
between each of the 14 dimensions, ends up being competitive everywhere.
Most candidates exit after 2-4 dims because they are far from the query,
and the AOT compiler emits tight straight-line code for the centroid loop.
The top-14 jvmoonshot solution arrives at the same conclusion in their
distance kernel — at 14 dimensions, SIMD setup + horizontal reduce wipes
out whatever vectorized arithmetic saves.

#### Why store top-5 as five `long` fields instead of an array?

The hot loop runs 3M times per request worst case (in practice much less
thanks to bbox). Every byte of state that lives in registers instead of
arrays saves cycles. Five `long` distances and five `int` indices fit
comfortably in the JIT's register allocator. The `insertTop` cascade only
shifts as many slots as needed (often one or two).

A `java.util.PriorityQueue` or even a `long[5]` would also work but add
indirection and (for PriorityQueue) object allocation.

---

### Repo layout

```
rinha-de-backend-andre-java/
├── pom.xml                              Maven, Java 25, microhttp + shade uberjar
├── Dockerfile                           Multi-stage Maven → JRE 25
├── Dockerfile.native                    Multi-stage Maven → GraalVM native
├── docker-compose.yml                   nginx + 2× api, cgroup limits (JVM)
├── docker-compose.native.yaml           Same stack, native image
├── nginx.conf                           Round-robin LB, keepalive to upstream
├── src/main/java/com/andre/rinha/
│   ├── App.java                         Entry point: load → preTouch → warmup → start
│   ├── http/
│   │   └── MicrohttpServer.java         microhttp NIO event loop: /ready + /fraud-score
│   ├── json/
│   │   ├── Payload.java                 Immutable record
│   │   └── JsonReader.java              Hand-rolled cursor parser, ISO parser inline
│   ├── vector/
│   │   ├── Vectorizer.java              14-dim normalization (no ZonedDateTime alloc)
│   │   ├── Dataset.java                 Load → short[] + centroids + bbox + offsets
│   │   ├── KnnSearcher.java             IVF + bbox-repair (exact k-NN)
│   │   └── KMeans.java                  Lloyd's + k-means++ (build-time only)
│   └── prep/
│       └── DatasetBuilder.java          CLI: json.gz → int16 + bbox + offsets
└── src/test/java/com/andre/rinha/
    ├── JsonReaderTest.java
    ├── VectorizerTest.java
    └── KnnSearcherIvfRecallTest.java    100 % vs float32 ground truth
```

---

### How we got here: v1 to v9

Each branch is a self-contained snapshot. The Docker score column is the
final number under rinha rules (1 vCPU + 350 MB).

| Branch | What changed | Score | Notes |
|---|---|---|---|
| `v1` | Scalar baseline, JDK HttpServer, brute-force k-NN | +2742 (host) | No Docker yet |
| `v2` | Docker stack + nginx + cgroup limits | −6000 | First reality check |
| `v3` | Explicit Vector API with KNN_MODE A/B switch | −6000 | SIMD not enough |
| `v4` | int8 quantization, 168 MB → 42 MB | −6000 | Memory budget met |
| `v5` | IVF k-means, NPROBE=3 approximation | **+2600** | First positive score |
| `v6` | GraalVM `native-image` attempt | −6000 | Regression. Documented. |
| `v7` | int16 + bbox-repair, 2 workers, random warmup | −3254 (rinha) | First exact k-NN under budget |
| `v7.1` | Virtual threads | −6000 (rinha) | More concurrency hurt |
| `v7.2` | 4 workers + realistic warmup | −6000 (rinha) | More workers hurt |
| `v7.3` | 2 workers + realistic warmup | **+4049.51 (rinha)** | The combination unlocked it |
| `v8` | GraalVM native + alloc-free hot path + heap pinning + sort | +2718.6 (rinha) | Pinning + sort regressed −1338 |
| `v8.1` | Reverted sort + heap pinning, kept native + alloc-free + pre-touch | **+4056.85 (rinha)** | Recovered the ground |
| `v9` | microhttp NIO event loop replaces JDK HttpServer; drop JDK path + instrumentation | _pending_ | Single-thread non-blocking loop; awaiting rinha number |

Three stories from this list are worth telling.

**v2 was the wake-up call.** The host JVM did +2742 on my laptop. The
Docker stack with rinha-spec resource limits did −6000. The host
measurements were inflating real performance by about 10×. Until v2 every
optimization was being measured against the wrong baseline.

**v6 was the most expensive lesson.** I expected GraalVM `native-image` to
shave 30-50 % off p99 by killing JIT warmup. The build worked on the first
try (no reflection config needed). The runtime regressed by 4×.

Root cause: GraalVM 25's `native-image` advertises experimental Vector API
support, but the supported operations list ("load, store, basic arithmetic,
reduce, compare, blend") doesn't include the `convertShape` and `fma` that
v5's hot path used. They fell back to a slow path that the JIT didn't
need. Switching the build flag on or off changed nothing material. v7
dropped the Vector API entirely, partly so a future native-image attempt
might actually work.

**v7 → v7.3 was the JIT cliff.** Locally v7 looked great. The official
rinha run hit 45.9 % errors with p99 pinned against the 2001 ms k6
timeout. Two more configurations made it worse before the right
combination clicked.

The bug was structural. v7's warmup ran 1000 random-vector searches at
startup, but random queries bbox-prune to nothing and never exercise the
cluster-scan + bbox-repair hot path. When real fraud queries arrived they
hit code C2 hadn't compiled yet, ran 10-100× slower, the queue overflowed
past the 2001 ms cap, and 23 000 requests timed out. The k-NN was correct
on every request it answered (0 false positives, 0 false negatives across
all 4 runs); throughput was the bottleneck the whole time.

The fix is two things together. **Realistic warmup** samples 2000 dataset
vectors with light noise as queries, forcing C2 to compile the same
bytecode real traffic will hit, before `/ready` returns 200. **WORKERS=2**
matches the rinha runner's CPU budget: with the 0.45 CPU per container,
more workers just split that budget across more contenders and raise
per-request latency. v7.1 (virtual threads) and v7.2 (4 workers) both
regressed for exactly that reason. v7.3 lands at p99 89 ms and 0 % errors.

**v8 was the cost of a "smart" optimization that wasn't.** I rebuilt with
GraalVM `native-image` (now that v7 has no Vector API, the native build
finally pays off), killed every per-request allocation on the parse and
respond paths, and added bbox-repair sorted by centroid distance so the
top-5 tightens earlier. Local k6 showed `-52 %` on the 800-RPS average and
`-49 %` on p95. The rinha submission dropped the score by 1338 points.

What I missed: local k6 does not punish per-request CPU overhead the way
the rinha runner does. The Mac Mini 2014 used for the official test has
genuinely scarce CPU (1 vCPU split across the whole stack) and aged
DRAM, and any extra microsecond per request amplifies through the worker
thread-pool queue into seconds of tail latency. The `Arrays.sort` on bbox
repair cost ~5 µs per call ; harmless locally, lethal there. Heap pinning
at `-R:MinHeapSize=80m` made it worse — pre-committed 80 MB of heap plus
the 84 MB dataset put us at 164 MB inside a 165 MB container, with zero
headroom for working-set spikes.

**v8.1 backed out both.** Reverted the sort. Removed the heap pinning.
Kept the native-image build, the alloc-free pipeline, the pre-touch loop,
and the per-span tracing endpoint. Score recovered to +4056.85, p99
87.73 ms — basically tied with v7.3 but on a native binary instead of JVM.
The journey from v8 to v8.1 cemented two rules: local k6 is for finding
catastrophes, not marginal wins; and never bundle multiple changes into
one submission, because when something regresses you can't tell which
piece did it.

**v9 attacks the transport, not the search.** Comparing against the top-14
jvmoonshot solution showed the detection was already identical (both score
+3000) — the entire gap was p99 latency. The JDK `com.sun.net.httpserver` runs
thread-per-request and, on 0.45 CPU, contends with itself; the tail balloons
under the k6 ramp. v9 swaps it for microhttp, a ~500-line single-threaded
non-blocking NIO event loop, and removes the now-dead JDK path, the /stats
instrumentation, and the PGO scripts (net −835 lines). Validated locally and
under native-image; the rinha number is pending its preview run.

For each branch's detailed numbers, the commit message has them.

---

### Future work

Things worth trying. v9 priorities come from a deep read of
[jvmoonshot-xxvi](https://github.com/gabsoftware/jvmoonshot-xxvi)
(top-14 rinha JVM solution at p99 1.36 ms).

**Phase 1 — keep architecture, close gaps with jvmoonshot:**

- **Real HTTP warmup driver**. Before `/ready` returns 200, send 15 000
  synthetic POST requests through the local socket. Warms the TCP accept
  path, header parsing, response write, kernel page cache — none of
  which the current internal `App.warmup()` reaches.
- **STRIDE=16 padding** on `vectors[]` and `centroids[]`. Each vector
  aligned to one 64 B cache line; halves cache misses on cold cluster
  scans.
- **Direct-mapped date cache** in the Vectorizer. Most requests cluster
  around a small set of calendar dates; a 64-slot LUT amortizes the
  Gregorian conversion.
- **Merchant ID bitmask** (`MERC-NNN` → int 0-99, `known_merchants` → long
  bitmask). Membership test becomes one bit test instead of a linear
  String scan, when codes fit. Partial — the contest uses MERC-000 to
  MERC-099, so 64-bit bitmask covers ~64 % of merchants.
- **Branchless top-K insert** — replace the `if-else if` cascade in
  `KnnSearcher.insertTop` with a chain of ternaries that the AOT compiler
  emits as CMOVs.

**Phase 2 — memory model:**

- **mmap the dataset off-heap**. The 84 MB `vectors[]` array becomes a
  memory-mapped file backed by the OS page cache. Heap shrinks to under
  10 MB, GC pressure drops to essentially zero.

**Phase 3 — algorithmic and architectural rewrites (v10 territory):**

- **KD-tree with BBF + epsilon-relaxation + refine-boundary**. Replaces
  the IVF + bbox-repair index. Probably 20-40 % faster on KNN; 2 000
  lines of code to port and tune.
- **Custom NIO HTTP server** — _landed in v9 via microhttp_ (single-threaded
  non-blocking event loop). A fully hand-rolled zero-alloc loop with the
  `selectNow()` keep-alive trick could shave a few µs more.
- **Rust load balancer with `SCM_RIGHTS` FD passing**. Replaces nginx
  data path. Saves ~50-100 µs per request.

The `submission` branch implements only the dataset bake. The rest is on
this `main` branch, organized by version.

---
---

## Português

Uma solução em Java para o desafio Rinha de Backend 2026. Dado um payload
de transação de cartão, a API decide aprovar ou negar encontrando as 5
transações mais similares num conjunto de referência de 3 milhões de
vetores e votando nas labels de fraude.

Tudo corre dentro dos limites do desafio: **1 vCPU e 350 MB de memória
distribuídos entre 2 instâncias da API e 1 load balancer**.

Sem Spring, sem Jackson, sem Helidon. JDK 25, um servidor HTTP NIO de thread
única não-bloqueante ([microhttp](https://github.com/ebarlas/microhttp), ~500
LOC, zero deps), um parser JSON escrito à mão, e um k-NN IVF com poda por
bounding box escrito do zero. ~1500 linhas de Java no total.

### Score nas regras da rinha

| Métrica | Valor |
|---|---|
| Score final | **+4056.85** |
| p99 | 87.73 ms |
| Taxa de falha | 0 % |
| Falsos positivos | 0 / 54 054 |
| Falsos negativos | 0 / 54 054 |
| Detection score | +3000 (máximo) |
| p99 score | +1056.85 |

Imagem: `anddev741/rinha-fraud-java:v8.1` no Docker Hub.
Commit da submissão: [`55d7c93`](../../tree/submission).

> **A `main` agora é v9** (event loop NIO microhttp — ver a jornada abaixo). O
> score acima é o último resultado *confirmado* na rinha (v8.1); o número da v9
> ainda está pendente da prévia.

O k-NN é exato: mesmo top-5 que brute force em float32, mas com a maioria
do dataset nunca varrido. A v8 trocou compilação JVM por GraalVM
`native-image`, eliminou alocações por pedido na pipeline parse / vectorize
/ resposta, e adicionou pre-touch das páginas do dataset no startup para
que o primeiro pedido nunca pague page-fault. Sob as regras da rinha
(1 vCPU, 350 MB, 2 instâncias atrás do nginx) cada pedido respondido é
classificado correctamente, e nenhum dá timeout.

Este repo também é um registo de aprendizagem. Cada branch (`v1` a
`v9`) é um snapshot funcional de uma optimização, com scores medidos.
A jornada inclui regressões documentadas — v6 (Vector API + native-image),
v7.1 (virtual threads), v8 (heap pinning + overhead do sort). Cada uma
ensinou algo concreto; a secção da jornada no fim deste README conta
essas histórias.

---

### Quick start

Precisas de Docker, Docker Compose, e [k6](https://k6.io) para o teste de
carga. O dataset é construído dentro da imagem, não há passo Java no host.

Há dois compose files disponíveis (ambos rodam o servidor microhttp da v9):
- `docker-compose.yml` — stack JVM (`Dockerfile`)
- `docker-compose.native.yaml` — stack native-image
  (`Dockerfile.native`, ~6 min de build na primeira vez)

```bash
# 1. Build da imagem. Na primeira vez vai buscar o references.json.gz ao
#    repo da rinha e correr o DatasetBuilder, ~3 min para JVM / ~6 min
#    para native. Depois fica em cache.
docker compose -f docker-compose.native.yaml build

# 2. Subir o stack
docker compose -f docker-compose.native.yaml up -d

# 3. Esperar pelo /ready e correr o teste oficial
until curl -sf http://localhost:9999/ready > /dev/null; do sleep 1; done
git clone https://github.com/zanfranceschi/rinha-de-backend-2026.git /tmp/rinha
cd /tmp/rinha
k6 run test/test.js
cat test/results.json
```

A branch `submission` traz um `docker-compose.yml` que faz pull da imagem
`anddev741/rinha-fraud-java:v8.1` do Docker Hub, para que o motor de teste
da rinha não precise de rebuild.

### Como a busca funciona

O algoritmo é **IVF (Inverted File) com reparo por bounding box**.

Em build time, o `DatasetBuilder` corre k-means com K=256 clusters sobre
os 3M vetores float. Cada vetor é quantizado para int16 (multiplicado por
10 000, o que preserva 4 dígitos decimais de precisão) e o array é
reordenado para que o cluster 0 ocupe os primeiros bytes, o cluster 1 os
seguintes, e por aí. Cada cluster também recebe uma bounding box de 14
dimensões: o menor e o maior valor de cada dim no cluster.

Em query time, a busca faz cinco coisas:

1. Quantiza a query float para int16, na mesma escala dos vetores.
2. Calcula a distância quadrada da query a todos os 256 centróides. Escolhe
   o cluster mais próximo.
3. Varre esse cluster, construindo um max-heap top-5.
4. Para cada outro cluster, calcula o lower bound da distância da query à
   bounding box. Se o lower bound já bate o pior do top-5 atual, salta o
   cluster. Caso contrário, varre-o.
5. Conta quantos do top-5 final são fraude e divide por 5.

O lower bound da bbox é um daqueles truques que parece bom demais até
fazeres a derivação. Para cada dim `d`, a distância mínima possível do
valor da query `q[d]` a qualquer vetor do cluster `c` nessa dim é zero se
`bbox_min[c][d] ≤ q[d] ≤ bbox_max[c][d]`, caso contrário é a distância à
fronteira mais próxima. Soma os quadrados e tens um lower bound estrito
para `distance(q, qualquer_vetor_de_c)`. Se esse bound já excede o pior
do top-5, nenhum vetor em `c` pode ganhar, e saltas 12k vetores com um
único teste de 14 dims.

Na prática a primeira varredura produz um top-5 apertado e a bbox rejeita
os outros 200+ clusters. A busca é exata (sem perda de recall) mas
comporta-se como uma pequena fração do brute force.

---

### Por que estas escolhas e não outras

#### Por que não Spring / Helidon / Netty?

O desafio corre em 1 vCPU. Um framework completo não compra nada aqui e custa
RSS. As v1–v8.1 usaram o `com.sun.net.httpserver` da JDK, mas o modelo
thread-per-request dele disputa consigo mesmo nos 0.45 CPU de cada container,
inflando o tail do p99. A v9 trocou para o microhttp — um event loop NIO de
thread única não-bloqueante de ~500 linhas (ainda não é framework, ainda zero
deps pesadas): uma thread de loop roda a todo vapor sem auto-disputa, que é
exatamente o que uma CPU fracionária quer. Netty ou Vert.x adicionariam a mesma
disputa de threads mais uma superfície bem maior.

#### Por que parser JSON escrito à mão?

Jackson usa reflection e aloca mapas intermediários. Para um schema fixo
de 10 campos, isso custa 50 a 100 µs por request e alguns MB de heap em
fly. Um parser cursor especializado neste payload corre em ~5 µs sem
alocação além do record `Payload`.

A contrapartida é fragilidade. Se o schema mudar, o parser quebra. O
contrato do desafio é fixo, então é uma escolha consciente.

#### Por que int16 em vez de int8?

Tentámos int8 na v4 com escala min/max por dimensão. A matemática
transforma a distância quantizada num L2 ponderado em float space, onde
booleanos (range 1) recebem um peso e a dim sentinela (range 2) recebe um
quarto do peso. A concordância de top-5 em dados sintéticos caiu para 14 %.

Depois tentámos int8 com uma escala global única (v5). Preservou ordenação
mas limitou a resolução a ~1/128 por dim, e a aproximação IVF com nprobe=3
custou cerca de 1300 pontos de detection em runs reais.

int16 com escala `× 10 000` cabe em `short` (±32 767) com margem, dá
~1/10 000 de resolução por dim, e permite-nos correr k-NN exato com
bbox repair. Custo de memória: 84 MB em vez de 42 MB. Ainda dentro do
orçamento.

#### Por que não Vector API?

Tentámos três vezes: v3, v6, e uma terceira durante o ciclo v8. O JIT
(v3) deu ganho pequeno em microbenchmarks sem restrições mas nenhuma
melhoria sob cgroup. GraalVM `native-image` (v6 e v8) regrediu
catastroficamente — o `ShortVector.convertShape(S2I)` não é
escape-analyzed pelo C2 nem pelo Substrate VM (o compilador AOT por
trás do native-image), então cada chamada aloca ~787 KB de wrappers.
O benchmark de 800 RPS foi de 44 ms avg para mais de 400 ms com 67 %
das requisições a falhar.

O loop scalar manual unrolled actual, com `if (dist > worst) continue`
entre cada uma das 14 dimensões, fica competitivo em todos os cenários.
A maioria dos candidatos sai após 2-4 dims porque estão longe da query,
e o compilador AOT emite código straight-line apertado para o loop de
centróides. A solução top-14 da jvmoonshot chega à mesma conclusão no
kernel de distância — a 14 dimensões, o setup + reduce horizontal do
SIMD anula o que a aritmética vectorizada poupa.

#### Por que top-5 em cinco campos `long` em vez de array?

O loop hot corre 3M vezes por request no pior caso (na prática muito
menos graças à bbox). Cada byte de estado que vive em registos em vez de
arrays poupa ciclos. Cinco distâncias `long` e cinco índices `int` cabem
confortavelmente no register allocator do JIT. A cascata `insertTop` só
desloca os slots necessários (frequentemente um ou dois).

Uma `java.util.PriorityQueue` ou mesmo um `long[5]` também funcionariam
mas adicionam indirecção e (para a PriorityQueue) alocação de objectos.

---

### Layout do repo

```
rinha-de-backend-andre-java/
├── pom.xml                              Maven, Java 25, microhttp + shade uberjar
├── Dockerfile                           Multi-stage Maven → JRE 25
├── Dockerfile.native                    Multi-stage Maven → GraalVM native
├── docker-compose.yml                   nginx + 2× api, limites cgroup (JVM)
├── docker-compose.native.yaml           Mesmo stack, imagem native
├── nginx.conf                           LB round-robin, keepalive ao upstream
├── src/main/java/com/andre/rinha/
│   ├── App.java                         Entry: load → preTouch → warmup → start
│   ├── http/
│   │   └── MicrohttpServer.java         Event loop NIO microhttp: /ready + /fraud-score
│   ├── json/
│   │   ├── Payload.java                 Record imutável
│   │   └── JsonReader.java              Parser cursor manual, ISO inline
│   ├── vector/
│   │   ├── Vectorizer.java              Normalização 14d (sem alloc ZonedDateTime)
│   │   ├── Dataset.java                 Load → short[] + centróides + bbox + offsets
│   │   ├── KnnSearcher.java             IVF + bbox-repair (k-NN exato)
│   │   └── KMeans.java                  Lloyd + k-means++ (build-time)
│   └── prep/
│       └── DatasetBuilder.java          CLI: json.gz → int16 + bbox + offsets
└── src/test/java/com/andre/rinha/
    ├── JsonReaderTest.java
    ├── VectorizerTest.java
    └── KnnSearcherIvfRecallTest.java    100 % vs float32 ground truth
```

---

### A jornada: v1 a v9

Cada branch é um snapshot independente. A coluna Score é o número final
sob as regras da rinha (1 vCPU + 350 MB).

| Branch | O que mudou | Score | Notas |
|---|---|---|---|
| `v1` | Baseline scalar, JDK HttpServer, brute-force k-NN | +2742 (host) | Ainda sem Docker |
| `v2` | Stack Docker + nginx + limites cgroup | −6000 | Primeira reality check |
| `v3` | Vector API explícita com switch A/B KNN_MODE | −6000 | SIMD não chega |
| `v4` | Quantização int8, 168 MB → 42 MB | −6000 | Orçamento de memória cumprido |
| `v5` | IVF k-means, aproximação NPROBE=3 | **+2600** | Primeiro score positivo |
| `v6` | Tentativa GraalVM `native-image` | −6000 | Regressão. Documentada. |
| `v7` | int16 + bbox-repair, 2 workers, warmup aleatório | −3254 (rinha) | Primeiro k-NN exato dentro do orçamento |
| `v7.1` | Virtual threads | −6000 (rinha) | Mais concorrência piorou |
| `v7.2` | 4 workers + warmup realista | −6000 (rinha) | Mais workers piorou |
| `v7.3` | 2 workers + warmup realista | **+4049.51 (rinha)** | A combinação desbloqueou |
| `v8` | GraalVM native + alloc-free + heap pinning + sort | +2718.6 (rinha) | Pinning + sort regrediram −1338 |
| `v8.1` | Reverteu sort + heap pinning, manteve native + alloc-free + pre-touch | **+4056.85 (rinha)** | Recuperou o terreno |
| `v9` | event loop NIO microhttp substitui o JDK HttpServer; remove path JDK + instrumentação | _pendente_ | Loop de thread única não-bloqueante; aguardando número da rinha |

Três histórias desta lista vale a pena contar.

**A v2 foi o despertador.** A JVM do host fez +2742 no meu laptop. O stack
Docker com os limites de recursos da rinha fez −6000. As medições no host
estavam a inflacionar a performance real em cerca de 10×. Até à v2 todas
as optimizações estavam a ser medidas contra a baseline errada.

**A v6 foi a lição mais cara.** Esperava que o GraalVM `native-image`
tirasse 30-50 % do p99 ao matar o warmup do JIT. O build funcionou à
primeira (sem precisar de config de reflexão). O runtime regrediu 4×.

Causa raiz: o `native-image` do GraalVM 25 anuncia suporte experimental à
Vector API, mas a lista de operações suportadas ("load, store, basic
arithmetic, reduce, compare, blend") não inclui o `convertShape` e o `fma`
que o hot path da v5 usava. Caíam para um caminho lento que o JIT não
precisava. Ligar ou desligar a flag não mudou nada material. A v7 abandonou
a Vector API, em parte para que uma futura tentativa de native-image possa
de facto funcionar.

**A v7 → v7.3 foi o JIT cliff.** Localmente a v7 parecia óptima. A
corrida oficial da rinha deu 45.9 % de erros com p99 colado ao timeout
de 2001 ms do k6. Duas configurações seguintes pioraram antes da
combinação certa.

O bug era estrutural. O warmup da v7 corria 1000 buscas com vetores
aleatórios ao arranque, mas queries aleatórias bbox-prune para nada e
nunca exercitam o hot path de cluster-scan + bbox-repair. Quando as
queries reais de fraude chegaram, bateram em bytecode que o C2 ainda
não tinha compilado, correram 10-100× mais lento, a fila ultrapassou
os 2001 ms e 23 000 pedidos deram timeout. O k-NN estava correcto em
cada pedido que respondeu (0 falsos positivos, 0 falsos negativos nas
4 corridas); o gargalo foi sempre throughput.

A correcção são duas coisas em conjunto. **Warmup realista** sampleia
2000 vetores do dataset com ruído ligeiro como queries, forçando o C2
a compilar o mesmo bytecode que o tráfego real vai bater, antes do
`/ready` responder 200. **WORKERS=2** alinha com o orçamento de CPU do
runner da rinha: com 0.45 CPU por container, mais workers só dividem
esse orçamento por mais concorrentes e sobem a latência por pedido.
A v7.1 (virtual threads) e a v7.2 (4 workers) regrediram exactamente
por isso. A v7.3 fecha em p99 89 ms e 0 % de erros.

**A v8 foi o custo duma optimização "smart" que não foi.** Reconstruí com
GraalVM `native-image` (agora que a v7 não tem Vector API, o build native
finalmente compensava), matei todas as alocações por pedido na pipeline
parse / response, e adicionei o bbox-repair ordenado por distância de
centróide para que o top-5 aperte mais cedo. O k6 local mostrou `-52 %` na
média a 800 RPS e `-49 %` no p95. A submissão à rinha baixou o score em
1338 pontos.

O que eu não vi: o k6 local não pune o overhead de CPU por pedido como o
runner da rinha pune. O Mac Mini 2014 do teste oficial tem CPU genuinamente
escassa (1 vCPU para todo o stack) e DRAM envelhecida, e qualquer
microsegundo extra por pedido amplifica através da fila do thread pool em
segundos de tail latency. O `Arrays.sort` no bbox-repair custou ~5 µs por
chamada ; inofensivo localmente, letal lá. Pinar o heap em
`-R:MinHeapSize=80m` piorou — pré-committed 80 MB de heap mais os 84 MB do
dataset puseram-nos em 164 MB dentro de um container de 165 MB, com zero
margem para spikes do working-set.

**A v8.1 reverteu ambos.** Tirou o sort. Tirou o heap pinning. Manteve o
build native-image, a pipeline alloc-free, o pre-touch, e o endpoint de
tracing por span. Score recuperou para +4056.85, p99 87.73 ms —
basicamente empata com a v7.3 mas em binário native em vez de JVM. A
jornada da v8 para a v8.1 cimentou duas regras: o k6 local serve para
encontrar catástrofes, não para ganhos marginais; e nunca empacotar
várias mudanças numa submissão, porque quando algo regride não consegues
saber qual foi.

**A v9 ataca o transporte, não a busca.** Comparar com a solução top-14 da
jvmoonshot mostrou que a detecção já era idêntica (ambas +3000) — todo o gap
era latência de p99. O `com.sun.net.httpserver` da JDK é thread-per-request e,
em 0.45 CPU, disputa consigo mesmo; o tail dispara sob a rampa do k6. A v9 troca
por microhttp, um event loop NIO de thread única não-bloqueante de ~500 linhas,
e remove o path JDK morto, a instrumentação /stats e os scripts de PGO
(−835 linhas líquidas). Validado localmente e sob native-image; o número da
rinha está pendente da prévia.

Para os números detalhados de cada branch, a mensagem de commit tem-nos
todos.

---

### Trabalho futuro

Prioridades da v9 vêm de um estudo profundo da
[jvmoonshot-xxvi](https://github.com/gabsoftware/jvmoonshot-xxvi) (top-14
da rinha JVM, p99 1.36 ms).

**Fase 1 — manter a arquitectura, fechar gaps com a jvmoonshot:**

- **Driver de warmup HTTP real**. Antes do `/ready` devolver 200, enviar
  15 000 pedidos POST sintéticos pelo socket local. Aquece o caminho de
  TCP accept, parsing de headers, write de resposta, page cache do kernel —
  nada disto é tocado pelo `App.warmup()` interno actual.
- **Padding STRIDE=16** em `vectors[]` e `centroids[]`. Cada vetor alinhado
  a uma cache line de 64 B; reduz para metade os cache misses em cluster
  scans frios.
- **Cache direct-mapped de datas** no Vectorizer. A maioria dos pedidos
  agrupa-se num conjunto pequeno de datas; uma LUT de 64 slots amortiza
  a conversão Gregoriana.
- **Bitmask de merchant IDs** (`MERC-NNN` → int 0-99, `known_merchants`
  → long bitmask). Teste de membership vira um bit test em vez de scan
  linear de Strings, quando os códigos cabem. Parcial — o concurso usa
  MERC-000 a MERC-099, então o bitmask de 64 bits cobre ~64 % dos
  merchants.
- **Top-K com inserts branchless** — substituir a cascata `if-else if`
  no `KnnSearcher.insertTop` por uma cadeia de ternários que o compilador
  AOT emite como CMOVs.

**Fase 2 — modelo de memória:**

- **mmap do dataset off-heap**. O array `vectors[]` de 84 MB torna-se um
  ficheiro mapeado em memória servido pelo page cache do OS. O heap
  encolhe para menos de 10 MB, pressão de GC essencialmente zero.

**Fase 3 — reescritas algorítmicas e arquitecturais (território da v10):**

- **KD-tree com BBF + epsilon-relaxation + refine-boundary**. Substitui o
  index IVF + bbox-repair. Provavelmente 20-40 % mais rápido no KNN; 2 000
  linhas de código para portar e afinar.
- **Servidor NIO HTTP custom** — _entregue na v9 via microhttp_ (event loop
  de thread única não-bloqueante). Um loop totalmente artesanal zero-alloc com
  o truque do `selectNow()` em keep-alive ainda pouparia uns µs.
- **Load balancer em Rust com FD passing via `SCM_RIGHTS`**. Substitui o
  data path do nginx. Poupa ~50-100 µs por pedido.

A branch `submission` implementa apenas o bake do dataset. O resto vive
nesta branch `main`, organizado por versão.
