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

No Spring, no Jackson, no Helidon. JDK 25, the built-in `HttpServer`, a
hand-rolled JSON parser, and a from-scratch IVF k-NN with bounding-box
pruning. ~1500 lines of Java total.

### Score under rinha rules

| Metric | Value |
|---|---|
| Final score | **+4049.51** |
| p99 latency | 89.23 ms |
| Failure rate | 0 % |
| False positives | 0 / 54 060 |
| False negatives | 0 / 54 060 |
| Detection score | +3000 (maximum) |
| p99 score | +1049.51 |

Image: `anddev741/rinha-fraud-java:v7.3` on Docker Hub.
Submission commit: [`5e68831`](../../tree/submission).

The k-NN is exact: same top-5 as float32 brute force, but with most of the
dataset never scanned. Under the rinha rules (1 vCPU, 350 MB, 2 instances
behind nginx) every answered request is classified correctly, and no
requests time out.

This repo is also a learning trail. Each branch (`v1` through `v7`) is a
working snapshot of one optimization, with measured scores. `v6` is a
documented regression (GraalVM `native-image` did not pay off for our code).
The first three rinha submissions also regressed before v7.3 found the
fix; that story is in the journey section at the bottom of this README.

---

### Quick start

You need Docker, Docker Compose, and [k6](https://k6.io) for the load test.
The dataset is built inside the image, so there is no Java step on the host.

```bash
# 1. Build the image. The first run fetches references.json.gz from the rinha
#    repo and runs DatasetBuilder, so it takes ~3 minutes. Cached after that.
docker compose build

# 2. Start the stack
docker compose up -d

# 3. Wait for /ready and run the official test
until curl -sf http://localhost:9999/ready > /dev/null; do sleep 1; done
git clone https://github.com/zanfranceschi/rinha-de-backend-2026.git /tmp/rinha
cd /tmp/rinha
k6 run test/test.js
cat test/results.json
```

The `submission` branch ships a `docker-compose.yml` that pulls the pre-built
image from Docker Hub, so the rinha test engine doesn't have to rebuild.

---

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

The challenge runs on 1 vCPU. The bottleneck is the k-NN, not the HTTP
parsing. `com.sun.net.httpserver.HttpServer` ships with the JDK, sets up in
five lines, and handles thread-per-request with an `ExecutorService`. Adding
a framework buys nothing here and costs RSS.

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

We tried it in v3 and v6. The JIT (v3) got a small win in unconstrained
benchmarks but no improvement under cgroup limits. GraalVM `native-image`
(v6) regressed by ~4× because its experimental `jdk.incubator.vector`
support does not cover the `convertShape(B→F)` and `FloatVector.fma`
operations we needed.

The current scalar manual-unrolled loop, with `if (dist > worst) continue`
between each of the 14 dimensions, ends up being competitive on the JIT
and clearly better in native-image scenarios we didn't end up using. Most
candidates exit after 2-4 dims because they are far from the query.

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
├── pom.xml                              Maven, Java 25, zero runtime deps
├── Dockerfile                           Multi-stage Maven → JRE 25
├── docker-compose.yml                   nginx + 2× api, cgroup limits
├── nginx.conf                           Round-robin LB
├── src/main/java/com/andre/rinha/
│   ├── App.java                         Entry point, loads dataset, starts HTTP
│   ├── http/
│   │   ├── ReadyHandler.java            GET /ready
│   │   └── FraudHandler.java            POST /fraud-score
│   ├── json/
│   │   ├── Payload.java                 Immutable record
│   │   └── JsonReader.java              Hand-rolled cursor parser
│   ├── vector/
│   │   ├── Vectorizer.java              14-dim normalization
│   │   ├── Dataset.java                 Load mmap → short[] + centroids + bbox + offsets
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

### How we got here: v1 to v7.3

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

The v8 attempt (HAProxy + Unix Domain Socket) is not on this list because
it regressed and got dropped. The lesson: localhost TCP under modern Linux
is faster than I expected, and replacing nginx + JDK `HttpServer` with my
own UDS HTTP server cost more in allocations and VT scheduling than UDS
saved in syscalls.

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

For each branch's detailed numbers, the commit message has them.

---

### Future work

Things worth trying that I didn't get to:

- **K=1024 with cluster splitting** (cap max cluster size at ~500). Tighter
  clusters mean faster per-bucket scans and more aggressive bbox pruning.
  The current top of the rinha JVM leaderboard does this and lands at
  p99 ≈ 3.7 ms.
- **GraalVM native-image revisited**. With Vector API gone in v7, the hot
  path is pure scalar, which is what `native-image` handles best. The v6
  build pipeline is on its branch and can be re-applied to v7.
- **PGO** (profile-guided optimization) for native-image. Collect a runtime
  profile under k6 load, rebuild with the profile.
- **Pre-baked HTTP responses**. There are only 6 possible response bodies
  (one per fraud count). Pre-build the byte arrays, write the right one,
  skip the StringBuilder + UTF-8 encode per request.

The `submission` branch implements only the dataset bake. The rest stays
as an exercise.

---
---

## Português

Uma solução em Java para o desafio Rinha de Backend 2026. Dado um payload
de transação de cartão, a API decide aprovar ou negar encontrando as 5
transações mais similares num conjunto de referência de 3 milhões de
vetores e votando nas labels de fraude.

Tudo corre dentro dos limites do desafio: **1 vCPU e 350 MB de memória
distribuídos entre 2 instâncias da API e 1 load balancer**.

Sem Spring, sem Jackson, sem Helidon. JDK 25, o `HttpServer` que vem na
JDK, um parser JSON escrito à mão, e um k-NN IVF com poda por bounding box
escrito do zero. ~1500 linhas de Java no total.

### Score nas regras da rinha

| Métrica | Valor |
|---|---|
| Score final | **+4049.51** |
| p99 | 89.23 ms |
| Taxa de falha | 0 % |
| Falsos positivos | 0 / 54 060 |
| Falsos negativos | 0 / 54 060 |
| Detection score | +3000 (máximo) |
| p99 score | +1049.51 |

Imagem: `anddev741/rinha-fraud-java:v7.3` no Docker Hub.
Commit da submissão: [`5e68831`](../../tree/submission).

O k-NN é exato: mesmo top-5 que brute force em float32, mas com a maioria
do dataset nunca varrido. Sob as regras da rinha (1 vCPU, 350 MB, 2
instâncias atrás do nginx) cada pedido respondido é classificado
correctamente, e nenhum dá timeout.

Este repo também é um registo de aprendizagem. Cada branch (`v1` a `v7`)
é um snapshot funcional de uma optimização, com scores medidos. `v6` é
uma regressão documentada (o `native-image` do GraalVM não compensou).
As três primeiras submissões à rinha também regrediram antes da v7.3
encontrar a solução; essa história está na secção da jornada no fim
deste README.

---

### Quick start

Precisas de Docker, Docker Compose, e [k6](https://k6.io) para o teste de
carga. O dataset é construído dentro da imagem, não há passo Java no host.

```bash
# 1. Build da imagem. Na primeira vez vai buscar o references.json.gz ao
#    repo da rinha e correr o DatasetBuilder, demora ~3 minutos. Depois fica
#    em cache.
docker compose build

# 2. Subir o stack
docker compose up -d

# 3. Esperar pelo /ready e correr o teste oficial
until curl -sf http://localhost:9999/ready > /dev/null; do sleep 1; done
git clone https://github.com/zanfranceschi/rinha-de-backend-2026.git /tmp/rinha
cd /tmp/rinha
k6 run test/test.js
cat test/results.json
```

A branch `submission` traz um `docker-compose.yml` que faz pull da imagem
pré-construída do Docker Hub, para que o motor de teste da rinha não precise
de rebuild.

---

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

O desafio corre em 1 vCPU. O gargalo é o k-NN, não o parsing HTTP. O
`com.sun.net.httpserver.HttpServer` vem na JDK, sobe em cinco linhas, e
trata thread-per-request com um `ExecutorService`. Adicionar um framework
não compra nada aqui e custa RSS.

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

Tentámos na v3 e na v6. O JIT (v3) deu um ganho pequeno em benchmarks sem
restrições mas nenhuma melhoria sob cgroup. O `native-image` do GraalVM
(v6) regrediu ~4× porque o seu suporte experimental a
`jdk.incubator.vector` não cobre as operações `convertShape(B→F)` e
`FloatVector.fma` de que precisávamos.

O loop scalar manual unrolled actual, com `if (dist > worst) continue`
entre cada uma das 14 dimensões, fica competitivo no JIT e claramente
melhor para cenários native-image. A maioria dos candidatos sai após 2-4
dims porque estão longe da query.

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
├── pom.xml                              Maven, Java 25, zero deps de runtime
├── Dockerfile                           Multi-stage Maven → JRE 25
├── docker-compose.yml                   nginx + 2× api, limites cgroup
├── nginx.conf                           LB round-robin
├── src/main/java/com/andre/rinha/
│   ├── App.java                         Entry point, carrega dataset, sobe HTTP
│   ├── http/
│   │   ├── ReadyHandler.java            GET /ready
│   │   └── FraudHandler.java            POST /fraud-score
│   ├── json/
│   │   ├── Payload.java                 Record imutável
│   │   └── JsonReader.java              Parser cursor manual
│   ├── vector/
│   │   ├── Vectorizer.java              Normalização das 14 dimensões
│   │   ├── Dataset.java                 mmap → short[] + centróides + bbox + offsets
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

### A jornada: v1 a v7.3

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

A tentativa v8 (HAProxy + Unix Domain Socket) não está nesta lista porque
regrediu e foi abandonada. A lição: o TCP em loopback no Linux moderno é
mais rápido do que eu esperava, e substituir nginx + `HttpServer` da JDK
por um servidor HTTP UDS escrito por mim custou mais em alocações e
scheduling de virtual threads do que o UDS poupou em syscalls.

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

Para os números detalhados de cada branch, a mensagem de commit tem-nos
todos.

---

### Trabalho futuro

Coisas que vale a pena tentar e que não cheguei a fazer:

- **K=1024 com split de clusters** (cap no tamanho máximo em ~500). Clusters
  mais apertados significam scans por bucket mais rápidos e poda de bbox
  mais agressiva. O topo do ranking JVM da rinha faz isto e fica em
  p99 ≈ 3.7 ms.
- **GraalVM native-image revisitado**. Com a Vector API fora na v7, o hot
  path é puro scalar, que é o que o `native-image` melhor lida. O pipeline
  de build da v6 está na sua branch e pode ser reaplicado à v7.
- **PGO** (profile-guided optimization) para o native-image. Coletar um
  profile em runtime sob carga k6, recompilar com o profile.
- **Respostas HTTP pré-prontas**. Só há 6 corpos de resposta possíveis (um
  por contagem de fraude). Pré-construir os byte arrays, escrever o
  correcto, saltar o StringBuilder + encode UTF-8 por request.

A branch `submission` implementa apenas o bake do dataset. O resto fica
como exercício.
