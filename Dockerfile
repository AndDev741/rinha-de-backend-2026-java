# syntax=docker/dockerfile:1.7

# ─────────────────────────────────────────────────────────────
# Stage 1 — build the JAR and bake the dataset
#
# We compile the project, fetch references.json.gz from the
# official rinha repo, then run DatasetBuilder to materialize
# the 5 int16/IVF binary files into /app/data.
#
# This step takes ~90s but only runs once per image build.
# The resulting binaries (~84 MB) are copied into the runtime
# stage so containers start in seconds.
# ─────────────────────────────────────────────────────────────
FROM maven:3.9-eclipse-temurin-25-alpine AS build

WORKDIR /app

# Resolve dependencies first so this layer caches as long as
# pom.xml doesn't change.
COPY pom.xml .
RUN mvn -q -B -e dependency:go-offline

COPY src ./src
RUN mvn -q -B -e -DskipTests package

# Fetch the canonical references file from the rinha repo.
# Pinning to main keeps reproducibility tied to the upstream commit.
ADD https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz \
    /app/resources/references.json.gz

# Build the int16 IVF dataset. The output is 5 files under /app/data.
RUN java -cp target/rinha-fraud.jar \
    com.andre.rinha.prep.DatasetBuilder \
    /app/resources/references.json.gz /app/data

# ─────────────────────────────────────────────────────────────
# Stage 2 — runtime with JRE 25 only
#
# Carries the JAR plus the pre-baked /data directory.
# Nothing is computed at startup beyond JIT warmup.
# ─────────────────────────────────────────────────────────────
FROM eclipse-temurin:25-jre-alpine

WORKDIR /app

COPY --from=build /app/target/rinha-fraud.jar app.jar
COPY --from=build /app/data /data

ENV PORT=9999 \
    DATA_DIR=/data \
    THREADS=2 \
    JAVA_OPTS="-Xmx128m -XX:+UseSerialGC"

EXPOSE 9999

# Exec form so the JVM is PID 1 and gets SIGTERM directly.
ENTRYPOINT ["sh", "-c", "exec java $JAVA_OPTS -jar app.jar"]
