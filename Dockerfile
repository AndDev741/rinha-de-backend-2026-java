# syntax=docker/dockerfile:1.7

# ─────────────────────────────────────────────────────────────
# v6 — GraalVM native-image build pipeline
#
# Stage 1: GraalVM 25 community edition with native-image preinstalled.
#          We add Maven and run the v5 dataset pipeline + native-image
#          compilation in this stage. The final binary lives in
#          /app/target/rinha-fraud.
#
# Stage 2: distroless/cc (glibc + minimal libc++ deps). Just the binary.
#          Dynamic linking against system glibc keeps the image small
#          (~30-40 MB) without needing musl gymnastics.
# ─────────────────────────────────────────────────────────────

FROM ghcr.io/graalvm/native-image-community:25 AS build

# Install Maven (the GraalVM image ships with the JDK + native-image but
# no build tool). Using the OS package keeps things tight; we don't need
# the latest Maven.
RUN microdnf install -y maven tar gzip && microdnf clean all

WORKDIR /app

# Copy pom first for layer caching: dependency resolution happens once
# and is reused as long as pom.xml doesn't change.
COPY pom.xml .
RUN mvn -q -B -e dependency:go-offline

# Copy sources and run the full native build.
# -Pnative activates the native-maven-plugin profile defined in pom.xml.
# -DskipTests because tests need surefire which doesn't play well with
# the native-image fork; we run tests separately on the JIT build.
COPY src ./src
RUN mvn -Pnative -q -B -e -DskipTests package

# ─────────────────────────────────────────────────────────────
# Runtime image — debian:12-slim (~75 MB).
# We tried distroless/cc-debian12 first (smaller) but native-image
# pulls in zlib transitively (DatasetBuilder uses GZIPInputStream),
# so the runtime binary needs libz.so.1 which distroless/cc lacks.
# debian:12-slim has the full glibc + zlib + ssl set without going
# all the way to a fat distro.
# ─────────────────────────────────────────────────────────────
FROM debian:12-slim

WORKDIR /app

COPY --from=build /app/target/rinha-fraud /app/rinha-fraud

# Defaults — overridable via env vars
ENV PORT=9999 \
    DATA_DIR=/data \
    THREADS=2

EXPOSE 9999

ENTRYPOINT ["/app/rinha-fraud"]
