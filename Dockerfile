# syntax=docker/dockerfile:1.7

# ─────────────────────────────────────────────────────────────
# Stage 1 — build with Maven and JDK 25
# ─────────────────────────────────────────────────────────────
FROM maven:3.9-eclipse-temurin-25-alpine AS build

WORKDIR /app

# Copy pom first for better layer caching: Maven resolves deps once
# and the layer is reused as long as pom.xml doesn't change.
COPY pom.xml .
RUN mvn -q -B -e dependency:go-offline

# Now copy sources and build
COPY src ./src
RUN mvn -q -B -e -DskipTests package

# ─────────────────────────────────────────────────────────────
# Stage 2 — runtime with JRE 25 only
# ─────────────────────────────────────────────────────────────
FROM eclipse-temurin:25-jre-alpine

WORKDIR /app

# Copy only the executable JAR from the build stage
COPY --from=build /app/target/rinha-fraud.jar app.jar

# Defaults — overridable via env vars
ENV PORT=9999 \
    DATA_DIR=/data \
    THREADS=2 \
    JAVA_OPTS="-Xmx240m -XX:+UseSerialGC --add-modules=jdk.incubator.vector"

EXPOSE 9999

# Use exec form so the JVM gets PID 1 and receives SIGTERM cleanly
ENTRYPOINT ["sh", "-c", "exec java $JAVA_OPTS -jar app.jar"]
