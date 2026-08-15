# C++ inference core image (CPU fallback path).
# Build context must be the cpp_inference directory:
#   docker build -t inference-core .

FROM gcc:13-bookworm AS builder
WORKDIR /core
COPY inference_core.cpp ./
RUN g++ -std=c++17 -O2 inference_core.cpp -o inference_core -pthread

FROM debian:bookworm-slim
# wget is used by the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /core/inference_core /usr/local/bin/inference_core
EXPOSE 8081
ENTRYPOINT ["inference_core", "--port", "8081"]