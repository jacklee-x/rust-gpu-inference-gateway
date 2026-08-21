# C++ inference core image (CPU fallback path).
# Build context must be the cpp_inference directory:
#   docker build -t inference-core .

FROM gcc:13-bookworm AS builder
WORKDIR /core
COPY inference_core.cpp ./
# Static-link the C++ runtime: the builder's libstdc++ provides
# GLIBCXX_3.4.32 while debian:bookworm-slim only ships GCC 12's
# (GLIBCXX_3.4.30); dynamic linking made the binary fail at load time.
RUN g++ -std=c++17 -O2 inference_core.cpp -o inference_core -pthread \
    -static-libstdc++ -static-libgcc

FROM debian:bookworm-slim
# wget is used by the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /core/inference_core /usr/local/bin/inference_core
EXPOSE 8081
# Bind 0.0.0.0 so the gateway container can reach the core over the
# compose bridge network (127.0.0.1 would only be reachable locally).
ENTRYPOINT ["inference_core", "--host", "0.0.0.0", "--port", "8081"]