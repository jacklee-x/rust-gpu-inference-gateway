# Kubernetes deployment examples

Manifests for running the gateway + C++ inference core on a Kubernetes
cluster, mirroring `deploy/docker-compose.yml` (same images, ports and
wire protocol).

| File | Contents |
|---|---|
| `00-namespace.yaml` | `inference-gateway` namespace |
| `10-inference-core.yaml` | inference core Deployment + ClusterIP Service (`:8081`) |
| `20-gateway.yaml` | gateway Deployment + ClusterIP Service (`:80`) + HPA (CPU 70%, 2..10 pods) |

## Prerequisites

- A cluster with `kubectl` configured (kind / minikube are fine).
- For the HPA to actually scale, `metrics-server` must be installed
  (kind/minikube addons: `kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml`
  plus the kind insecure-TLS patch, or `minikube addons enable metrics-server`).

## 1. Build and load the images

CI currently only builds images for smoke tests; it does not push them.
Build locally and load into your cluster:

```bash
# gateway (from repo root)
docker build -f deploy/Dockerfile.gateway \
  -t ghcr.io/jacklee-x/rust-gpu-inference-gateway/gateway:latest .

# inference core
docker build -t ghcr.io/jacklee-x/rust-gpu-inference-gateway/inference-core:latest \
  cpp_inference

# kind
kind load docker-image ghcr.io/jacklee-x/rust-gpu-inference-gateway/gateway:latest
kind load docker-image ghcr.io/jacklee-x/rust-gpu-inference-gateway/inference-core:latest

# minikube (alternative)
minikube image load ghcr.io/jacklee-x/rust-gpu-inference-gateway/gateway:latest
minikube image load ghcr.io/jacklee-x/rust-gpu-inference-gateway/inference-core:latest
```

Or push both tags to a registry your cluster can pull from.

## 2. Apply the manifests

```bash
kubectl apply -f deploy/k8s/
```

## 3. Verify

```bash
kubectl -n inference-gateway get pods,hpa
kubectl -n inference-gateway port-forward svc/gateway 8080:80

curl http://127.0.0.1:8080/health          # {"status":"healthy"}
curl http://127.0.0.1:8080/models
```

## Notes

- **HPA vs worker pool**: each gateway pod already adapts its internal
  concurrency between `MIN_CONCURRENCY` and `MAX_CONCURRENCY`; the HPA
  adds cluster-level elasticity by growing the pod count under sustained
  CPU pressure.
- **GPU mode**: for real llama.cpp GPU inference, run llama-server as its
  own Deployment requesting `nvidia.com/gpu` (NVIDIA device plugin +
  `nvidia.com/gpu-driver` runtime required) and point the gateway at it:
  `CORE_URL=http://llama-server:8081`, `CORE_PROTOCOL=llama-chat`.
- **Exposing externally**: switch the gateway Service to `NodePort`/`LoadBalancer`
  or add an Ingress in front of `svc/gateway`.
