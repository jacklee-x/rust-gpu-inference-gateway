import httpx
import time

API_URL = "http://127.0.0.1:8080/infer"


def main():
    payload = {
        "model": "llama-7b",
        "input": "Benchmark test request",
        "options": {"max_tokens": 32, "temperature": 0.8, "top_p": 0.95},
    }
    iterations = 10

    with httpx.Client() as client:
        latencies = []
        for i in range(iterations):
            start = time.perf_counter()
            response = client.post(API_URL, json=payload)
            response.raise_for_status()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            print(f"Iteration {i+1}: {latencies[-1]:.2f} ms")

    print(f"Average latency: {sum(latencies) / len(latencies):.2f} ms")


if __name__ == "__main__":
    main()
