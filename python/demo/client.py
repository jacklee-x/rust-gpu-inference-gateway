import argparse
import httpx

API_URL = "http://127.0.0.1:8080/infer"


def main():
    parser = argparse.ArgumentParser(description="Inference demo client")
    parser.add_argument("--model", default="llama-7b")
    parser.add_argument("--input", default="Hello world")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "input": args.input,
        "options": {"max_tokens": 32, "temperature": 0.8, "top_p": 0.95},
    }

    with httpx.Client() as client:
        response = client.post(API_URL, json=payload)
        response.raise_for_status()
        print(response.json())


if __name__ == "__main__":
    main()
