import json


def generate_example_input():
    return {
        "model": "llama-7b",
        "input": "Hello from generate_input",
        "options": {"max_tokens": 32, "temperature": 0.8, "top_p": 0.95},
    }


if __name__ == "__main__":
    print(json.dumps(generate_example_input(), indent=2))
