import httpx

def test_infer_endpoint():
    payload = {
        "model": "llama-7b",
        "input": "Test request",
        "options": {"max_tokens": 16},
    }
    response = httpx.post("http://127.0.0.1:8080/infer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "output" in data
