import httpx

def test_infer_endpoint():
    payload = {
        "model": "llama-7b",
        "input": "Test request",
        "options": {"max_tokens": 16},
    }
    # Ensure environment proxy variables do not interfere with local calls (CI or dev machines may set http_proxy)
    with httpx.Client(trust_env=False) as client:
        response = client.post("http://127.0.0.1:8080/infer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "output" in data
