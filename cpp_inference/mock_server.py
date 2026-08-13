import json
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "127.0.0.1"
PORT = 8081

class MockInferenceHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/infer":
            self.send_error(404, "Not Found")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            self.send_error(400, f"Invalid JSON: {exc}")
            return

        input_text = payload.get("input", "")
        model_name = payload.get("model", "unknown")
        max_tokens = payload.get("options", {}).get("max_tokens", 0)

        response = {
            "request_id": "mock-" + model_name,
            "model": model_name,
            "output": f"mocked GPU output for: {input_text}",
            "usage": {"latency_ms": 10, "tokens": max_tokens},
            "status": "ok",
        }

        body = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def run_server() -> None:
    server = HTTPServer((HOST, PORT), MockInferenceHandler)
    print(f"Mock inference core running at http://{HOST}:{PORT}/infer")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down mock inference core...")
        server.server_close()


if __name__ == "__main__":
    run_server()
