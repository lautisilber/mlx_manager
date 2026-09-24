#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat completions server for MLX packs that ship
their own runtime loader (see pack_common.py / pack_chat.py's docstring).

No streaming support: every response is returned as a single JSON body once
generation finishes, regardless of the client's "stream" field. Requests are
handled one at a time (plain HTTPServer, not threaded) since the underlying
model isn't safe to call concurrently.
"""

import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from pack_common import make_generator

PACK = Path(sys.argv[1]).resolve()
PORT = int(sys.argv[2])
MODEL_NAME = sys.argv[3]
HOST = sys.argv[4]


def build_handler(generate):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "mlx-manager"}
                        ],
                    },
                )
            else:
                self._send_json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                self._send_json(404, {"error": {"message": "not found"}})
                return

            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": {"message": "invalid JSON"}})
                return

            messages = body.get("messages")
            if not messages:
                self._send_json(400, {"error": {"message": "'messages' is required"}})
                return

            print(f"[INFO] Generating response for {len(messages)} message(s)...")
            text, completion_tokens = generate(messages)

            self._send_json(
                200,
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": completion_tokens,
                        "total_tokens": completion_tokens,
                    },
                },
            )

        def log_message(self, fmt, *args):
            print(f"[INFO] {self.address_string()} - {fmt % args}")

    return Handler


def main():
    print(f"[INFO] Loading {MODEL_NAME} from {PACK}...")
    generate = make_generator(PACK)
    print("[INFO] Model loaded.")

    server = HTTPServer((HOST, PORT), build_handler(generate))
    print(f"[INFO] Serving {MODEL_NAME} on http://{HOST}:{PORT} (no streaming support)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
