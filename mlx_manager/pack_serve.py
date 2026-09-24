#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat completions server for MLX packs that ship
their own runtime loader (see pack_common.py / pack_chat.py's docstring).

Supports both a single-JSON-body response and, when the client sets
"stream": true, a Server-Sent-Events stream of chat.completion.chunk objects
(HTTP/1.1 chunked transfer encoding, since the response length isn't known
upfront). Requests are handled one at a time (plain HTTPServer, not
threaded) since the underlying model isn't safe to call concurrently.
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


def _chat_completion(text, completion_tokens, finish_reason):
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens,
        },
    }


def _chunk(completion_id, created, delta, finish_reason=None):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def build_handler(generate):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_chunk(self, data: bytes):
            self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
            self.wfile.flush()

        def _write_sse_event(self, payload):
            self._write_chunk(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))

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

            if body.get("stream"):
                self._handle_streaming(messages)
            else:
                self._handle_non_streaming(messages)

        def _handle_non_streaming(self, messages):
            print(f"[INFO] Generating response for {len(messages)} message(s)...")
            text, completion_tokens, finish_reason = generate(messages)
            self._send_json(200, _chat_completion(text, completion_tokens, finish_reason))

        def _handle_streaming(self, messages):
            print(f"[INFO] Generating streaming response for {len(messages)} message(s)...")
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            self._write_sse_event(_chunk(completion_id, created, {"role": "assistant"}))

            def on_token(token_text):
                self._write_sse_event(_chunk(completion_id, created, {"content": token_text}))

            _, _, finish_reason = generate(messages, on_token=on_token)

            self._write_sse_event(_chunk(completion_id, created, {}, finish_reason=finish_reason))
            self._write_chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def log_message(self, fmt, *args):
            print(f"[INFO] {self.address_string()} - {fmt % args}")

    return Handler


def main():
    print(f"[INFO] Loading {MODEL_NAME} from {PACK}...")
    generate = make_generator(PACK)
    print("[INFO] Model loaded.")

    server = HTTPServer((HOST, PORT), build_handler(generate))
    print(f"[INFO] Serving {MODEL_NAME} on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
