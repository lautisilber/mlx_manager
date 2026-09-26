#!/usr/bin/env python3
"""Minimal OpenAI-compatible server for MLX packs that ship their own runtime
loader (see pack_common.py / pack_chat.py's docstring).

Implements /v1/chat/completions (message-based chat) and /v1/completions
(legacy raw-prompt completion, e.g. for editor inline-completion features
like Zed's), plus /v1/models. Both completion endpoints support a single-
JSON-body response and, when the client sets "stream": true, a Server-Sent-
Events stream of chunk objects (HTTP/1.1 chunked transfer encoding, since
the response length isn't known upfront). Both also accept per-request
sampling overrides (max_tokens, temperature, top_p, top_k, min_p,
repetition_penalty, stop); anything not given falls back to the pack's own
generation_config.json defaults. stop (a string or list of strings) ends
generation as soon as any of them appears in the output, excluding the
match itself from the returned/streamed text. Requests are handled one at
a time (plain HTTPServer, not threaded) since the underlying model isn't
safe to call concurrently.
"""

import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from pack_common import make_generator, render

PACK = Path(sys.argv[1]).resolve()
PORT = int(sys.argv[2])
MODEL_NAME = sys.argv[3]
HOST = sys.argv[4]

OVERRIDE_KEYS = ("max_tokens", "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "stop")


def _overrides(body):
    """Per-request sampling overrides recognized from the request body; anything not
    present (or explicitly null) falls back to the pack's own generation_config.json."""
    overrides = {k: body[k] for k in OVERRIDE_KEYS if body.get(k) is not None}
    if isinstance(overrides.get("stop"), str):
        overrides["stop"] = [overrides["stop"]]
    return overrides


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


def _chat_chunk(completion_id, created, delta, finish_reason=None):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _text_completion(text, completion_tokens, finish_reason):
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens,
        },
    }


def _completion_chunk(completion_id, created, text, finish_reason=None):
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}],
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

        def _start_sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

        def _write_chunk(self, data: bytes):
            self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
            self.wfile.flush()

        def _write_sse_event(self, payload):
            self._write_chunk(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))

        def _end_sse(self):
            self._write_chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def _read_json_body(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                return json.loads(self.rfile.read(length) or b"{}"), None
            except json.JSONDecodeError:
                return None, "invalid JSON"

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
            try:
                if self.path in ("/v1/chat/completions", "/chat/completions"):
                    self._handle_chat_completions()
                elif self.path in ("/v1/completions", "/completions"):
                    self._handle_completions()
                else:
                    self._send_json(404, {"error": {"message": "not found"}})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # normal when a client cancels an in-flight request (e.g. an editor
                # discarding a stale autocomplete suggestion mid-stream); generation
                # already stopped at this point since the failed write raised here
                print("[INFO] Client disconnected before the response finished.")

        def _handle_chat_completions(self):
            body, err = self._read_json_body()
            if err:
                self._send_json(400, {"error": {"message": err}})
                return
            messages = body.get("messages")
            if not messages:
                self._send_json(400, {"error": {"message": "'messages' is required"}})
                return
            prompt_text = render(PACK, messages)
            overrides = _overrides(body)

            if body.get("stream"):
                print(f"[INFO] Generating streaming chat response for {len(messages)} message(s)...")
                completion_id = f"chatcmpl-{uuid.uuid4().hex}"
                created = int(time.time())
                self._start_sse()
                self._write_sse_event(_chat_chunk(completion_id, created, {"role": "assistant"}))

                def on_token(token_text):
                    self._write_sse_event(_chat_chunk(completion_id, created, {"content": token_text}))

                _, _, finish_reason = generate(prompt_text, on_token=on_token, **overrides)
                self._write_sse_event(_chat_chunk(completion_id, created, {}, finish_reason=finish_reason))
                self._end_sse()
            else:
                print(f"[INFO] Generating chat response for {len(messages)} message(s)...")
                text, completion_tokens, finish_reason = generate(prompt_text, **overrides)
                self._send_json(200, _chat_completion(text, completion_tokens, finish_reason))

        def _handle_completions(self):
            body, err = self._read_json_body()
            if err:
                self._send_json(400, {"error": {"message": err}})
                return
            prompt = body.get("prompt")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else None
            if not prompt:
                self._send_json(400, {"error": {"message": "'prompt' is required"}})
                return
            overrides = _overrides(body)

            if body.get("stream"):
                print("[INFO] Generating streaming completion...")
                completion_id = f"cmpl-{uuid.uuid4().hex}"
                created = int(time.time())
                self._start_sse()

                def on_token(token_text):
                    self._write_sse_event(_completion_chunk(completion_id, created, token_text))

                _, _, finish_reason = generate(prompt, on_token=on_token, **overrides)
                self._write_sse_event(_completion_chunk(completion_id, created, "", finish_reason=finish_reason))
                self._end_sse()
            else:
                print("[INFO] Generating completion...")
                text, completion_tokens, finish_reason = generate(prompt, **overrides)
                self._send_json(200, _text_completion(text, completion_tokens, finish_reason))

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
