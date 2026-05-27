#!/usr/bin/env python3
"""
bytropix Local Inference Server — OpenAI-compatible HTTP API for gen_text_cpu.

Calls the local CPU inference binary directly. NOT a proxy.

Usage:
  python3 tools/serve_local.py --port 8001
  python3 tools/serve_local.py --port 8001 --model ~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf

Endpoints:
  POST /v1/chat/completions  — Chat completions (OpenAI-compatible)
  POST /v1/completions       — Text completions
  GET  /v1/models            — List available models
  GET  /health               — Health check
"""

import os
import sys
import json
import time
import uuid
import re
import logging
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from typing import Optional

# ============================================================
# Config
# ============================================================

BYTROPIX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(os.path.expanduser("~"), "models", "qwen3.6-35b-a3b-UD-IQ2_M.gguf")
INFER_BIN = os.path.join(BYTROPIX_DIR, "gen_text_cpu")
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 40
DEFAULT_TIMEOUT = 300  # 5 min for model load + inference
MAX_REQUEST_SIZE = 1_000_000

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("serve_local")

# ============================================================
# Inference Runner — calls gen_text_cpu via subprocess
# ============================================================

class InferenceRunner:
    """Runs the local bytropix CPU inference binary."""

    def __init__(self, bin_path: str, model_path: str):
        self.bin_path = os.path.abspath(bin_path)
        self.model_path = os.path.abspath(model_path)
        self.workdir = os.path.dirname(self.bin_path)
        self._lock = threading.Lock()

    def generate(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE,
                 top_k: int = DEFAULT_TOP_K,
                 timeout: int = DEFAULT_TIMEOUT) -> tuple[str, str, float]:
        """Run inference. Returns (generated_text, full_output, elapsed_seconds)."""
        env = os.environ.copy()
        env["MODEL"] = self.model_path
        env["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
        if temperature > 0:
            env["TEMP"] = str(temperature)
        env["TOP_K"] = str(top_k)
        env["MOE"] = "1"

        cmd = [self.bin_path, prompt, str(max_tokens), str(top_k)]

        start = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                env=env, cwd=self.workdir
            )
            elapsed = time.time() - start
            full = result.stdout + result.stderr

            # Extract generated text from gen_text_cpu output
            # Output format: "--- Stats ---\nPrefill: ... tok in ...s (... tok/s)\nDecode:  ...\n"
            # Generated text appears on its own line(s) before "--- Stats ---"
            text = self._extract_generated(full)
            return text, full, elapsed

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return "", f"TIMEOUT after {timeout}s", elapsed
        except FileNotFoundError:
            return "", f"Binary not found: {self.bin_path}", 0
        except Exception as e:
            return "", f"Error: {e}", 0

    def _extract_generated(self, full_output: str) -> str:
        """Extract generated text from gen_text_cpu output."""
        lines = full_output.split('\n')
        gen_parts = []
        in_gen = False
        for line in lines:
            if line.startswith('--- Stats ---'):
                in_gen = False
            if in_gen:
                gen_parts.append(line)
            if line.startswith('Input:') and not in_gen:
                in_gen = True
                continue
        # Skip the Input: line itself
        text = '\n'.join(gen_parts).strip()
        # Remove special tokens
        text = re.sub(r'<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|im_end\|>', '', text)
        # Remove thinking tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()


# ============================================================
# Chat Template
# ============================================================

CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def render_chat(messages: list) -> str:
    """Render chat messages into a single prompt string for the model."""
    system = "You are a helpful assistant."
    user_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system = content
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            user_parts.append(f"[assistant: {content}]")
    user_text = "\n".join(user_parts)
    return CHAT_TEMPLATE.format(system=system, user=user_text)


# ============================================================
# Chat Template for RAW mode (no chat template)
# ============================================================

def render_prompt(messages: list) -> str:
    """Render as RAW text — concatenate all messages."""
    parts = []
    for msg in messages:
        parts.append(msg.get("content", ""))
    return "\n".join(parts)


# ============================================================
# HTTP Request Handler
# ============================================================

class APIHandler(BaseHTTPRequestHandler):
    inference = None  # Set by server
    model_path = None
    server_start = 0

    def log_message(self, format, *args):
        log.info(f"{self.client_address[0]} - {format % args}")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400, code: str = "error"):
        self._send_json({"error": {"message": message, "type": code, "code": status}}, status)

    def _read_body(self) -> Optional[dict]:
        length = int(self.headers.get('Content-Length', 0))
        if length > MAX_REQUEST_SIZE:
            self._send_error(f"Request too large (max {MAX_REQUEST_SIZE} bytes)", 413)
            return None
        if length == 0:
            return {}
        try:
            raw = self.rfile.read(length)
            return json.loads(raw)
        except json.JSONDecodeError:
            self._send_error("Invalid JSON body", 400)
            return None
        except Exception as e:
            self._send_error(f"Read error: {e}", 400)
            return None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/health':
            model_exists = os.path.exists(self.model_path) if self.model_path else False
            bin_exists = os.path.exists(APIHandler.inference.bin_path) if APIHandler.inference else False
            self._send_json({
                "status": "ok",
                "model_loaded": model_exists,
                "binary_ready": bin_exists,
                "uptime": time.time() - APIHandler.server_start,
                "model": self.model_path or "",
                "backend": "local_cpu",
            })
        elif path == '/v1/models':
            model_name = os.path.basename(self.model_path) if self.model_path else "bytropix-local"
            self._send_json({
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": int(APIHandler.server_start),
                        "owned_by": "bytropix",
                    }
                ]
            })
        else:
            self._send_error("Not found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        data = self._read_body()
        if data is None:
            return

        if path == '/v1/chat/completions':
            self._handle_chat_completion(data)
        elif path == '/v1/completions':
            self._handle_completion(data)
        else:
            self._send_error("Not found", 404)

    def _handle_chat_completion(self, data: dict):
        messages = data.get("messages", [])
        if not messages:
            self._send_error("messages is required", 400)
            return

        max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
        temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
        top_k = int(data.get("top_k", DEFAULT_TOP_K))
        stream = data.get("stream", False)

        # Render prompt
        prompt = render_chat(messages)

        # Run inference
        text, full, elapsed = APIHandler.inference.generate(
            prompt, max_tokens=max_tokens, temperature=temperature, top_k=top_k
        )

        if stream:
            self._send_streaming(text)
        else:
            self._send_json({
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": os.path.basename(self.model_path) if self.model_path else "bytropix-local",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            })

    def _handle_completion(self, data: dict):
        prompt = data.get("prompt", "")
        if not prompt:
            self._send_error("prompt is required", 400)
            return

        max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
        temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
        top_k = int(data.get("top_k", DEFAULT_TOP_K))
        stream = data.get("stream", False)

        text, full, elapsed = APIHandler.inference.generate(
            prompt, max_tokens=max_tokens, temperature=temperature, top_k=top_k
        )

        if stream:
            self._send_streaming(text)
        else:
            self._send_json({
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": os.path.basename(self.model_path) if self.model_path else "bytropix-local",
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            })

    def _send_streaming(self, text: str):
        """Send a simple streaming response (non-chunked for simplicity)."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Send as a single chunk
        chunk = json.dumps({
            "choices": [{"delta": {"content": text}, "finish_reason": "stop", "index": 0}]
        })
        self.wfile.write(f"data: {chunk}\n\n".encode('utf-8'))
        self.wfile.write(b"data: [DONE]\n\n")


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="bytropix Local Inference Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to model GGUF file")
    parser.add_argument("--bin", type=str, default=INFER_BIN, help="Path to gen_text_cpu binary")
    args = parser.parse_args()

    model_path = os.path.abspath(os.path.expanduser(args.model))
    bin_path = os.path.abspath(os.path.expanduser(args.bin))

    # Validate paths
    if not os.path.exists(model_path):
        log.error(f"Model not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(bin_path):
        log.error(f"Inference binary not found: {bin_path}")
        log.error(f"Build it first: cd {BYTROPIX_DIR} && make gen_text_cpu")
        sys.exit(1)

    # Set up inference runner
    APIHandler.inference = InferenceRunner(bin_path, model_path)
    APIHandler.model_path = model_path
    APIHandler.server_start = time.time()

    log.info(f"Starting bytropix local inference server on {args.host}:{args.port}")
    log.info(f"Model: {model_path}")
    log.info(f"Binary: {bin_path}")
    log.info(f"CPU threads: {os.environ.get('OMP_NUM_THREADS', '4')}")

    server = HTTPServer((args.host, args.port), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
