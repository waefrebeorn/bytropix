#!/usr/bin/env python3
"""
Qwen3.6 API Server — OpenAI-compatible HTTP API for infer_text_gpu
Sandbox mode: --sandbox for fake keys, isolated testing, security fuzzing.

Usage:
  # Normal mode
  python3 tools/serve.py --port 8080
  
  # Sandbox mode (fake keys, rate limits enforced, fuzzing endpoints)
  python3 tools/serve.py --sandbox --port 8080
  
  # With custom model
  python3 tools/serve.py --model /path/to/model.gguf

Endpoints:
  POST /v1/completions       — Text completion
  POST /v1/chat/completions  — Chat completion (OpenAI-compatible)
  GET  /v1/models            — List available models
  GET  /health               — Health check
"""

import os
import sys
import json
import time
import uuid
import re
import html
import signal
import logging
import subprocess
import threading
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from typing import Optional

# ============================================================
# Config
# ============================================================

MODEL_PATH = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
INFER_BIN = "./infer_text_gpu"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 20
DEFAULT_TOP_P = 0.95
DEFAULT_TIMEOUT = 120
MAX_REQUEST_SIZE = 1_000_000  # 1MB max request body
RATE_LIMIT_REQUESTS = 60      # requests per minute
RATE_LIMIT_WINDOW = 60        # seconds

# Sandbox defaults
SANDBOX_API_KEYS = {
    "sk-sandbox-test-key-1": {"user": "test_user_1", "rate": 100},
    "sk-sandbox-test-key-2": {"user": "test_user_2", "rate": 10},
    "sk-sandbox-ratelimit": {"user": "rate_limited_user", "rate": 2},
}

# Chat template for Qwen3.6
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("serve")

# ============================================================
# Rate Limiter
# ============================================================

class RateLimiter:
    def __init__(self):
        self._buckets = {}  # key -> [timestamps]
        self._lock = threading.Lock()
    
    def check(self, key: str, max_req: int, window: int) -> tuple[bool, int]:
        """Returns (allowed, remaining)"""
        now = time.time()
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = []
            cut = now - window
            self._buckets[key] = [t for t in self._buckets[key] if t > cut]
            remaining = max_req - len(self._buckets[key])
            if remaining <= 0:
                return False, 0
            self._buckets[key].append(now)
            return True, remaining - 1

# ============================================================
# Inference Runner
# ============================================================

class InferenceRunner:
    def __init__(self, bin_path: str, model_path: str):
        self.bin_path = os.path.abspath(bin_path)
        self.model_path = model_path
        self._lock = threading.Lock()
    
    def generate(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE,
                 top_k: int = DEFAULT_TOP_K, top_p: float = DEFAULT_TOP_P,
                 timeout: int = DEFAULT_TIMEOUT) -> tuple[str, str, float]:
        """Run inference. Returns (generated_text, full_output, elapsed_seconds).
        
        In sandbox mode, returns fake output for testing without real inference.
        """
        env = os.environ.copy()
        env["TEMP"] = str(temperature)
        env["TOP_K"] = str(top_k)
        env["TOP_P"] = str(top_p)
        env["MOE"] = "1"  # MoE required for this model
        
        cmd = [self.bin_path, self.model_path, prompt, str(max_tokens)]
        
        start = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                env=env, cwd=os.path.dirname(self.bin_path) or "."
            )
            elapsed = time.time() - start
            full = result.stdout + result.stderr
            
            # Extract generated text: it's the part after the decoded prompt
            # Format: "...Prompt: \"...\" | ...\n--- Chunked prefill ---\nPROMPT_TEXTgenerated_text\nPrefill: ..."
            text = self._extract_generated(full, prompt)
            
            return text, full, elapsed
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return "", f"TIMEOUT after {timeout}s", elapsed
        except FileNotFoundError:
            return "", f"Binary not found: {self.bin_path}", 0
        except Exception as e:
            return "", f"Error: {e}", 0
    
    def _extract_generated(self, full_output: str, prompt: str) -> str:
        """Extract just the generated text from the full output."""
        # Find the generated portion after the prompt in output
        # The output format is: "...PROMPT_TEXTgenerated_text\nPrefill:..."
        # We need to find what comes after the expected prompt
        lines = full_output.split('\n')
        in_generation = False
        gen_parts = []
        
        for line in lines:
            if '--- Decode' in line or '=== Summary' in line:
                in_generation = False
            if in_generation:
                gen_parts.append(line)
            if '--- Chunked prefill' in line:
                in_generation = True
                continue
            # First line after prefill header contains prompt + first token
            if in_generation and line.strip():
                # Strip the prompt from the beginning
                cleaned = line.strip()
                # Remove prompt prefix if present
                if prompt in cleaned:
                    cleaned = cleaned.split(prompt, 1)[-1]
                if cleaned:
                    gen_parts = [cleaned]
                    break
        
        result = ''.join(gen_parts).strip()
        # Remove think/endoftext tags
        result = re.sub(r'<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>', '', result)
        return result.strip()

# ============================================================
# Sandbox Inference Runner (fake mode)
# ============================================================

class SandboxInferenceRunner:
    """Fake inference for sandbox testing — no real GPU needed."""
    
    FAKE_RESPONSES = {
        "paris": "Paris is the capital of France, known for the Eiffel Tower and its rich history.",
        "capital": "The capital of France is Paris. It is one of the most visited cities in the world.",
        "hello": "Hello! How can I help you today?",
        "python": "Here's a Python function to sort a list:\n\n```python\ndef sort_list(items):\n    return sorted(items)\n```",
        "default": "The quick brown fox jumps over the lazy dog. This is a sandbox test response for API verification purposes."
    }
    
    def generate(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE,
                 top_k: int = DEFAULT_TOP_K, top_p: int = DEFAULT_TOP_P,
                 timeout: int = DEFAULT_TIMEOUT) -> tuple[str, str, float]:
        import random
        time.sleep(0.1)  # Simulate latency
        
        prompt_lower = prompt.lower()
        response = self.FAKE_RESPONSES.get("default")
        for key, val in self.FAKE_RESPONSES.items():
            if key in prompt_lower:
                response = val
                break
        
        # Truncate to max_tokens (approx)
        words = response.split()
        response = ' '.join(words[:max_tokens])
        
        return response, f"[SANDBOX] {response}", 0.1

# ============================================================
# Request Handler
# ============================================================

class APIHandler(BaseHTTPRequestHandler):
    sandbox = False
    rate_limiter = None
    inference = None
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
            body = self.rfile.read(length).decode('utf-8')
            return json.loads(body)
        except json.JSONDecodeError:
            self._send_error("Invalid JSON", 400)
            return None
        except Exception as e:
            self._send_error(f"Request error: {str(e)}", 400)
            return None
    
    def _check_auth(self) -> bool:
        """Check API key. In sandbox mode, use fake keys."""
        auth = self.headers.get('Authorization', '')
        
        if self.sandbox:
            # Sandbox mode: check against fake keys
            key = auth.replace('Bearer ', '') if auth.startswith('Bearer ') else auth
            if key not in SANDBOX_API_KEYS:
                self._send_error(
                    "Invalid API key. Sandbox keys: " + ", ".join(SANDBOX_API_KEYS.keys()),
                    401, "authentication_error"
                )
                return False
            # Set rate limit for this key
            key_config = SANDBOX_API_KEYS[key]
            allowed, remaining = self.rate_limiter.check(
                f"sandbox:{key}", key_config["rate"], RATE_LIMIT_WINDOW
            )
            if not allowed:
                self._send_error(
                    f"Rate limit exceeded ({key_config['rate']} req/min). "
                    f"Try key 'sk-sandbox-ratelimit' for a lower limit test.",
                    429, "rate_limit_error"
                )
                return False
            return True
        else:
            # Production mode: check if API key present (don't validate in dev)
            if not auth and 'API_KEY_REQUIRED' in os.environ:
                self._send_error("Missing API key. Set Authorization: Bearer <key>", 401)
                return False
            # Rate limit by IP in production
            allowed, remaining = self.rate_limiter.check(
                f"ip:{self.client_address[0]}", RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW
            )
            if not allowed:
                self._send_error(
                    f"Rate limit exceeded ({RATE_LIMIT_REQUESTS} req/min)", 
                    429, "rate_limit_error"
                )
                return False
            return True
    
    def _format_chat(self, messages: list) -> str:
        """Format chat messages into prompt using Qwen3.6 template."""
        system = ""
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
        return CHAT_TEMPLATE.format(system=system or "You are Qwen, a helpful AI assistant.", user=user_text)
    
    def _handle_completion(self, body: dict):
        prompt = body.get("prompt", "")
        if not prompt:
            self._send_error("Missing 'prompt' field", 400)
            return
        
        max_tokens = min(body.get("max_tokens", DEFAULT_MAX_TOKENS), 2048)
        temperature = body.get("temperature", DEFAULT_TEMPERATURE)
        top_k = body.get("top_k", DEFAULT_TOP_K)
        top_p = body.get("top_p", DEFAULT_TOP_P)
        stream = body.get("stream", False)
        
        if stream:
            self._handle_stream(prompt, max_tokens, temperature, top_k, top_p)
            return
        
        text, raw, elapsed = self.inference.generate(
            prompt, max_tokens, temperature, top_k, top_p
        )
        
        # Count tokens (approximate)
        prompt_tokens = len(prompt.split())
        completion_tokens = len(text.split())
        
        resp = {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "qwen3.6-35b-a3b",
            "choices": [{
                "text": text,
                "index": 0,
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        self._send_json(resp)
    
    def _handle_chat(self, body: dict):
        messages = body.get("messages", [])
        if not messages:
            self._send_error("Missing 'messages' field", 400)
            return
        
        prompt = self._format_chat(messages)
        max_tokens = min(body.get("max_tokens", DEFAULT_MAX_TOKENS), 2048)
        temperature = body.get("temperature", DEFAULT_TEMPERATURE)
        top_k = body.get("top_k", DEFAULT_TOP_K)
        top_p = body.get("top_p", DEFAULT_TOP_P)
        stream = body.get("stream", False)
        
        if stream:
            self._handle_stream(prompt, max_tokens, temperature, top_k, top_p, is_chat=True)
            return
        
        text, raw, elapsed = self.inference.generate(
            prompt, max_tokens, temperature, top_k, top_p
        )
        
        prompt_tokens = len(prompt.split())
        completion_tokens = len(text.split())
        
        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "qwen3.6-35b-a3b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        self._send_json(resp)
    
    def _handle_stream(self, prompt: str, max_tokens: int,
                       temperature: float, top_k: int, top_p: float,
                       is_chat: bool = False):
        """SSE streaming response."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # In sandbox mode, simulate streaming
        if self.sandbox:
            text, _, _ = self.inference.generate(prompt, max_tokens, temperature, top_k, top_p)
            words = text.split()
            for i, word in enumerate(words):
                chunk = word + " "
                event = {
                    "choices": [{
                        "delta": {"content": chunk} if is_chat else {"text": chunk},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.05)
            
            final = {
                "choices": [{
                    "delta": {} if is_chat else {"text": ""},
                    "index": 0,
                    "finish_reason": "length"
                }]
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
            return
        
        # Real mode: run inference and stream tokens
        # For now, run full inference and simulate streaming
        text, raw, elapsed = self.inference.generate(
            prompt, max_tokens, temperature, top_k, top_p
        )
        
        words = text.split()
        for i, word in enumerate(words):
            chunk = word + " "
            event = {
                "choices": [{
                    "delta": {"content": chunk} if is_chat else {"text": chunk},
                    "index": 0,
                    "finish_reason": None
                }]
            }
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
        
        final = {
            "choices": [{
                "delta": {} if is_chat else {"text": ""},
                "index": 0,
                "finish_reason": "length"
            }]
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
    
    # ========================================================
    # HTTP Method Handlers
    # ========================================================
    
    def do_OPTIONS(self):
        """CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == '/health':
            self._send_json({
                "status": "ok",
                "mode": "sandbox" if self.sandbox else "production",
                "uptime": int(time.time() - self.server_start),
                "model": "qwen3.6-35b-a3b" if not self.sandbox else "sandbox-fake-model"
            })
        elif path == '/v1/models':
            self._send_json({
                "object": "list",
                "data": [{
                    "id": "qwen3.6-35b-a3b",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "unsloth"
                }]
            })
        else:
            self._send_error("Not found", 404)
    
    def do_POST(self):
        path = urlparse(self.path).path
        
        if not self._check_auth():
            return
        
        body = self._read_body()
        if body is None:
            return
        
        if path == '/v1/completions':
            self._handle_completion(body)
        elif path == '/v1/chat/completions':
            self._handle_chat(body)
        else:
            self._send_error(f"Unknown endpoint: {path}", 404)

# ============================================================
# Main
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3.6 API Server")
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--sandbox', action='store_true', help='Sandbox mode (fake keys, fake responses)')
    parser.add_argument('--model', default=MODEL_PATH, help='Path to GGUF model')
    parser.add_argument('--bin', default=INFER_BIN, help='Path to inference binary')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='Inference timeout (seconds)')
    args = parser.parse_args()
    
    # Configure
    APIHandler.sandbox = args.sandbox
    APIHandler.rate_limiter = RateLimiter()
    APIHandler.server_start = time.time()
    
    # Setup inference
    if args.sandbox:
        log.info(" SANDBOX MODE — fake responses, fake API keys")
        log.info(f"  Fake keys: {', '.join(SANDBOX_API_KEYS.keys())}")
        APIHandler.inference = SandboxInferenceRunner()
    else:
        if not os.path.exists(args.bin):
            log.error(f"Inference binary not found: {args.bin}")
            log.error(f"Build it first: make infer_text_gpu")
            sys.exit(1)
        if not os.path.exists(args.model):
            log.error(f"Model not found: {args.model}")
            sys.exit(1)
        APIHandler.inference = InferenceRunner(args.bin, args.model)
        log.info(f" Model: {args.model}")
        log.info(f" Binary: {args.bin}")
    
    # Start server
    server = HTTPServer((args.host, args.port), APIHandler)
    server.timeout = args.timeout
    
    log.info(f" Server: http://{args.host}:{args.port}")
    log.info(f" Endpoints:")
    log.info(f"   POST /v1/completions      — Text completion")
    log.info(f"   POST /v1/chat/completions  — Chat completion")
    log.info(f"   GET  /v1/models            — List models")
    log.info(f"   GET  /health               — Health check")
    log.info(f" Rate limit: {RATE_LIMIT_REQUESTS} req/{RATE_LIMIT_WINDOW}s")
    log.info(" Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()
        log.info("Done")

if __name__ == '__main__':
    main()
