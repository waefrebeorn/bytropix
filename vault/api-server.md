# API Server — Qwen3.6 Inference Endpoint

## Overview
OpenAI-compatible HTTP API for the `infer_text_gpu` binary. Supports text completions, chat completions with Qwen3.6 template, and SSE streaming.

## Files
- `tools/serve.py` — HTTP server (Python)
- `tests/test_api.sh` — Sandbox integration test suite

## Usage

```bash
# Production mode (uses real GPU inference)
python3 tools/serve.py --port 8080 --model /path/to/model.gguf

# Sandbox mode (fake responses, fake API keys — for security testing)
python3 tools/serve.py --sandbox --port 8080
```

## Endpoints

### GET /health
Health check.
```json
{"status": "ok", "mode": "sandbox", "uptime": 42, "model": "qwen3.6-35b-a3b"}
```

### GET /v1/models
List available models.

### POST /v1/completions
Text completion (OpenAI-compatible).

**Request:**
```json
{
  "prompt": "The capital of France is",
  "max_tokens": 128,
  "temperature": 1.0,
  "top_k": 20,
  "top_p": 0.95,
  "stream": false
}
```

**Response:**
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1778846889,
  "model": "qwen3.6-35b-a3b",
  "choices": [{"text": "...", "index": 0, "finish_reason": "length"}],
  "usage": {"prompt_tokens": 5, "completion_tokens": 128, "total_tokens": 133}
}
```

### POST /v1/chat/completions
Chat completion with Qwen3.6 template.

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is Python?"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

Template format:
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
```

### SSE Streaming
Set `"stream": true` for token-by-token streaming via Server-Sent Events.

```
data: {"choices": [{"delta": {"text": "The "}, "index": 0, "finish_reason": null}]}
data: {"choices": [{"delta": {"text": "quick "}, "index": 0, "finish_reason": null}]}
...
data: {"choices": [{"delta": {"text": ""}, "index": 0, "finish_reason": "length"}]}
data: [DONE]
```

## Security

### Sandbox Mode (`--sandbox`)
- Fake API keys: `sk-sandbox-test-key-1`, `sk-sandbox-test-key-2`, `sk-sandbox-ratelimit`
- Per-key rate limiting: key-1=100req/min, key-2=10req/min, ratelimit=2req/min
- Fake responses — no GPU used, no real inference
- For integration testing, security fuzzing, and CI validation

### Rate Limiting
- Default: 60 req/min per IP (configurable in source at `RATE_LIMIT_REQUESTS`)
- Sandbox mode: per-key limits
- Returns 429 with `rate_limit_error` type on exceed

### Input Validation
- Max request size: 1MB
- JSON validation with descriptive errors
- max_tokens clamped to 2048
- CORS headers on all responses
- SQL injection attempts in prompts handled safely

## Test Suite
```bash
# Full test suite (starts server, runs 14 tests, cleans up)
bash tests/test_api.sh

# Quick smoke test
bash tests/test_api.sh --quick

# Security fuzzing
bash tests/test_api.sh --fuzz
```

Requires: python3, curl, no other dependencies.
