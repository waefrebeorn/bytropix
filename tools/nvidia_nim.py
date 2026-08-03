#!/usr/bin/env python3
"""
nvidia_nim.py -- the NVIDIA NIM free-tier client for WuBu.

The user's directive: "live stream embedding and free API". NVIDIA's
build.nvidia.com NIM gives a FREE OpenAI-compatible endpoint:
    base_url: https://integrate.api.nvidia.com/v1
    key:      the NVIDIA API key (from build.nvidia.com, free tier,
              ~1000 credits, no credit card)

What this client does for WuBu:
  1. EMBEDDINGS -- live-stream embedding: text -> vectors via
     nvidia/NV-Embed-QA (or nvidia/embed-qa-4). The vectors feed the
     hive (wubu_hive) as observations, giving WuBu a live semantic
     sense of the streaming corpus.
  2. INFERENCE -- the bigger-brother coherency checks: ask a frontier
     NIM model to score WuBu's drafts (draft -> score -> the trainer's
     reward, the R1/Prover pattern but with a live oracle).

The key is read from $NVIDIA_API_KEY or the secrets file.
"""
import os
import sys
import json
import urllib.request


def get_key():
    """Find the NVIDIA API key: env, or the mind-palace secrets file."""
    k = os.environ.get("NVIDIA_API_KEY", "").strip()
    if k:
        return k
    secrets = os.path.expanduser(
        "~/.hermes/profiles/mind-palace/secrets/hf.env")
    if os.path.exists(secrets):
        for line in open(secrets):
            if line.startswith("NVIDIA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


BASE = "https://integrate.api.nvidia.com/v1"

# the free-tier embedding models (verified working on build.nvidia.com 2026-08-03)
EMBED_MODELS = ["nvidia/nv-embed-v1", "nvidia/nemotron-3-embed-1b",
                "nvidia/embed-qa-4", "nvidia/llama-nemotron-embed-1b-v2"]
# the free-tier chat models (verified LIVE on build.nvidia.com 2026-08-03:
# minimax-m3, glm-5.2, deepseek-v4-pro all answered; gpt-oss-20b listed)
CHAT_MODELS = ["minimaxai/minimax-m3", "z-ai/glm-5.2", "deepseek-ai/deepseek-v4-pro",
               "openai/gpt-oss-20b", "nvidia/nemotron-3-nano-30b-a3b"]


def call_chat(model, messages, max_tokens=512, temperature=0.7):
    """A chat completion via the NVIDIA NIM free endpoint."""
    key = get_key()
    if not key:
        return None, "no NVIDIA_API_KEY (build.nvidia.com free tier)"
    body = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        return data["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, str(e)[:200]


def call_embed(model, texts):
    """Text -> vectors via the NVIDIA NIM embeddings endpoint."""
    key = get_key()
    if not key:
        return None, "no NVIDIA_API_KEY (build.nvidia.com free tier)"
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{BASE}/embeddings", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        vecs = [d["embedding"] for d in data["data"]]
        return vecs, None
    except Exception as e:
        return None, str(e)[:200]


def score_draft(draft, prompt, model=None):
    """The R1-style oracle: score WuBu's draft against the prompt.
    Returns (score 0-100, critique, error)."""
    model = model or CHAT_MODELS[0]
    sys_msg = ("You are the verification oracle for an AGI seed named "
               "WuBu. Score the draft 0-100 for correctness and "
               "coherence against the prompt. Reply: SCORE:<n> then a "
               "one-line critique.")
    out, err = call_chat(model, [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"PROMPT: {prompt}\nDRAFT: {draft}"},
    ], max_tokens=256, temperature=0)
    if err:
        return None, None, err
    score = None
    for line in out.splitlines():
        if "SCORE:" in line.upper():
            try:
                score = int("".join(ch for ch in line.split(":")[1] if ch.isdigit()))
            except Exception:
                pass
    return score, out, None


if __name__ == "__main__":
    key = get_key()
    print("NVIDIA NIM client for WuBu")
    print(f"  key: {'<set>' if key else 'MISSING -- get one free at build.nvidia.com'}")
    if len(sys.argv) > 1 and sys.argv[1] == "embed":
        vecs, err = call_embed(EMBED_MODELS[0], ["The hive is the AGI way."])
        if err:
            print(f"  embed error: {err}")
        else:
            print(f"  embed OK: {len(vecs[0])} dims, first 3: {vecs[0][:3]}")
    elif len(sys.argv) > 1 and sys.argv[1] == "chat":
        out, err = call_chat(CHAT_MODELS[0],
                             [{"role": "user", "content": "Say hi from WuBu."}])
        if err:
            print(f"  chat error: {err}")
        else:
            print(f"  chat OK: {out[:120]}")
