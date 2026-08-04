#!/usr/bin/env python3
"""wubu_live_learn.py -- the LIVE SPEED LEARNING loop driver.

The user's directive: "live speed learning for AGI... the learning loop
is more proper LLM corpus and we feedback loop with nvidia cloud keys."

The loop (one iteration = one live learning event):
  1. take a prompt from the corpus (SFT/agentic pack) or stdin
  2. WuBu (the C live_learn binary, WuBu-arch load) GENERATES a draft
  3. the NVIDIA NIM oracle (tools/nvidia_nim.py score_draft) SCORES the
     draft 0-100 + gives a critique (the R1 pattern, live oracle)
  4. ACCUMULATE {prompt, draft, critique, score} into the live-SFT
     buffer (JSONL) — the next SFT training round consumes it
  5. the kernel supervisor (wubuos wubu_agi_kernel) logs the event as a
     trace span (the loop is the AGI's heartbeat)

Usage:
  python3 tools/wubu_live_learn.py \
      --model models/wubu/model.safetensors \
      --tokenizer models/wubu/tokenizer.json \
      --prompt "What is the capital of France?" \
      [--steps 24] [--temp 0.7] [--out models/corpus/live/wubu-live-sft.jsonl]
  python3 tools/wubu_live_learn.py --stream corpus.txt   # batch mode
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_BIN = os.path.join(ROOT, "wubu_live_learn")
NIM = os.path.join(ROOT, "tools", "nvidia_nim.py")
DEFAULT_OUT = "/home/wubu/models/corpus/live/wubu-live-sft.jsonl"


def tokenc_prompt(tokenizer, text):
    """Tokenize a prompt via wubu_tokenc (the .tok uint16 stream)."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write(text + "\n")
        tmp = tf.name
    out = tmp + ".tok"
    r = subprocess.run([os.path.join(ROOT, "wubu_tokenc"), tokenizer, tmp, out],
                       capture_output=True, text=True, timeout=120)
    os.unlink(tmp)
    if r.returncode != 0:
        return None, r.stderr
    data = open(out, "rb").read()
    os.unlink(out)
    toks = struct.unpack("<%dH" % (len(data) // 2), data)
    return list(toks), None


def decode_tokens(tokenizer, toks):
    """Decode a token stream to text via the HF tokenizer (python side)."""
    try:
        from transformers import AutoTokenizer
        tz = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
        return tz.decode(toks, skip_special_tokens=True)
    except Exception:
        return "[tokens:%d]" % len(toks)


def score_via_nvidia(prompt, draft):
    """Call the NVIDIA NIM oracle (score_draft): (score, critique, err)."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    try:
        import nvidia_nim
        s, c, e = nvidia_nim.score_draft(draft, prompt)
        return s, c, e
    except Exception as e:
        return None, None, str(e)


def live_step(model, tokenizer, prompt, steps, temp, out_path):
    """One full live learning event; returns the record or None."""
    toks, terr = tokenc_prompt(tokenizer, prompt)
    if toks is None:
        print(f"  tokenize error: {terr}")
        return None
    with open("/tmp/live_prompt.tok", "wb") as f:
        f.write(struct.pack("<%dH" % len(toks), *toks))
    r = subprocess.run([LIVE_BIN, model, tokenizer, "/tmp/live_prompt.tok",
                        "--steps", str(steps), "--temp", str(temp)],
                       capture_output=True, timeout=600)
    if r.returncode != 0:
        print(f"  live_learn rc={r.returncode}: {r.stderr.decode()[:200]}")
        return None
    gen = struct.unpack("<%dH" % (len(r.stdout) // 2), r.stdout)
    # The C binary decodes the draft with the model's OWN HF tokenizer and
    # prints it to stderr as "live_learn DRAFT: <text>". Prefer that (it
    # always works — no transformers dependency). Fall back to the python
    # decode only when the C line is missing.
    draft = None
    err_txt = r.stderr.decode("utf-8", errors="replace")
    for line in err_txt.splitlines():
        if "live_learn DRAFT:" in line:
            draft = line.split("live_learn DRAFT:", 1)[1].strip()
            break
    if draft is None:
        draft = decode_tokens(tokenizer, list(gen))
    if not draft or draft.startswith("[tokens"):
        draft = "[draft:%d tokens]" % len(gen)
    score, critique, err = score_via_nvidia(prompt, draft)
    rec = {
        "prompt": prompt, "draft": draft,
        "score": score, "critique": critique, "oracle_err": err,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  score={score} | draft={draft[:60]!r}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/wubu/model.safetensors")
    ap.add_argument("--tokenizer", default="models/wubu/tokenizer.json")
    ap.add_argument("--prompt")
    ap.add_argument("--stream")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=10)
    a = ap.parse_args()

    if not os.path.exists(LIVE_BIN):
        print("build first: gcc ... tools/wubu_live_learn.c (see Makefile wubu_train link line)")
        return 1

    if a.prompt:
        live_step(a.model, a.tokenizer, a.prompt, a.steps, a.temp, a.out)
        return 0

    if a.stream:
        prompts = [l.strip() for l in open(a.stream, encoding="utf-8")
                   if l.strip() and len(l.strip()) > 10][:a.limit]
        for i, p in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] {p[:60]!r}")
            live_step(a.model, a.tokenizer, p, a.steps, a.temp, a.out)
            time.sleep(1)
        print(f"live loop done: {len(prompts)} events -> {a.out}")
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
