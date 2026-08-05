#!/usr/bin/env python3
"""Glue: encode prompt (LFM2.5 tokenizer) -> lfm2_gen.exe (C forward) -> decode.
Proper hybrid: Python owns the battle-tested 128K Llama-BPE tokenizer; the C
engine owns the verified LFM2.5 forward. Neither fakes the other's job."""
import sys, subprocess, json, os

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "D:/models/LFM2.5-2.6B"
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "The future of AI is"
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 32
EXE = "C:/Users/eman5/wubuwizard/lfm2_gen.exe"

from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
enc = tok.encode(PROMPT)
ids = enc.ids
print(f"[glue] prompt={PROMPT!r} -> {len(ids)} seed tokens: {ids[:12]}{'...' if len(ids)>12 else ''}", flush=True)

cmd = [EXE, MODEL_DIR, str(MAX_TOKENS), "0.8", "0.9"] + [str(i) for i in ids]
out = subprocess.run(cmd, capture_output=True, text=True)
if out.returncode != 0:
    print("[glue] lfm2_gen failed:", out.stderr[:2000]); sys.exit(1)

# parse "T<pos>:<id>" tokens from stdout
gen_ids = []
for tokstr in out.stdout.split():
    if tokstr.startswith("T") and ":" in tokstr:
        try: gen_ids.append(int(tokstr.split(":")[1]))
        except: pass
print(f"[glue] generated {len(gen_ids)} token ids", flush=True)

# decode the CONTINUATION (exclude the seed tokens we fed in)
cont_ids = gen_ids[len(ids):] if len(gen_ids) > len(ids) else gen_ids
text = tok.decode(cont_ids)
print("\n=== LFM2.5 GENERATED TEXT ===")
print(text)
print("\n=== (full sequence decode) ===")
print(tok.decode(gen_ids))
