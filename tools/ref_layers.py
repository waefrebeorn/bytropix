import torch, json, sys
MODEL="D:/models/LFM2.5-2.6B"
from transformers import AutoModelForCausalLM, AutoTokenizer

# CRITICAL: use add_special_tokens=False so the ids match what the C engine
# feeds (the C engine does a raw embed[id] lookup, no BOS, no specials).
# Both sides must consume the IDENTICAL id sequence or the comparison is void.
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "The future of artificial intelligence is"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, trust_remote_code=True)
model.eval()

ids = tok.encode(PROMPT, add_special_tokens=False)
print("SEED", ids, flush=True)

norms = []
def make_hook(idx):
    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hp = hs[0, -1]
        norms.append((idx, float(hp.float().pow(2).mean().sqrt())))
    return hook

hooks = []
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.register_forward_hook(make_hook(i)))

with torch.no_grad():
    t = torch.tensor([ids])
    out = model(t)
    logits = out.logits[0, -1].float().numpy()

import numpy as np
order = np.argsort(-logits)[:5]
top5 = [(int(i), float(logits[i])) for i in order]
for idx, nrm in norms:
    print(f"L{idx} norm={nrm:.4f}", flush=True)
print("HF_TOP5", top5, flush=True)

ref = {
    "prompt": PROMPT,
    "ids": ids,
    "norms": [n for _, n in norms],
    "top5": top5,
}
with open("tools/hf_ref.json", "w") as f:
    json.dump(ref, f, indent=2)
print("WROTE tools/hf_ref.json", flush=True)
