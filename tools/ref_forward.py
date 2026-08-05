"""ref_forward.py -- self-contained LFM2.5 reference forward (our own oracle).

Uses safetensors mmap (fast, no transformers 11h load path). Implements the
exact LFM2.5 hybrid math to mirror the C engine for byte-level comparison.
Dumps per-layer hidden norms + top-5 logits for a given token id sequence to
tools/hf_ref.json. The C engine (lfm2_test) must consume the SAME ids.

LFM2.5 block (technical report arxiv 2511.23404):
  hidden = embedding_norm(embed(tokens))
  for each layer:
    op_in = operator_norm(hidden)
    if conv:  (B,C,h) = in_proj(op_in)            # [3*cd, d]
              y = B * h ; z = causal_depthwise_conv(y) ; out = out_proj(C*z)
    else:     q/k/v proj -> q/k layernorm -> RoPE -> GQA -> o_proj
    hidden = hidden + op_out
    ffn_in = ffn_norm(hidden)
    ffn = w2(silu(w1(ffn_in)) * w3(ffn_in))
    hidden = hidden + ffn
  logits = lm_head(embedding_norm(hidden))   # tied embed
"""
import json, sys, math
import torch

MODEL = "D:/models/LFM2.5-2.6B"

def st_load_all(model_dir):
    from safetensors import safe_open
    import os, glob
    shards = sorted(glob.glob(os.path.join(model_dir, "model-*-of-*.safetensors")))
    tensors = {}
    for sh in shards:
        with safe_open(sh, framework="pt", device="cpu") as f:
            for n in f.keys():
                tensors[n] = f.get_tensor(n)
    return tensors

def rmsnorm(x, gamma, eps=1e-5):
    # x: [..., d]
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * gamma

def causal_depthwise_conv(y, w, k):
    # y: [T, C], w: [C, 1, k] (PyTorch depthwise Conv1d groups=C). Causal via left-pad.
    # out[t,c] = sum_j w[c,0,k-1-j] * in[t-j,c].
    import torch.nn.functional as F
    C = y.shape[1]
    yT = y.T.unsqueeze(0)  # [1, C, T]
    # flip kernel along k so F.conv1d(leftpad) == sum_j w[c,0,k-1-j]*in[t-j,c] (matches C)
    w_ = w.flip(-1)  # [C, 1, k] -> reversed k
    yp = torch.nn.functional.pad(yT, (k - 1, 0))
    out = F.conv1d(yp, w_, groups=C)  # [1, C, T]
    return out.squeeze(0).T  # [T, C]

def main():
    ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [597,3365,302,13086,11561,355]
    W = st_load_all(MODEL)
    embed = W["model.embed_tokens.weight"].float()            # [V, d]
    embed_norm = W["model.embedding_norm.weight"].float()     # [d] applied once AFTER layers
    d = embed.shape[1]
    cfg = json.load(open(f"{MODEL}/config.json"))
    n_layers = cfg["num_hidden_layers"]
    n_q = cfg["num_attention_heads"]; n_kv = cfg["num_key_value_heads"]
    hd = d // n_q; kv_dim = n_kv * hd
    ff = cfg["intermediate_size"]; cd = cfg["conv_dim"]
    theta = 1e7  # rope_parameters.rope_theta
    layer_types = cfg["layer_types"]

    hidden = embed[ids]  # [T, d] -- raw embed; norm applied ONCE after layers (HF Lfm2Model)

    def rope(vec, pos):
        # vec: [..., hd]
        freqs = theta ** (-2.0 * torch.arange(0, hd, 2) / hd)
        ang = pos * freqs
        c = torch.cos(ang); s = torch.sin(ang)
        v0 = vec[..., 0::2]; v1 = vec[..., 1::2]
        vec[..., 0::2] = v0 * c - v1 * s
        vec[..., 1::2] = v0 * s + v1 * c

    norms = []
    for l in range(n_layers):
        op_norm = W[f"model.layers.{l}.operator_norm.weight"].float()
        ffn_norm = W[f"model.layers.{l}.ffn_norm.weight"].float()
        op_in = rmsnorm(hidden, op_norm)
        if layer_types[l] == "conv":
            BCh = W[f"model.layers.{l}.conv.in_proj.weight"].float()  # [3cd, d]
            conv_w = W[f"model.layers.{l}.conv.conv.weight"].float()  # [cd, cd, k]
            out_proj = W[f"model.layers.{l}.conv.out_proj.weight"].float()  # [d, cd]
            proj = op_in @ BCh.T  # [T, 3cd]
            Bp = proj[:, :cd]; Cp = proj[:, cd:2*cd]; Hp = proj[:, 2*cd:]
            y = Bp * Hp
            z = causal_depthwise_conv(y, conv_w, 3)
            op_out = (Cp * z) @ out_proj.T
        else:
            q_w = W[f"model.layers.{l}.self_attn.q_proj.weight"].float()
            k_w = W[f"model.layers.{l}.self_attn.k_proj.weight"].float()
            v_w = W[f"model.layers.{l}.self_attn.v_proj.weight"].float()
            o_w = W[f"model.layers.{l}.self_attn.out_proj.weight"].float()
            qln = W[f"model.layers.{l}.self_attn.q_layernorm.weight"].float()
            kln = W[f"model.layers.{l}.self_attn.k_layernorm.weight"].float()
            q_proj = (op_in @ q_w.T)            # [T, n_q*hd]
            k_proj = (op_in @ k_w.T)            # [T, n_kv*hd]
            q = rmsnorm(q_proj.view(-1, hd), qln).view(-1, n_q*hd)   # per-head layernorm
            k = rmsnorm(k_proj.view(-1, hd), kln).view(-1, n_kv*hd)
            v = (op_in @ v_w.T).view(-1, n_kv, hd).reshape(-1, n_kv*hd)
            T = op_in.shape[0]
            q3 = q.reshape(T, n_q, hd); k3 = k.reshape(T, n_kv, hd)
            for t in range(T):
                rope(q3[t], t); rope(k3[t], t)
            q = q3.reshape(T, n_q, hd); k = k3.reshape(T, n_kv, hd)
            v3 = v.reshape(T, n_kv, hd)
            qpk = n_q // n_kv
            scale = 1.0 / math.sqrt(hd)
            # batched GQA attention: scores [T, n_q, T], softmax over last (kv positions)
            # q: [T, n_q, hd] -> expand kv group: qh -> kvh = qh // qpk
            qg = q.view(T, n_kv, qpk, hd)  # [qpos, n_kv, qpk, hd]
            scores = torch.einsum("abcd,ebd->abce", qg, k) * scale  # [qpos, n_kv, qpk, kpos]
            scores = torch.softmax(scores, dim=-1)
            ctx = torch.einsum("abce,ebd->abcd", scores, v3)      # [qpos, n_kv, qpk, hd]
            ctx = ctx.permute(0, 1, 2, 3).reshape(T, n_q, hd)
            out = ctx.reshape(T, n_q * hd) @ o_w.T

        hidden = hidden + op_out
        ffnin = rmsnorm(hidden, ffn_norm)
        w1 = W[f"model.layers.{l}.feed_forward.w1.weight"].float()
        w2 = W[f"model.layers.{l}.feed_forward.w2.weight"].float()
        w3 = W[f"model.layers.{l}.feed_forward.w3.weight"].float()
        g = torch.nn.functional.silu(ffnin @ w1.T) * (ffnin @ w3.T)
        hidden = hidden + g @ w2.T

        norms.append(float(hidden[-1].pow(2).mean().sqrt()))

    logits = embed @ rmsnorm(hidden[-1:], embed_norm).T  # tied lm_head via embedding_norm (HF Lfm2Model)
    logits = logits[0]
    order = torch.argsort(-logits)[:5]
    top5 = [(int(i), float(logits[i])) for i in order]
    for l, n in enumerate(norms):
        print(f"L{l} norm={n:.4f}")
    print("TOP5", top5)
    json.dump({"ids": ids, "norms": norms, "top5": top5}, open("tools/hf_ref.json", "w"), indent=2)
    print("WROTE tools/hf_ref.json")

if __name__ == "__main__":
    main()
