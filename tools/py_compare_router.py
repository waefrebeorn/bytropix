#!/usr/bin/env python3
"""
Compare MoE router weights and scores between bytropix and llama.cpp.
Extract ffn_gate_inp.weight from GGUF, compute router scores with our input,
and compare against reference.
"""
import struct
import numpy as np

# GGUF reader
f = open('/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf', 'rb')
f.read(4); f.read(4); f.read(8); f.read(8)  # header
meta_kv = 54
for i in range(meta_kv):
    n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
    typ = struct.unpack('<i', f.read(4))[0]
    if typ == 8:
        n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
    elif typ == 9:
        arr_type = struct.unpack('<i', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len):
            if arr_type == 8: n = struct.unpack('<Q', f.read(8))[0]; f.read(n)
            else: f.read(4)
    elif typ in (10, 11, 12): f.read(8)
    elif typ == 6: f.read(4)
    elif typ == 7: f.read(1)
    elif typ in (0, 1): f.read(1)
    elif typ in (2, 3): f.read(2)
    elif typ in (4, 5): f.read(4)

tensor_info_start = f.tell()
tensors = []
for i in range(733):
    n = struct.unpack('<Q', f.read(8))[0]
    name = f.read(n).decode()
    n_dims = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
    ggml_type = struct.unpack('<i', f.read(4))[0]
    data_offset = struct.unpack('<Q', f.read(8))[0]
    tensors.append((name, n_dims, dims, ggml_type, data_offset))

data_blob_start = f.tell()
data_blob_start_aligned = ((data_blob_start + 31) // 32) * 32
print(f"Data blob at: {data_blob_start_aligned}")

# Find ffn_gate_inp.weight for layer 0
targets = [
    'blk.0.ffn_gate_inp.weight',   # router F32
    'blk.0.ffn_gate_exps.weight',  # gate IQ2_XXS
    'blk.0.ffn_up_exps.weight',    # up IQ2_XXS
    'blk.0.ffn_down_exps.weight',  # down IQ3_XXS or IQ4_XS
    'blk.0.ffn_gate_shexp.weight', # shared gate Q5_K
    'blk.0.ffn_up_shexp.weight',   # shared up Q5_K
    'blk.0.ffn_down_shexp.weight', # shared down Q6_K
    'blk.0.ffn_gate_inp_shexp.weight', # shared gate F32
]

for name, nd, dims, gtype, offset in tensors:
    if name in targets:
        abs_off = data_blob_start_aligned + offset
        print(f"\n{name}:")
        print(f"  ne = {dims}, type={gtype}, offset={offset}, abs={abs_off}")
        
        if gtype == 0:  # F32
            n_elems = 1
            for d in dims: n_elems *= d
            f.seek(abs_off)
            data = np.frombuffer(f.read(n_elems * 4), dtype=np.float32)
            if 'gate_inp.weight' in name and 'shexp' not in name:
                # router: [D_MODEL, N_EXPERTS] = [2048, 256]
                # ne[0] = 2048 (D_MODEL fastest), ne[1] = 256 (N_EXPERTS)
                # element at (expert e, dmodel k) = e * 2048 + k
                print(f"  Shape (Python index): {data.shape}")
                print(f"  First 16 values: {data[:16].tolist()}")
                print(f"  Stats: mean={np.mean(data):.6f} std={np.std(data):.6f} range=[{np.min(data):.6f},{np.max(data):.6f}]")
                
                # Compute router scores with our MoE input
                moe_input = np.frombuffer(open('/tmp/dbg_moe_input.bin','rb').read(), dtype=np.float32)
                print(f"  MoE input stats: mean={np.mean(moe_input):.6f} std={np.std(moe_input):.6f}")
                
                # Our code: score[e] = sum_k input[k] * weight[k + e * D_MODEL]
                # With ne[0]=2048 (D_MODEL fastest): weight at (k, e) = k + e * 2048
                scores = np.zeros(256, dtype=np.float64)
                for e in range(256):
                    for k in range(2048):
                        scores[e] += moe_input[k] * data[k + e * 2048]
                print(f"  Router scores[0..7]: {scores[:8].tolist()}")
                print(f"  Max score: {np.max(scores):.4f}, Min: {np.min(scores):.4f}")
                
                # Top-8
                top8_idx = np.argsort(-scores)[:8]
                top8_wgt = scores[top8_idx]
                # Normalize
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / np.sum(exp_scores)
                top8_probs = probs[top8_idx]
                top8_wgt_norm = top8_probs / np.sum(top8_probs)
                print(f"  Top-8 indices: {top8_idx.tolist()}")
                print(f"  Top-8 weights: {top8_wgt_norm.tolist()}")
                
                # Compare with our output
                our_scores = np.frombuffer(open('/tmp/dbg_moe_scores.bin','rb').read(), dtype=np.float32)
                our_topk_idx = np.frombuffer(open('/tmp/dbg_moe_topk_idx.bin','rb').read(), dtype=np.int32)
                our_topk_wgt = np.frombuffer(open('/tmp/dbg_moe_topk_wgt.bin','rb').read(), dtype=np.float32)
                print(f"\n  Our scores[0..7]: {our_scores[:8].tolist()}")
                print(f"  Our top-8 indices: {our_topk_idx.tolist()}")
                print(f"  Our top-8 weights: {our_topk_wgt.tolist()}")
                
                # Compare scores
                diff = scores - our_scores
                print(f"  Score max_diff: {np.max(np.abs(diff)):.10f}")
                print(f"  Score correlation: {np.corrcoef(scores, our_scores)[0,1]:.10f}")
                
            if 'gate_inp_shexp' in name:
                print(f"  First 8: {data[:8].tolist()}")

f.close()
