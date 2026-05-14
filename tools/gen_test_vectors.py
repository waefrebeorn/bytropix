#!/usr/bin/env python3
"""
Generate test vectors for C SSM forward pass verification.
Outputs binary files with input, weights, and expected output.
Usage: python3 tools/gen_test_vectors.py
"""
import numpy as np
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.ssm_reference import (
    ssm_layer_forward, gqa_layer_forward,
    D_MODEL, D_INNER, SSM_K_HEADS, SSM_V_HEADS, SSM_D_STATE,
    KEY_DIM, VALUE_DIM, CONV_DIM, CONV_KERNEL, DT_RANK
)

B, T = 1, 4
np.random.seed(42)

def gen_ssm_test():
    """Generate SSM test with known weights and save to file"""
    np.random.seed(123)  # Use different seed for SSM
    # Create weights matching C struct layout
    w = {}
    w['attn_qkv.weight'] = np.random.randn(D_MODEL, KEY_DIM * 2 + VALUE_DIM).astype(np.float32) * 0.01
    w['attn_gate.weight'] = np.random.randn(D_MODEL, VALUE_DIM).astype(np.float32) * 0.01
    w['ssm_beta.weight'] = np.random.randn(D_MODEL, DT_RANK).astype(np.float32) * 0.01
    w['ssm_alpha.weight'] = np.random.randn(D_MODEL, DT_RANK).astype(np.float32) * 0.01
    w['ssm_dt.bias'] = np.random.randn(DT_RANK).astype(np.float32) * 0.01
    w['ssm_a'] = np.random.randn(DT_RANK).astype(np.float32) * 0.01
    w['ssm_conv1d.weight'] = np.random.randn(CONV_KERNEL, CONV_DIM).astype(np.float32) * 0.01
    w['ssm_norm.weight'] = np.ones(SSM_D_STATE, dtype=np.float32)
    w['ssm_out.weight'] = np.random.randn(VALUE_DIM, D_MODEL).astype(np.float32) * 0.01

    x = np.random.randn(B, T, D_MODEL).astype(np.float32) * 0.1
    ssm_state = np.zeros((SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE), dtype=np.float32)
    conv_state = np.zeros((B, CONV_KERNEL - 1, CONV_DIM), dtype=np.float32)

    # Run Python forward
    y = ssm_layer_forward(x, w)

    # Save to binary file
    path = 'data/ssm_test_vectors.bin'
    with open(path, 'wb') as f:
        # Write dimensions
        f.write(struct.pack('iiii', B, T, D_MODEL, VALUE_DIM))
        # Write input x
        f.write(x.tobytes())
        # Write weights
        f.write(w['attn_qkv.weight'].tobytes())
        f.write(w['attn_gate.weight'].tobytes())
        f.write(w['ssm_beta.weight'].tobytes())
        f.write(w['ssm_alpha.weight'].tobytes())
        f.write(w['ssm_dt.bias'].tobytes())
        f.write(w['ssm_a'].tobytes())
        f.write(w['ssm_conv1d.weight'].tobytes())
        f.write(w['ssm_norm.weight'].tobytes())
        f.write(w['ssm_out.weight'].tobytes())
        # Write expected output
        f.write(y.tobytes())

    print(f"SSM test vectors written to {path}")
    print(f"  Input shape: ({B},{T},{D_MODEL}) = {B*T*D_MODEL} floats")
    print(f"  Output shape: ({B},{T},{D_MODEL}) = {B*T*D_MODEL} floats")
    print(f"  Output range: [{y.min():.6f}, {y.max():.6f}]")
    return path


def gen_gqa_test():
    """Generate GQA test vectors"""
    np.random.seed(456)  # Different seed
    w = {}
    w['attn_q.weight'] = np.random.randn(D_MODEL, 16 * 256 * 2).astype(np.float32) * 0.01
    w['attn_k.weight'] = np.random.randn(D_MODEL, 2 * 256).astype(np.float32) * 0.01
    w['attn_v.weight'] = np.random.randn(D_MODEL, 2 * 256).astype(np.float32) * 0.01
    w['attn_output.weight'] = np.random.randn(16 * 256, D_MODEL).astype(np.float32) * 0.01
    w['attn_q_norm.weight'] = np.ones(256, dtype=np.float32)
    w['attn_k_norm.weight'] = np.ones(256, dtype=np.float32)

    x = np.random.randn(B, T, D_MODEL).astype(np.float32) * 0.1

    y = gqa_layer_forward(x, w)
    assert not np.any(np.isnan(y)), f"GQA output has NaN! y[0,:8]={y[0,0,:8]}"
    print(f"GQA output OK: range=[{y.min():.6f}, {y.max():.6f}]")
    y = np.ascontiguousarray(y) if not y.flags['C_CONTIGUOUS'] else y

    path = 'data/gqa_test_vectors.bin'
    with open(path, 'wb') as f:
        f.write(struct.pack('iiii', B, T, D_MODEL, 16 * 256))
        f.write(x.tobytes())
        f.write(w['attn_q.weight'].tobytes())
        f.write(w['attn_k.weight'].tobytes())
        f.write(w['attn_v.weight'].tobytes())
        f.write(w['attn_output.weight'].tobytes())
        f.write(w['attn_q_norm.weight'].tobytes())
        f.write(w['attn_k_norm.weight'].tobytes())
        f.write(y.tobytes())

    print(f"GQA test vectors written to {path}")
    return path


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    gen_ssm_test()
    gen_gqa_test()
    print("\nTest vectors ready. Compile and run test_ssm_forward.c to verify.")
