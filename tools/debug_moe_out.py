"""Dump dequantized expert weight elements for L0 gate_exps.
Compare with reference by computing what the MoE output should be."""
import numpy as np
import struct

# Read our MoE input (normed) and output (ffn)
moe_in = np.fromfile('/tmp/moe_dump/moe_in_00.bin', dtype=np.float32)
moe_out = np.fromfile('/tmp/moe_dump/moe_out_00.bin', dtype=np.float32)

print("Input stats:", moe_in.mean(), moe_in.max(), moe_in.min())
print("Output stats:", moe_out.mean(), moe_out.max(), moe_out.min())

# Check the dequantized expert weight. The raw data is at some offset.
# Let me compute: gate = x @ gate_w (4096 x 512)
# For position j=0: gate_out[0] = sum_k x[k] * gate_w[k]
# For our dequantized expert: if this is too large, weights are wrong

# Print output values to see pattern  
print("\nFirst 20 output values:")
print(moe_out[:20])

# Check if output is just the input with wrong scaling
print("\nCorrelation with input:", np.corrcoef(moe_in[:2048], moe_out[:2048])[0,1] if len(moe_out)>=2048 else "N/A")

# Check if some values are reasonable  
mask_small = np.abs(moe_out) < 100
print(f"\nOutput abs < 100: {mask_small.sum()} / {len(moe_out)}")
print(f"  Those values: mean={moe_out[mask_small].mean():.4f}" if mask_small.sum() > 0 else "")

# Look at specific positions to find if some are 0
print(f"\nOutput at indices 0, 512, 1024, 2048:")
for i in [0, 512, 1024, 2048, 4095]:
    if i < len(moe_out):
        print(f"  [{i}] = {moe_out[i]:.4f}")

# The output should be the SHARED expert contribution + routed expert contributions
# The shared expert is computed from the SAME moe_in
# Let me compute: what would a single expert contribute?
# For expert j=0: gate = x @ W_gate[:,0], up = x @ W_up[:,0], down = act @ W_down[:,0]
# If weights have mean ~0, the output should be mean ~0 * mean(weights)^2 ~ 0

# Maybe the issue is an index mismatch in the dequant. Let me check if
# the output is consistent with reading from the wrong data offset.

print("\nTrying to understand value range:")
print(f"  |moe_in| max: {np.max(np.abs(moe_in)):.4f}")
print(f"  |moe_out| ranges:")
percentiles = [0, 25, 50, 75, 90, 99]
for p in percentiles:
    print(f"    {p}%: {np.percentile(np.abs(moe_out), p):.4f}")
print(f"    100%: {np.max(np.abs(moe_out)):.4f}")

# Check if the output is actually the shared expert output only
# The shared expert uses SHARED_D_FF=512 and output D_MODEL=4096
# out_s[j] = sum_k sa[k] * sh_down[k + j * SHARED_D_FF]

# Let me try: if we zero out the input (x=0), what happens?
# The shared expert should produce: gate=0, silu(0)=0, up=0, act=0, out=0
# So the garbage MUST come from the routed experts

# The issue might be that find_cached returns wrong pointer  
# or the expert weights within a single expert are garbage

# Let me check the layer_100 versus moe_out
lay100 = np.fromfile('/tmp/moe_dump/layer_100.bin', dtype=np.float32)  
print(f"\npost_moe (= pre_moe + moe_out): vs layer_100:")
diff = moe_out - (lay100 - np.fromfile('/tmp/moe_dump/layer_00.bin', dtype=np.float32))
print(f"  moe_out - (lay100 - lay00): max|diff|={np.max(np.abs(diff)):.8f}")

# Add a quick check: what's the value of the first dequantized element
# of the expert weights for expert 0? We'd need to read the raw file.
# Instead, let me check if the output has any structure:
# Check if out[i] / out[i+1] follows a pattern
print(f"\nOutput ratio pattern (idx 0-10 / 4096-4106):")
for i in range(10):
    if i+4096 < len(moe_out):
        ratio = moe_out[i] / (moe_out[i+4096] + 1e-30)
        print(f"  moe_out[{i}] / moe_out[{i+4096}] = {ratio:.4f}")
