"""Compare our post-MoE dump vs reference post_moe dump for L0."""
import numpy as np
import struct

# Our L0 post-MoE: 16384 bytes = 4096 floats = 1 token
our = np.fromfile('/tmp/our_post_moe/layer_100.bin', dtype=np.float32)
print(f"Our post-MoE L0: {our.shape} floats, {our.nbytes} bytes")
print(f"  mean={our.mean():.6f} max={our.max():.6f} min={our.min():.6f} rms={np.sqrt((our**2).mean()):.6f}")

# Reference L0: 90112 bytes
ref_raw = open('/tmp/ref_post_moe/layer_00.bin', 'rb').read()
print(f"\nReference L0: {len(ref_raw)} bytes")

# Try as f32 and f16
as_f32 = np.frombuffer(ref_raw, dtype=np.float32)
as_f16 = np.frombuffer(ref_raw, dtype=np.float16)
print(f"  As f32: {as_f32.shape} floats")
print(f"  As f16: {as_f16.shape} floats")

# Check token count from size
for n_tok in range(1, 20):
    if len(ref_raw) == n_tok * 4096 * 4: # f32
        print(f"  Match: {n_tok} tokens × 4096 × 4 = {n_tok*4096*4} bytes (f32)")
    if len(ref_raw) == n_tok * 4096 * 2: # f16
        print(f"  Match: {n_tok} tokens × 4096 × 2 = {n_tok*4096*2} bytes (f16)")

# Check if it's actually f16 stored as f32 (first half valid)
# 90112/4 = 22528. 22528/2 = 11264 f16 values. 11264/4096 = 2.75
# Let's check by reading as f16 and looking at f32 representation
f32_view = np.frombuffer(ref_raw[:45056], dtype=np.float32)
f16_view = np.frombuffer(ref_raw[:45056], dtype=np.float16)
print(f"\n  First 45056 bytes as f32: {f32_view.shape} -> mean={f32_view.mean():.6f}")
print(f"  First 45056 bytes as f16: {f16_view.shape} -> mean={float(f16_view.mean()):.6f}")

# Maybe it's all f32 but there are more tokens than we think
# Check values in the f32 interpretation
print(f"\n  f32 mean={as_f32.mean():.6f} max={as_f32.max():.6f} min={as_f32.min():.6f} rms={np.sqrt((as_f32**2).mean()):.6f}")

# Look at where reasonable values end
for offset in range(0, min(len(ref_raw), 200000), 4096*4):
    chunk = np.frombuffer(ref_raw[offset:offset+4096*4], dtype=np.float32)
    if len(chunk) > 0:
        print(f"  f32 block at byte {offset}: mean={chunk.mean():.6f} max={chunk.max():.6f} min={chunk.min():.6f}")

# Check if reference stores BOS token separately  
# Or maybe there's just 1 token in f32 and it's 4096 floats = 16384 bytes
# But 90112 != 16384. Maybe it's 5.5 tokens = garbage in last half-token?
# Or nelements is different from n_tokens*D_MODEL
# Let's check what happens if we assume the last token is what we want
# 22528 floats = last token is at [22528-4096:22528] = [18432:22528]
# But this doesn't align with the file size neatly
if as_f32.shape[0] > 4096:
    for i in range(as_f32.shape[0] // 4096):
        blk = as_f32[i*4096:(i+1)*4096]
        print(f"  f32 block {i}: mean={blk.mean():.6f} max={blk.max():.6f} min={blk.min():.6f} rms={np.sqrt((blk**2).mean()):.6f}")

print(f"\nOur shape={our.shape}")
# Compare last 4096 elements
if as_f32.shape[0] >= 4096:
    ref_last = as_f32[-4096:]
    dot = np.dot(our, ref_last)
    norm_our = np.linalg.norm(our)
    norm_ref = np.linalg.norm(ref_last)
    cs = dot / (norm_our * norm_ref + 1e-30)
    print(f"\nOur vs ref (last 4096): cos_sim={cs:.6f}")

# Also try with f16 interpretation
if as_f16.shape[0] >= 4096:
    ref_last_f16 = as_f16[-4096:].astype(np.float32)
    dot = np.dot(our, ref_last_f16)
    norm_our = np.linalg.norm(our)
    norm_ref = np.linalg.norm(ref_last_f16)
    cs = dot / (norm_our * norm_ref + 1e-30)
    print(f"Our vs ref f16 (last 4096): cos_sim={cs:.6f}")
