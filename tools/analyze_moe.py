"""Analyze MoE input/output dumps for L0."""
import numpy as np

# Load data
moe_in = np.fromfile('/tmp/moe_dump/moe_in_00.bin', dtype=np.float32)
moe_out = np.fromfile('/tmp/moe_dump/moe_out_00.bin', dtype=np.float32)
pre_moe = np.fromfile('/tmp/moe_dump/layer_00.bin', dtype=np.float32)   # post-SSM residual
post_moe = np.fromfile('/tmp/moe_dump/layer_100.bin', dtype=np.float32) # post-MoE residual

print(f"moe_in:  {moe_in.shape}  mean={moe_in.mean():.4f} max={moe_in.max():.4f} min={moe_in.min():.4f} rms={np.sqrt((moe_in**2).mean()):.4f}")
print(f"moe_out: {moe_out.shape} mean={moe_out.mean():.4f} max={moe_out.max():.4f} min={moe_out.min():.4f} rms={np.sqrt((moe_out**2).mean()):.4f}")
print(f"pre_moe:  {pre_moe.shape}  mean={pre_moe.mean():.4f} max={pre_moe.max():.4f} min={pre_moe.min():.4f} rms={np.sqrt((pre_moe**2).mean()):.4f}")
print(f"post_moe: {post_moe.shape} mean={post_moe.mean():.4f} max={post_moe.max():.4f} min={post_moe.min():.4f} rms={np.sqrt((post_moe**2).mean()):.4f}")

# Verify: post_moe should equal pre_moe + moe_out (if moe ran)
diff = post_moe - (pre_moe + moe_out)
print(f"\npost_moe - (pre_moe + moe_out): mean={diff.mean():.8f} max={diff.max():.8f}")

# Alternative: moe_out = post_moe - pre_moe
alt_diff = moe_out - (post_moe - pre_moe)
print(f"moe_out - (post_moe - pre_moe): mean={alt_diff.mean():.8f} max={alt_diff.max():.8f}")

# Compare moe_in (normed) with pre_moe (residual)
# post-attn RMSNorm: normed should be RMSNorm(residual)
rms_pre = np.sqrt((pre_moe**2).mean())
normed_expected = pre_moe / (rms_pre + 1e-6)  # simplified RMSNorm without weight
print(f"\nmoe_in vs pre_moe/rms: cos_sim={np.dot(moe_in, pre_moe)/(np.linalg.norm(moe_in)*np.linalg.norm(pre_moe)+1e-30):.6f}")

# Check NaN in moe_out
has_nan = np.isnan(moe_out).any() or np.isinf(moe_out).any()
print(f"\nmoe_out has NaN/Inf: {has_nan}")
print(f"moe_out nonzero: {(np.abs(moe_out) > 1e-10).sum()} / {moe_out.shape[0]}")
print(f"moe_out abs > 100: {(np.abs(moe_out) > 100).sum()}")
print(f"moe_out abs > 10000: {(np.abs(moe_out) > 10000).sum()}")
print(f"moe_out abs > 1e6: {(np.abs(moe_out) > 1e6).sum()}")

# Check if moe_out looks like identity (moe skipped, memcpy normed -> ffn)
diff_identity = moe_out - moe_in
print(f"\nmoe_out - moe_in (identity check): mean={diff_identity.mean():.8f} max={diff_identity.max():.8f} max_abs={np.max(np.abs(diff_identity)):.8f}")

# If it's IDENTITY, then post_moe = pre_moe + normed
# And the garbage must come from somewhere else
print(f"\nExpected post_moe='residual+ffn', if ffn=normed:")
id_post = pre_moe + moe_in
diff_id = post_moe - id_post
print(f"  post_moe - (pre_moe + moe_in): mean={diff_id.mean():.6f} max={diff_id.max():.6f} rms={np.sqrt((diff_id**2).mean()):.6f}")

# Print some values from each array
print(f"\nFirst 10 values:")
print(f"  moe_in:  {moe_in[:10]}")
print(f"  moe_out: {moe_out[:10]}")
print(f"  pre_moe: {pre_moe[:10]}")
print(f"  post_moe:{post_moe[:10]}")
