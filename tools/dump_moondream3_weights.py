#!/usr/bin/env python3
"""
Moondream3 Vision Encoder Weight Dump Script (Phase 5b.1)

Loads cached safetensors from ~/.cache/huggingface/models--moondream--moondream3-preview/
and dumps ONLY the vision encoder weights (model.vision.*) to f32 binary.

Architecture (from config.py VisionConfig):
  - Type: SigLIP-style ViT
  - Depth: 27 layers (enc_n_layers)
  - Hidden: 1152 (enc_dim)
  - Intermediate: 4304 (enc_ff_dim)
  - Heads: 16 (enc_n_heads)
  - Patch size: 14×14 (enc_patch_size)
  - Crop size: 378×378 (crop_size, NOT 448 as previously documented)
  - Output dim: 2048 (proj_out_dim, matches text hidden)
  - Proj inner: 8192 (proj_inner_dim)
  - Grid: 27×27 = 729 patches
  - Activation: GELU (approximate tanh)
  - Weights stored as BF16

Weight naming (from model.safetensors.index.json):
  model.vision.patch_emb.{weight,bias}
  model.vision.pos_emb                         — learned position embeddings
  model.vision.post_ln.{weight,bias}
  model.vision.proj_mlp.fc1.{weight,bias}
  model.vision.proj_mlp.fc2.{weight,bias}
  model.vision.blocks.{0-26}.attn.qkv.{weight,bias}
  model.vision.blocks.{0-26}.attn.proj.{weight,bias}
  model.vision.blocks.{0-26}.ln1.{weight,bias}
  model.vision.blocks.{0-26}.ln2.{weight,bias}
  model.vision.blocks.{0-26}.mlp.fc1.{weight,bias}
  model.vision.blocks.{0-26}.mlp.fc2.{weight,bias}

Output:
  data/moondream3_vision_weights.bin    — flat f32 binary, all weights concatenated
  data/moondream3_vision_config.txt     — architecture parameters
  data/moondream3_vision_index.json     — tensor name → offset/size in binary
"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "models--moondream--moondream3-preview"
SNAPSHOT_DIR = CACHE_DIR / "snapshots"

# Find snapshot
if not SNAPSHOT_DIR.exists():
    print(f"ERROR: Cache not found at {CACHE_DIR}")
    print("The model may not be cached. Try downloading with:")
    print("  pip install transformers && python -c \"from transformers import AutoModel; AutoModel.from_pretrained('moondream/moondream3-preview', trust_remote_code=True)\"")
    sys.exit(1)

snapshots = list(SNAPSHOT_DIR.iterdir())
if not snapshots:
    print(f"ERROR: No snapshots found in {SNAPSHOT_DIR}")
    sys.exit(1)
snapshot = snapshots[0]
print(f"Found snapshot: {snapshot.name}")

# Load index
index_path = snapshot / "model.safetensors.index.json"
with open(index_path) as f:
    index = json.load(f)

weight_map = index["weight_map"]

# ── Architecture Spec (from config.py VisionConfig) ──────────────────
ARCH = {
    "enc_dim": 1152,          # hidden size
    "enc_patch_size": 14,     # patch size
    "enc_n_layers": 27,       # depth
    "enc_ff_dim": 4304,       # MLP intermediate
    "enc_n_heads": 16,        # attention heads
    "proj_out_dim": 2048,     # output projection
    "proj_inner_dim": 8192,   # projection MLP hidden
    "crop_size": 378,         # actual input crop size (NOT 448)
    "in_channels": 3,         # RGB
}

grid_size = ARCH["crop_size"] // ARCH["enc_patch_size"]  # 27
num_patches = grid_size * grid_size  # 729

print(f"\nArchitecture:")
print(f"  Depth: {ARCH['enc_n_layers']} layers")
print(f"  Hidden: {ARCH['enc_dim']}")
print(f"  Intermediate (FF): {ARCH['enc_ff_dim']}")
print(f"  Heads: {ARCH['enc_n_heads']} (head_dim={ARCH['enc_dim']//ARCH['enc_n_heads']})")
print(f"  Patch size: {ARCH['enc_patch_size']}×{ARCH['enc_patch_size']}")
print(f"  Crop size: {ARCH['crop_size']}×{ARCH['crop_size']}")
print(f"  Grid: {grid_size}×{grid_size} = {num_patches} patches")
print(f"  Proj inner: {ARCH['proj_inner_dim']} → out: {ARCH['proj_out_dim']}")


# ── Low-level safetensors reader (no PyTorch dependency) ──────────────
def bf16_to_f32(bf16_bytes):
    """Convert bfloat16 bytes to float32 numpy array.
    bf16 bits: [sign(1), exp(8), mantissa(7)]
    f32 bits:  [sign(1), exp(8), mantissa(23)]
    Pad lower 16 bits with zeros."""
    u16 = np.frombuffer(bf16_bytes, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def load_safetensors(filepath, wanted_keys=None):
    """Load tensors from a safetensors file.
    Returns dict of {key: numpy_array_f32}.
    If wanted_keys is set, only load those keys (faster)."""
    if not os.path.exists(filepath):
        return {}

    with open(filepath, 'rb') as f:
        # Read header size (8 bytes, little-endian u64)
        header_size = struct.unpack('<Q', f.read(8))[0]
        # Read header JSON
        header = json.loads(f.read(header_size))
        
        result = {}
        for key, info in header.items():
            if wanted_keys is not None and key not in wanted_keys:
                continue

            dtype_str = info["dtype"]
            shape = info["shape"]
            start, end = info["data_offsets"]

            # Seek to data position
            f.seek(8 + header_size + start)
            raw_bytes = f.read(end - start)

            if dtype_str == "BF16":
                tensor = bf16_to_f32(raw_bytes).reshape(shape)
            elif dtype_str == "F32":
                tensor = np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape)
            elif dtype_str == "F16":
                tensor = np.frombuffer(raw_bytes, dtype=np.float16).reshape(shape).astype(np.float32)
            else:
                print(f"  WARNING: unknown dtype {dtype_str} for {key}, skipping")
                continue

            result[key] = tensor

    return result


# ── Collect vision weights ──────────────────────────────────────────────
def collect_vision_weights(weight_map, snapshot):
    """Load all model.vision.* weights from safetensors files."""
    # Find which files contain vision weights
    vision_keys = {}
    for key, fname in weight_map.items():
        if key.startswith("model.vision."):
            vision_keys[key] = fname

    vision_files = set(vision_keys.values())
    print(f"\nVision weights found across {len(vision_files)} file(s):")
    for f in sorted(vision_files):
        print(f"  {f}")

    # Load tensors
    tensors = {}
    for fname in sorted(vision_files):
        filepath = snapshot / fname
        if not filepath.exists():
            # Try blobs directory
            blob_fname = os.path.basename(fname)
            filepath = CACHE_DIR / "blobs" / blob_fname
            # Resolve symlink target
            if filepath.exists() and filepath.is_symlink():
                filepath = filepath.resolve()
        if not filepath.exists():
            print(f"  WARNING: {fname} not found, trying alternative resolution...")
            continue

        print(f"\nLoading {fname}...")
        # Get the subset of vision keys in THIS file
        keys_in_file = {k for k, v in vision_keys.items() if v == fname}
        file_tensors = load_safetensors(str(filepath), wanted_keys=keys_in_file)
        tensors.update(file_tensors)

        for key in sorted(file_tensors):
            t = file_tensors[key]
            short = key.replace("model.vision.", "")
            mb = t.nbytes / (1024 * 1024)
            print(f"  {short:50s} {str(list(t.shape)):25s} f32 {mb:.2f}MB")

    print(f"\nTotal vision tensors loaded: {len(tensors)}")
    return tensors


# ── Dump to binary ──────────────────────────────────────────────────────
def dump_to_binary(tensors, output_dir):
    """Dump vision weights to flat f32 binary with index."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_path = output_dir / "moondream3_vision_weights.bin"
    idx_path = output_dir / "moondream3_vision_index.json"

    # Order: sorted by key for deterministic layout
    ordered_keys = sorted(tensors.keys())

    index = {}
    offset = 0

    with open(bin_path, "wb") as f:
        for key in ordered_keys:
            t = tensors[key]
            data = t.tobytes()
            shape = list(t.shape)
            n_elems = int(np.prod(shape))
            n_bytes = n_elems * 4  # f32 = 4 bytes

            index[key] = {
                "offset": offset,
                "size_bytes": n_bytes,
                "shape": shape,
                "n_elems": n_elems,
            }
            f.write(data)
            offset += n_bytes

            short = key.replace("model.vision.", "")
            print(f"  Dumped {short:50s} offset={offset-n_bytes:>10d}  size={n_bytes:>10d}B")

    # Save index
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)

    total_mb = offset / (1024 * 1024)
    print(f"\nBinary dumped: {bin_path}")
    print(f"  Total size: {offset} bytes ({total_mb:.2f} MB)")
    print(f"  Total tensors: {len(ordered_keys)}")
    print(f"Index saved: {idx_path}")

    return bin_path, idx_path


# ── Save config ─────────────────────────────────────────────────────────
def save_config(arch, output_dir):
    """Save architecture config as text file."""
    output_dir = Path(output_dir)
    config_path = output_dir / "moondream3_vision_config.txt"

    lines = [
        "# Moondream3 Vision Encoder Architecture",
        "# Source: config.py VisionConfig + safetensors index",
        "",
        f"enc_dim         = {arch['enc_dim']}",
        f"enc_patch_size  = {arch['enc_patch_size']}",
        f"enc_n_layers    = {arch['enc_n_layers']}",
        f"enc_ff_dim      = {arch['enc_ff_dim']}",
        f"enc_n_heads     = {arch['enc_n_heads']}",
        f"proj_out_dim    = {arch['proj_out_dim']}",
        f"proj_inner_dim  = {arch['proj_inner_dim']}",
        f"crop_size       = {arch['crop_size']}",
        f"grid_size       = {arch['crop_size'] // arch['enc_patch_size']}",
        f"num_patches     = {(arch['crop_size'] // arch['enc_patch_size']) ** 2}",
        f"head_dim        = {arch['enc_dim'] // arch['enc_n_heads']}",
        f"activation      = gelu_approx (tanh)",
        f"weight_dtype    = bfloat16 (original) → float32 (dumped)",
        "",
        "# Layer layout (per block 0..26):",
        "#   model.vision.blocks.N.ln1:           LayerNorm(1152)       [weight, bias]",
        "#   model.vision.blocks.N.attn.qkv:      Linear(1152 → 3456)  [weight, bias]  — fused Q,K,V",
        "#   model.vision.blocks.N.attn.proj:     Linear(1152 → 1152)  [weight, bias]",
        "#   model.vision.blocks.N.ln2:           LayerNorm(1152)       [weight, bias]",
        "#   model.vision.blocks.N.mlp.fc1:       Linear(1152 → 4304)  [weight, bias]",
        "#   model.vision.blocks.N.mlp.fc2:       Linear(4304 → 1152)  [weight, bias]",
        "",
        "# Global weights:",
        "#   model.vision.patch_emb:              Linear(588 → 1152)    [weight, bias]  14*14*3=588",
        "#   model.vision.pos_emb:                Parameter(1, 729, 1152)",
        "#   model.vision.post_ln:                LayerNorm(1152)       [weight, bias]",
        "#   model.vision.proj_mlp.fc1:           Linear(2304 → 8192)  [weight, bias]  — concatenated global+reconstructed",
        "#   model.vision.proj_mlp.fc2:           Linear(8192 → 2048)  [weight, bias]",
        "",
        "# NOTE: Actual crop_size=378, not 448 as previously documented in the plan.",
        "# This gives a 27×27 grid (729 patches).",
        "# The 448×448 in the plan is incorrect; 378×378 is from VisionConfig.crop_size.",
    ]

    with open(config_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Config saved: {config_path}")


# ── Verify dumped weights ──────────────────────────────────────────────
def verify_dump(bin_path, idx_path):
    """Reload binary and check sizes match expected."""
    with open(idx_path) as f:
        index = json.load(f)

    file_size = os.path.getsize(bin_path)
    total_indexed = sum(v["size_bytes"] for v in index.values())

    print(f"\nVerification:")
    print(f"  Binary file size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
    print(f"  Indexed total:    {total_indexed} bytes ({total_indexed/1024/1024:.2f} MB)")
    match_size = "✓" if file_size == total_indexed else "✗ MISMATCH"
    print(f"  Size match: {match_size}")

    # Verify expected number of weights
    n_blocks = ARCH["enc_n_layers"]
    # patch_emb(w+b), pos_emb, post_ln(w+b), proj_mlp.fc1(w+b), proj_mlp.fc2(w+b) = 9
    # per block: ln1(w+b), attn.qkv(w+b), attn.proj(w+b), ln2(w+b), mlp.fc1(w+b), mlp.fc2(w+b) = 12
    expected_weights = 9 + n_blocks * 12

    n_actual = len(index)
    match_count = "✓" if n_actual == expected_weights else "✗"
    print(f"  Expected tensors: {expected_weights}")
    print(f"  Actual tensors:   {n_actual}  {match_count}")

    # Check pos_emb shape
    pos_emb_key = "model.vision.pos_emb"
    if pos_emb_key in index:
        pe = index[pos_emb_key]
        print(f"\n  pos_emb shape: {pe['shape']}")
        if pe["shape"] == [1, 729, 1152]:
            print(f"  → ✓ 27×27 grid (729 patches)")
        else:
            print(f"  → ✗ expected [1, 729, 1152]")

    return file_size == total_indexed and n_actual == expected_weights


# ── Size breakdown ──────────────────────────────────────────────────────
def print_size_breakdown(tensors):
    """Print size breakdown by component category."""
    categories = {
        "patch_emb": 0,
        "pos_emb": 0,
        "post_ln": 0,
        "proj_mlp": 0,
        "blocks.attn": 0,
        "blocks.mlp": 0,
        "blocks.ln": 0,
    }

    for key, t in tensors.items():
        n_bytes = t.nbytes
        if "patch_emb" in key:
            categories["patch_emb"] += n_bytes
        elif "pos_emb" in key:
            categories["pos_emb"] += n_bytes
        elif "post_ln" in key:
            categories["post_ln"] += n_bytes
        elif "proj_mlp" in key:
            categories["proj_mlp"] += n_bytes
        elif "attn" in key:
            categories["blocks.attn"] += n_bytes
        elif "mlp" in key:
            categories["blocks.mlp"] += n_bytes
        elif "ln" in key:
            categories["blocks.ln"] += n_bytes

    print(f"\n{'='*60}")
    print(f"Size breakdown (f32):")
    print(f"{'='*60}")
    total = 0
    for cat, size in sorted(categories.items()):
        mb = size / (1024 * 1024)
        print(f"  {cat:20s} {mb:8.2f} MB")
        total += size
    print(f"  {'─'*30}")
    print(f"  {'TOTAL':20s} {total/1024/1024:8.2f} MB")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    output_dir = Path(__file__).resolve().parent.parent / "data"

    print("=" * 60)
    print("Moondream3 Vision Weight Dump")
    print("=" * 60)

    # Collect weights from cached safetensors
    tensors = collect_vision_weights(weight_map, snapshot)

    if not tensors:
        print("\nERROR: No vision tensors loaded!")
        print("Check that the model is properly cached and safetensors files exist.")
        sys.exit(1)

    # Print size breakdown
    print_size_breakdown(tensors)

    # Dump to binary
    bin_path, idx_path = dump_to_binary(tensors, output_dir)

    # Save config
    save_config(ARCH, output_dir)

    # Verify
    ok = verify_dump(bin_path, idx_path)
    print(f"\n{'=' * 60}")
    print(f"Status: {'SUCCESS' if ok else 'PARTIAL (check warnings above)'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
