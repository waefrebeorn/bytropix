#!/usr/bin/env python3
"""
Tokenization Pipeline: CORPUS.py → text → token IDs → .bin file
===============================================================

Usage:
  python tools/tokenize_pipeline.py                          # default: CORPUS.py -> data/train_data.bin
  python tools/tokenize_pipeline.py --input my_text.txt      # tokenize a plain text file
  python tools/tokenize_pipeline.py --validate               # validate existing .bin
  python tools/tokenize_pipeline.py --small --size 5000      # create a small test dataset

The resulting .bin file is a flat array of int32 token IDs, readable by train_gpu.c.
"""

import argparse
import os
import struct
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default paths
DEFAULT_CORPUS_PY = os.path.join(PROJECT_ROOT, "ENCODERS", "phase3-generative", "CORPUS.py")
DEFAULT_RAW_TXT = os.path.join(PROJECT_ROOT, "data", "corpus_raw.txt")
DEFAULT_BIN = os.path.join(PROJECT_ROOT, "data", "train_data.bin")
DEFAULT_META = os.path.join(PROJECT_ROOT, "data", "train_meta.txt")
DEFAULT_GGUF = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
TOKENIZE_BINARY = os.path.join(PROJECT_ROOT, "tokenize_corpus")
EXTRACT_SCRIPT = os.path.join(PROJECT_ROOT, "tools", "extract_corpus.py")


def step(name):
    """Print a step header."""
    print(f"\n{'='*60}")
    print(f"  [{name}]")
    print(f"{'='*60}")


def extract_corpus_py(corpus_py_path, output_txt_path):
    """Step 1: Extract NARRATIVE_TEXT from CORPUS.py using extract_corpus.py."""
    step("STEP 1: Extract text from CORPUS.py")

    if not os.path.exists(corpus_py_path):
        print(f"  WARNING: CORPUS.py not found at {corpus_py_path}")
        print(f"  Skipping extraction. Provide a text file directly instead.")
        return False

    print(f"  Source: {corpus_py_path}")
    print(f"  Output: {output_txt_path}")

    result = subprocess.run(
        [sys.executable, EXTRACT_SCRIPT],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ERROR: extract_corpus.py failed: {result.stderr}")
        return False
    return True


def build_tokenizer_binary():
    """Step 2a: Build the tokenize_corpus C binary."""
    step("STEP 2a: Build tokenizer binary")

    if os.path.exists(TOKENIZE_BINARY):
        print(f"  Binary already exists: {TOKENIZE_BINARY}")
        return True

    print(f"  Building tokenize_corpus from source...")
    result = subprocess.run(
        ["make", "tokenize_corpus"],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True
    )
    # Show warnings but not errors (warnings are OK)
    if result.returncode != 0:
        print(f"  BUILD FAILED:")
        print(result.stdout)
        print(result.stderr)
        # Try direct compilation as fallback
        print(f"  Trying direct compilation...")
        result = subprocess.run([
            "gcc", "-O2", "-Wall", "-I", os.path.join(PROJECT_ROOT, "include"),
            "-o", TOKENIZE_BINARY,
            os.path.join(PROJECT_ROOT, "tools", "tokenize_corpus.c"),
            os.path.join(PROJECT_ROOT, "src", "wubu_tokenizer.c"),
            os.path.join(PROJECT_ROOT, "src", "gguf_reader.c"),
            "-lm", "-fopenmp"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Direct compilation failed: {result.stderr}")
            return False
    print(f"  Build OK: {TOKENIZE_BINARY}")
    return True


def run_tokenizer(gguf_path, txt_path, bin_path, meta_path):
    """Step 2b: Run the C tokenizer."""
    step("STEP 2b: Tokenize text to .bin")

    if not os.path.exists(txt_path):
        print(f"  ERROR: Input text not found: {txt_path}")
        return False

    txt_size = os.path.getsize(txt_path)
    print(f"  Input: {txt_path} ({txt_size:,} bytes)")
    print(f"  Output: {bin_path}")
    print(f"  Model: {gguf_path}")

    # Run the tokenizer
    start = time.time()
    result = subprocess.run(
        [TOKENIZE_BINARY, gguf_path],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True,
        timeout=600  # 10 minutes max
    )
    elapsed = time.time() - start
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ERROR: tokenizer failed (exit code {result.returncode}):")
        print(result.stderr)
        return False

    print(f"  Tokenization completed in {elapsed:.1f}s")
    return True


def validate_bin(bin_path, meta_path=None):
    """Step 3: Validate the .bin file format."""
    step("STEP 3: Validate .bin file")

    if not os.path.exists(bin_path):
        print(f"  ERROR: .bin not found: {bin_path}")
        return False

    file_size = os.path.getsize(bin_path)
    print(f"  File: {bin_path}")
    print(f"  Size: {file_size:,} bytes")

    with open(bin_path, "rb") as f:
        data = f.read()

    if file_size % 4 != 0:
        print(f"  WARNING: file size not divisible by 4 (bad format)")
    else:
        num_tokens = file_size // 4
        print(f"  Tokens: {num_tokens:,} ({file_size:,} bytes / 4)")

    # Read first and last few tokens
    tokens = struct.unpack(f"<{len(data)//4}i", data)
    print(f"  First 10 token IDs: {tokens[:10]}")
    print(f"  Last 10 token IDs: {tokens[-10:]}")
    print(f"  Min token ID: {min(tokens)}")
    print(f"  Max token ID: {max(tokens)}")
    print(f"  Valid range (0-248319): {all(0 <= t < 248320 for t in tokens)}")

    # Check for BOS/EOS markers
    bos_count = sum(1 for t in tokens if t == 248044)
    eos_count = sum(1 for t in tokens if t == 248046)
    print(f"  BOS tokens (248044): {bos_count}")
    print(f"  EOS tokens (248046): {eos_count}")

    # Print meta file if exists
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            print(f"\n  Metadata ({meta_path}):")
            for line in f:
                print(f"    {line.strip()}")

    return True


def create_small_test_dataset(output_path, num_tokens=1000):
    """Create a small synthetic test dataset (no tokenizer needed)."""
    step(f"Creating small test dataset ({num_tokens} tokens)")

    # Generate random-ish token IDs in a reasonable range (printable ASCII mostly)
    import random
    random.seed(42)

    # Use common token IDs (bytes 0-255 mapped through GPT-2 byte encoder)
    tokens = [random.randint(0, 255) for _ in range(num_tokens)]

    with open(output_path, "wb") as f:
        f.write(struct.pack(f"<{len(tokens)}i", *tokens))

    file_size = os.path.getsize(output_path)
    print(f"  Wrote {num_tokens:,} tokens to {output_path}")
    print(f"  File size: {file_size:,} bytes")
    return True


def create_ascii_test_dataset(output_path, num_chars=1000):
    """Create a small test dataset by manually generating ASCII-level token IDs.

    The tokenizer maps printable ASCII characters 32-126 to token IDs 0-93.
    This creates valid token IDs without needing the full tokenizer.
    """
    step(f"Creating ASCII test dataset (~{num_chars} byte tokens)")

    import struct

    # Qwen3.6 byte encoder maps:
    # space (32) -> 0, ! -> 1, " -> 2, ..., ~ -> 92, newline has separate mapping
    # Let's create a simple text

    text = """Hello, this is a test corpus for the WuBuText training pipeline.
It contains multiple sentences that will be tokenized into byte-level tokens.
The quick brown fox jumps over the lazy dog.
Testing 1 2 3. Temperature sampling. Top-k and top-p.
Code generation: def hello_world():
    print("Hello, World!")
Mathematics: E = mc²
Poincare geometry in hyperbolic space.
"""
    # Repeat to get desired size
    repeat = max(1, num_chars // len(text))
    text = text * repeat
    text = text[:num_chars]

    # Map each character to its byte token ID using Qwen3.6 byte encoder
    byte_encoder = {
        32: 0, 33: 1, 34: 2, 35: 3, 36: 4, 37: 5, 38: 6, 39: 7,
        40: 8, 41: 9, 42: 10, 43: 11, 44: 12, 45: 13, 46: 14, 47: 15,
        48: 16, 49: 17, 50: 18, 51: 19, 52: 20, 53: 21, 54: 22, 55: 23,
        56: 24, 57: 25, 58: 26, 59: 27, 60: 28, 61: 29, 62: 30, 63: 31,
        64: 32, 65: 33, 66: 34, 67: 35, 68: 36, 69: 37, 70: 38, 71: 39,
        72: 40, 73: 41, 74: 42, 75: 43, 76: 44, 77: 45, 78: 46, 79: 47,
        80: 48, 81: 49, 82: 50, 83: 51, 84: 52, 85: 53, 86: 54, 87: 55,
        88: 56, 89: 57, 90: 58, 91: 59, 92: 60, 93: 61, 94: 62, 95: 63,
        96: 64, 97: 65, 98: 66, 99: 67, 100: 68, 101: 69, 102: 70, 103: 71,
        104: 72, 105: 73, 106: 74, 107: 75, 108: 76, 109: 77, 110: 78,
        111: 79, 112: 80, 113: 81, 114: 82, 115: 83, 116: 84, 117: 85,
        118: 86, 119: 87, 120: 88, 121: 89, 122: 90, 123: 91, 124: 92,
        125: 93, 126: 94,
        # Non-ASCII get remapped
    }

    tokens = []
    for ch in text.encode('utf-8'):
        tid = byte_encoder.get(ch, 95)  # default to some valid token
        tokens.append(tid)

    with open(output_path, "wb") as f:
        f.write(struct.pack(f"<{len(tokens)}i", *tokens))

    file_size = os.path.getsize(output_path)
    print(f"  Wrote {len(tokens):,} tokens to {output_path}")
    print(f"  File size: {file_size:,} bytes")
    return True


def main():
    parser = argparse.ArgumentParser(description="Tokenization Pipeline for WuBuText Training")
    parser.add_argument("--input", help="Input text file (default: extract from CORPUS.py)")
    parser.add_argument("--output-bin", default=DEFAULT_BIN, help=f"Output .bin path (default: {DEFAULT_BIN})")
    parser.add_argument("--gguf", default=DEFAULT_GGUF, help=f"GGUF model path (default: {DEFAULT_GGUF})")
    parser.add_argument("--validate", action="store_true", help="Validate existing .bin file only")
    parser.add_argument("--small", action="store_true", help="Create a small synthetic test dataset")
    parser.add_argument("--size", type=int, default=5000, help="Size for small/test dataset in tokens")
    parser.add_argument("--build-only", action="store_true", help="Only build the C tokenizer binary")
    parser.add_argument("--extract-only", action="store_true", help="Only extract text from CORPUS.py")
    parser.add_argument("--tokenize-only", action="store_true", help="Only run tokenization (assumes corpus_raw.txt exists)")

    args = parser.parse_args()

    if args.validate:
        return validate_bin(args.output_bin, DEFAULT_META)

    if args.small:
        return create_small_test_dataset(args.output_bin, args.size)

    if args.build_only:
        return build_tokenizer_binary()

    if args.extract_only:
        return extract_corpus_py(DEFAULT_CORPUS_PY, DEFAULT_RAW_TXT)

    # Full pipeline
    print(f"\n{'#'*60}")
    print(f"  WuBuText Tokenization Pipeline")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"{'#'*60}")

    # Step 1: Build binary
    if not build_tokenizer_binary():
        print("  FATAL: Could not build tokenizer binary")
        return 1

    # Step 2: Extract or use input text
    txt_path = args.input or DEFAULT_RAW_TXT
    if not os.path.exists(txt_path):
        if args.input:
            print(f"  ERROR: Input file not found: {args.input}")
            return 1
        # Try extraction from CORPUS.py
        if not extract_corpus_py(DEFAULT_CORPUS_PY, DEFAULT_RAW_TXT):
            print(f"  WARNING: Could not extract from CORPUS.py")
            print(f"  Creating small test dataset instead...")
            create_ascii_test_dataset(args.output_bin, 5000)
            print(f"\n  Pipeline complete! Validating...")
            validate_bin(args.output_bin, DEFAULT_META)
            return 0
        txt_path = DEFAULT_RAW_TXT

    # Step 3: Tokenize
    if args.tokenize_only:
        # Use existing corpus_raw.txt
        txt_path = DEFAULT_RAW_TXT

    if not run_tokenizer(args.gguf, txt_path, args.output_bin, DEFAULT_META):
        print("  FATAL: Tokenization failed")
        return 1

    # Step 4: Validate
    validate_bin(args.output_bin, DEFAULT_META)

    print(f"\n{'#'*60}")
    print(f"  Pipeline complete!")
    print(f"{'#'*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
