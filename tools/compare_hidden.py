# Quick check: does our embedding table match llama.cpp's?
# Run a single token through both, check first-layer hidden state

import struct, math, subprocess, os

# Path to our embedded dump
our_bin = '/tmp/our_hidden.bin'
with open(our_bin, 'rb') as f:
    our = list(struct.unpack('2048f', f.read(8192)))

# Ref final hidden (after all layers)
ref_bin = '/tmp/ref_last_hidden.bin'
with open(ref_bin, 'rb') as f:
    ref = list(struct.unpack('2048f', f.read(8192)))

# Check embedding stats  
print("=== EMBEDDING CHECK ===")
print(f"Our first layer input (BOS token 248044):")
print(f"  mean={sum(our)/2048:.4f}, max={max(our):.4f}, min={min(our):.4f}")

print(f"\nLLaMA.cpp final hidden (after ALL 40 layers):")
print(f"  mean={sum(ref)/2048:.4f}, max={max(ref):.4f}, min={min(ref):.4f}")

# The embedding table should be identical since we load it from the same GGUF.
# Let me check if the token_embd.weight in the GGUF is in a different layout.
# In llama.cpp, the embedding weight is stored as [n_vocab, n_embd]
# and used directly. We do the same.

# Let me check: what if our model normalizes the embedding differently?
print("\n=== LAYER 0 PRE-ATTN NORM ===")
# RMSNorm on embedding should match if it's using same weight

# Let me also check if the issue is that our code uses `--temp 0` sampling 
# vs the ref program that uses whatever default temp

# Actually, the key difference: our code adds BOS token + prompt, while
# ref program tokenizes "Hello" WITHOUT adding BOS (add_bos=false)
# But we both have 1 token... Wait let me check:
# Our output: <|endoftext|>Helloeschi - means BOS=248044 was prepended
# So our model processes BOS + "Hello" = 2 tokens
# While ref processes just "Hello" = 1 token

# THIS IS THE BUG! Different number of input tokens!

print("\nCRITICAL: Different input tokenization!")
print("  Our code: BOS(248044) + 'Hello' = 2 tokens, hidden from LAST token")
print("  Ref code: 'Hello' = 1 token (no BOS), hidden from ONLY token")
print()
print("The hidden states are from DIFFERENT positions!")
print("Need to make both process exactly the same input.")
