"""Test IQ2_XXS dequant for expert weights by dequantizing first block."""
import numpy as np
import struct
import sys
sys.path.insert(0, '/home/wubu/bytropix/tools')

# Read first 66 bytes of L0 gate_exps tensor from the GGUF data blob
# The model is 10GB+ so let me compute the offset from the tensor info
# Instead, let me use our existing C test

import subprocess
result = subprocess.run([
    '/home/wubu/bytropix/tools/gguf_read_type',
    '/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf',
    'blk.0.ffn_gate_exps.weight'
], capture_output=True, text=True, timeout=30)
print("gguf_read_type stdout:", result.stdout[:500])
print("gguf_read_type stderr:", result.stderr[:500])
print("gguf_read_type rc:", result.returncode)
