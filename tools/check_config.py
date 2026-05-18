"""Read GGUF model config key-value pairs to verify hardcoded constants."""
import sys
sys.path.insert(0, '/home/wubu/bytropix')
from src.gguf_reader import *

# Open the model and read KV pairs
import subprocess
result = subprocess.run([
    'python3', '-c', '''
from src import gguf_reader
ctx = gguf_reader.gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")
if ctx:
    print("KV count:", ctx.n_kv)
    for i in range(min(ctx.n_kv, 60)):
        key = gguf_reader.gguf_key(ctx, i)
        val_type = gguf_reader.gguf_value_type(ctx, i)
        print(f"  {i}: {key} (type={val_type})")
    gguf_reader.gguf_close(ctx)
'''
], capture_output=True, text=True, timeout=30)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print(result.stderr[:500] if result.stderr else '')
