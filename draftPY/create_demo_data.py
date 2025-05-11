# create_demo_data.py
import numpy as np
import os

pattern_str = "DEMO" # Simple 4-byte pattern
pattern_bytes = np.array([ord(c) for c in pattern_str], dtype=np.uint8)

# Repeat the pattern to make a reasonably sized dataset
# e.g., 4096 repetitions = 16384 bytes.
# This will give plenty of (16384 - 64 + 1) = 16321 possible sequences of length 64
num_repetitions = 4096
data_array = np.tile(pattern_bytes, num_repetitions) # Total size = 4 * 4096 = 16384 bytes

# --- Relative path setup (similar to your batch script) ---
# Assuming this script is in YourMainProjectFolder\draftPY\
# And data folder is YourMainProjectFolder\data\
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True) # Ensure data directory exists

file_path = os.path.join(data_dir, "functional_demo_data.npy")

np.save(file_path, data_array)
print(f"Saved functional demo data to: {file_path}")
print(f"Data shape: {data_array.shape}, Data snippet: {data_array[:20]}")
# Expected snippet: [68 69 77 79 68 69 77 79 ...] (ASCII for D E M O D E M O)