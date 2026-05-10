import os
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
import time

# Settings
NUM_IMAGES = 1000
WIDTH = 512
HEIGHT = 512
OUTPUT_DIR = "fast_dataset"

def generate_one_image(index):
    np.random.seed(index)
    
    # 1. Create coordinate grid
    x = np.linspace(0, 4 * np.pi, WIDTH)
    y = np.linspace(0, 4 * np.pi, HEIGHT)
    X, Y = np.meshgrid(x, y)
    
    # 2. Random Frequency Components (Plasma Effect)
    # This creates "topological" features (hills, valleys, curves)
    f1 = np.random.uniform(0.5, 3.0)
    f2 = np.random.uniform(0.5, 3.0)
    p1 = np.random.uniform(0, 2*np.pi)
    
    # Create wave patterns
    wave1 = np.sin(X * f1 + p1)
    wave2 = np.cos(Y * f2 + X * 0.5)
    wave3 = np.sin(np.sqrt(X**2 + Y**2) * np.random.uniform(1.0, 2.0))
    
    # Combine into channels
    r = (np.sin(wave1 + wave2) * 127.5 + 127.5).astype(np.uint8)
    g = (np.cos(wave2 + wave3) * 127.5 + 127.5).astype(np.uint8)
    b = (np.sin(wave1 + wave3 + np.pi) * 127.5 + 127.5).astype(np.uint8)
    
    # 3. Add Geometric Shapes (Hard edges for the decoder to learn)
    if np.random.rand() > 0.5:
        # Random Circle
        cx, cy = np.random.randint(100, 412, 2)
        rad = np.random.randint(50, 150)
        mask = ((X - cx/40)**2 + (Y - cy/40)**2) < (rad/40)**2
        r[mask] = 255 - r[mask]
        g[mask] = 255 - g[mask]
    
    # Stack
    img_array = np.stack([r, g, b], axis=-1)
    
    # Save
    img = Image.fromarray(img_array)
    img.save(f"{OUTPUT_DIR}/img_{index:04d}.jpg", quality=90)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"--- Generating {NUM_IMAGES} synthetic topological images ({WIDTH}x{HEIGHT})... ---")
    start = time.time()
    
    # Use all CPU cores to blast this out
    with Pool(cpu_count()) as pool:
        pool.map(generate_one_image, range(NUM_IMAGES))
        
    end = time.time()
    print(f"--- Done! Generated {NUM_IMAGES} images in {end - start:.2f} seconds. ---")
    print(f"--- Saved to: {os.path.abspath(OUTPUT_DIR)} ---")

if __name__ == "__main__":
    main()