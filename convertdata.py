import numpy as np
import pandas as pd
import os

def preprocess_text_to_bytes(csv_path: str, npy_path: str):
    """
    Converts a CSV file to a byte-level NPY file for training.

    Args:
        csv_path (str): Path to the CSV file containing text data.
        npy_path (str): Path to save the resulting NPY file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Processing {csv_path}...")
    # Read the CSV file, ensuring we only load the 'text' column
    df = pd.read_csv(csv_path, header=0, names=['index', 'text'], usecols=['text'], dtype=str)

    # Drop NaN and empty rows
    df['text'] = df['text'].fillna('').str.strip()
    df = df[df['text'] != '']

    # Concatenate all rows into a single text string
    combined_text = " ".join(df['text'].values)

    # Encode the text into bytes
    byte_data = np.array(list(combined_text.encode('utf-8')), dtype=np.uint8)

    # Save the byte array as an NPY file
    np.save(npy_path, byte_data)
    print(f"Saved byte data to {npy_path} (length: {len(byte_data)} bytes)")

# Convert training and validation datasets
preprocess_text_to_bytes("C:/projects/bytropix/data/wikitext_train.csv", "C:/projects/bytropix/data/wikitext_train.npy")
preprocess_text_to_bytes("C:/projects/bytropix/data/wikitext_test.csv", "C:/projects/bytropix/data/wikitext_val.npy")
