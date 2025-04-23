import numpy as np
import os
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_poem_dataset(output_dir: str = "C:/projects/bytropix/data/poems"):
    """
    Creates small poem datasets for training and validation in the NPY format
    compatible with the ByteTropix model.
    
    Args:
        output_dir (str): Directory to save the processed NPY files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths
    train_npy_path = os.path.join(output_dir, "poems_train.npy")
    val_npy_path = os.path.join(output_dir, "poems_val.npy")
    
    # Sample placeholders for poem structures
    # (We're using structural patterns instead of actual poems)
    poem_structures = [
        # Basic structures with varying line patterns and lengths
        "Title\n\nStanza 1, line 1\nStanza 1, line 2\nStanza 1, line 3\nStanza 1, line 4\n\nStanza 2, line 1\nStanza 2, line 2\nStanza 2, line 3\nStanza 2, line 4",
        "Title\n\nLine 1\nLine 2\nLine 3\n\nLine 4\nLine 5\nLine 6",
        "Title\n\nCouplet 1, line 1\nCouplet 1, line 2\n\nCouplet 2, line 1\nCouplet 2, line 2\n\nCouplet 3, line 1\nCouplet 3, line 2",
        "Title\n\nQuatrain 1, line 1\nQuatrain 1, line 2\nQuatrain 1, line 3\nQuatrain 1, line 4",
        
        # Different formats
        "Title\n\nFree verse line 1\nFree verse line 2\nFree verse line 3\nFree verse line 4\nFree verse line 5",
        "Title\n\nHaiku line 1\nHaiku line 2\nHaiku line 3",
        "Title\n\nSonnet line 1\nSonnet line 2\nSonnet line 3\nSonnet line 4\nSonnet line 5\nSonnet line 6\nSonnet line 7\nSonnet line 8\nSonnet line 9\nSonnet line 10\nSonnet line 11\nSonnet line 12\nSonnet line 13\nSonnet line 14",
        
        # Different indentation patterns
        "Title\n\nLine 1\n    Indented line 2\nLine 3\n    Indented line 4",
        "Title\n\nLine 1\n  Line 2\n    Line 3\n      Line 4\n    Line 5\n  Line 6\nLine 7",
        
        # Patterns with different punctuation structures
        "Title\n\nLine 1?\nLine 2.\nLine 3!\nLine 4;",
        "Title\n\nLine 1... Line 2... Line 3...\nLine 4—Line 5—Line 6",
    ]
    
    # Generate variations for training set (repeat structures with variations)
    train_poems = []
    
    for i in range(10):  # Generate 10 training samples
        base_structure = poem_structures[i % len(poem_structures)]
        # Create variations by adding IDs and minor alterations
        variation = f"Poem #{i+1}\n" + base_structure.replace("Title", f"Poem Title {i+1}")
        train_poems.append(variation)
    
    # Generate smaller validation set
    val_poems = []
    
    for i in range(20):  # Generate 20 validation samples
        base_structure = poem_structures[i % len(poem_structures)]
        # Create variations by adding IDs and minor alterations
        variation = f"Val Poem #{i+1}\n" + base_structure.replace("Title", f"Validation Poem Title {i+1}")
        val_poems.append(variation)
    
    # Convert train data to bytes
    logger.info(f"Converting training data to byte format ({len(train_poems)} poems)")
    combined_train_text = "\n\n".join(train_poems)
    train_byte_data = np.array(list(combined_train_text.encode('utf-8')), dtype=np.uint8)
    
    # Convert validation data to bytes
    logger.info(f"Converting validation data to byte format ({len(val_poems)} poems)")
    combined_val_text = "\n\n".join(val_poems)
    val_byte_data = np.array(list(combined_val_text.encode('utf-8')), dtype=np.uint8)
    
    # Save the byte arrays
    np.save(train_npy_path, train_byte_data)
    logger.info(f"Saved training byte data to {train_npy_path} (length: {len(train_byte_data)} bytes)")
    
    np.save(val_npy_path, val_byte_data)
    logger.info(f"Saved validation byte data to {val_npy_path} (length: {len(val_byte_data)} bytes)")
    
    return train_npy_path, val_npy_path

if __name__ == "__main__":
    # Create the poem datasets
    train_path, val_path = create_poem_dataset()
    
    logger.info(f"Created poem datasets:")
    logger.info(f"Training: {train_path}")
    logger.info(f"Validation: {val_path}")
    
    # Print instructions for using these datasets
    logger.info("\nTo use these datasets with your model, run with:")
    logger.info("python integrated_hyper_hakmem_model.py \\")
    logger.info("    --data_path=C:/projects/bytropix/data/poems/poems_train.npy \\")
    logger.info("    --val_data_path=C:/projects/bytropix/data/poems/poems_val.npy \\")
    logger.info("    --batch_size=2 --grad_accum_steps=4 --learning_rate=5e-05 \\")
    logger.info("    --epochs=3 --data_fraction=1.0 --max_grad_norm=0.5 \\")
    logger.info("    --save_interval=50 --log_interval=5")