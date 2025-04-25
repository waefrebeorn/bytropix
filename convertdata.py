import numpy as np
import pandas as pd
import os
import json
import requests
from datasets import load_dataset
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_and_preprocess_hh_rlhf(output_dir: str = "C:/projects/bytropix/data"):
    """
    Downloads and preprocesses the Anthropic HH-RLHF dataset for Bytropix training.
    
    Args:
        output_dir (str): Directory to save the processed NPY files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths (same names as the original script for compatibility)
    train_npy_path = os.path.join(output_dir, "Anthropic_HH-RLHF_train.npy")
    val_npy_path = os.path.join(output_dir, "Anthropic_HH-RLHF_val.npy")
    
    logger.info("Loading Anthropic HH-RLHF dataset...")
    
    # Load all subsets of the dataset
    subdatasets = ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"]
    all_text = []
    val_text = []
    
    for subset in subdatasets:
        try:
            # Load dataset using the Hugging Face datasets library
            logger.info(f"Loading subset: {subset}")
            dataset = load_dataset("Anthropic/hh-rlhf", data_dir=subset)
            
            # Process each split (train/test) if available
            for split in dataset:
                data = dataset[split]
                
                # Determine if this should go to train or validation
                # Use 95% for training, 5% for validation
                split_idx = int(len(data) * 0.95)
                
                for i, item in enumerate(tqdm(data, desc=f"Processing {subset}/{split}")):
                    # Extract both chosen and rejected texts
                    chosen_text = item.get("chosen", "")
                    rejected_text = item.get("rejected", "")
                    
                    # Add to appropriate set
                    if i < split_idx:
                        if chosen_text:
                            all_text.append(chosen_text)
                        if rejected_text:
                            all_text.append(rejected_text)
                    else:
                        if chosen_text:
                            val_text.append(chosen_text)
                        if rejected_text:
                            val_text.append(rejected_text)
        
        except Exception as e:
            logger.error(f"Error processing subset {subset}: {str(e)}")
            continue
    
    # Convert train data to bytes
    logger.info(f"Converting training data to byte format ({len(all_text)} texts)")
    combined_train_text = "\n\n".join(all_text)
    train_byte_data = np.array(list(combined_train_text.encode('utf-8')), dtype=np.uint8)
    
    # Convert validation data to bytes
    logger.info(f"Converting validation data to byte format ({len(val_text)} texts)")
    combined_val_text = "\n\n".join(val_text)
    val_byte_data = np.array(list(combined_val_text.encode('utf-8')), dtype=np.uint8)
    
    # Save the byte arrays
    np.save(train_npy_path, train_byte_data)
    logger.info(f"Saved training byte data to {train_npy_path} (length: {len(train_byte_data)} bytes)")
    
    np.save(val_npy_path, val_byte_data)
    logger.info(f"Saved validation byte data to {val_npy_path} (length: {len(val_byte_data)} bytes)")
    
    return train_npy_path, val_npy_path

# Additional function to handle the red-teaming data which has a different format
def add_red_team_data(train_npy_path: str, val_npy_path: str):
    """
    Adds red-team data to the existing NPY files.
    
    Args:
        train_npy_path (str): Path to the training NPY file
        val_npy_path (str): Path to the validation NPY file
    """
    try:
        logger.info("Loading red-team dataset...")
        red_team_dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
        
        red_team_texts = []
        for split in red_team_dataset:
            data = red_team_dataset[split]
            
            for item in tqdm(data, desc=f"Processing red-team/{split}"):
                full_transcript = item.get("transcript", "")
                if full_transcript:
                    red_team_texts.append(full_transcript)
        
        # Split 95/5 between train and validation
        split_idx = int(len(red_team_texts) * 0.95)
        red_team_train = red_team_texts[:split_idx]
        red_team_val = red_team_texts[split_idx:]
        
        # Load existing byte data
        train_bytes = np.load(train_npy_path)
        val_bytes = np.load(val_npy_path)
        
        # Convert and append red team data to training
        red_team_train_text = "\n\n".join(red_team_train)
        red_team_train_bytes = np.array(list(red_team_train_text.encode('utf-8')), dtype=np.uint8)
        combined_train = np.concatenate([train_bytes, red_team_train_bytes])
        
        # Convert and append red team data to validation
        red_team_val_text = "\n\n".join(red_team_val)
        red_team_val_bytes = np.array(list(red_team_val_text.encode('utf-8')), dtype=np.uint8)
        combined_val = np.concatenate([val_bytes, red_team_val_bytes])
        
        # Save combined data
        np.save(train_npy_path, combined_train)
        logger.info(f"Added red-team data to training file (new length: {len(combined_train)} bytes)")
        
        np.save(val_npy_path, combined_val)
        logger.info(f"Added red-team data to validation file (new length: {len(combined_val)} bytes)")
        
    except Exception as e:
        logger.error(f"Error processing red-team data: {str(e)}")

if __name__ == "__main__":
    # Step 1: Process the main helpfulness/harmlessness datasets
    train_path, val_path = download_and_preprocess_hh_rlhf()
    
    # Step 2: Add the red-team data which has a different format
    add_red_team_data(train_path, val_path)
    
    logger.info("All processing complete!")