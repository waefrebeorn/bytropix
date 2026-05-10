# coco_preprocessor.py
import json
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm

def create_coco_text_files(image_dir: str, json_path: str, output_dir: str):
    """
    Parses the COCO annotations JSON to create a clean directory of aligned
    image and text file pairs.

    Args:
        image_dir (str): Path to the directory containing COCO images (e.g., 'train2017').
        json_path (str): Path to the COCO captions annotation JSON file.
        output_dir (str): The directory where the aligned images and .txt files will be saved.
    """
    print("--- 🚀 Starting COCO Data Preprocessing ---")
    
    # Setup paths
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    
    # Create the output directory, clearing it if it exists to ensure a clean slate
    if output_path.exists():
        print(f"⚠️ Output directory '{output_path}' already exists. Clearing it first.")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    print(f"--- 📖 Loading annotations from '{json_path}'... ---")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # --- Create an efficient mapping from image ID to its filename ---
    print("--- 🗺️ Creating image ID to filename map... ---")
    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    # --- Process annotations and create aligned files ---
    print(f"--- ✍️ Processing {len(data['annotations'])} annotations to create aligned pairs... ---")
    
    processed_image_ids = set()
    num_pairs_created = 0
    
    for annotation in tqdm(data['annotations'], desc="Creating image-text pairs"):
        image_id = annotation['image_id']
        caption = annotation['caption'].strip()
        
        # We only want one caption per image to maintain a 1:1 ratio
        if image_id in processed_image_ids:
            continue
            
        if image_id not in image_id_to_filename:
            print(f"Warning: Annotation found for image ID {image_id}, but no corresponding image file entry.")
            continue
            
        filename = image_id_to_filename[image_id]
        source_image_path = image_path / filename
        
        if not source_image_path.exists():
            # This is rare for COCO but good practice to check
            print(f"Warning: Image file '{filename}' not found at '{source_image_path}'. Skipping.")
            continue
            
        # Define destination paths
        dest_image_path = output_path / filename
        dest_text_path = output_path / Path(filename).with_suffix('.txt')
        
        # 1. Copy the image file to the new directory
        shutil.copy(source_image_path, dest_image_path)
        
        # 2. Write the caption to the corresponding .txt file
        with open(dest_text_path, 'w', encoding='utf-8') as f:
            f.write(caption)
            
        processed_image_ids.add(image_id)
        num_pairs_created += 1

    print("\n--- 🎉 COCO Preprocessing Complete! ---")
    print(f"✅ Created {num_pairs_created} perfectly aligned image-text pairs in:")
    print(f"[green]{output_path.resolve()}[/green]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare COCO dataset for Phase 3 training.")
    parser.add_argument('--image-dir', type=str, required=True, help="Path to the COCO image directory (e.g., 'path/to/train2017').")
    parser.add_argument('--json-path', type=str, required=True, help="Path to the COCO captions JSON file (e.g., 'path/to/annotations/captions_train2017.json').")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save the new, aligned dataset.")
    args = parser.parse_args()
    
    create_coco_text_files(args.image_dir, args.json_path, args.output_dir)