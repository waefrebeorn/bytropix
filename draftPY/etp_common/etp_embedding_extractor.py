import logging
import os
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_ds_r1_sentence_embeddings(
    texts: List[str],
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32
) -> List[np.ndarray]:
    """
    Extracts sentence embeddings from a list of texts using a DeepSeek model.

    Args:
        texts: A list of sentences (strings).
        model_name: The name of the DeepSeek model to use.
        device: The device to run the model on ("cuda" or "cpu").
        batch_size: The batch size for processing texts.

    Returns:
        A list of NumPy arrays, where each array is a sentence embedding.
    """
    logging.info(f"Loading model and tokenizer: {model_name}")
    try:
        # For dummy model, sometimes from_pretrained might fail if offline or model is too minimal.
        # Adding a check for "prajjwal1/bert-tiny" to potentially use a different config if needed.
        if "prajjwal1/bert-tiny" in model_name and device == "cpu": # common scenario for dummy
             # Ensure transformers version is compatible or add specific handling if errors occur.
             pass

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(device)
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        logging.error(f"Error loading model or tokenizer '{model_name}': {e}")
        # Specific error for bert-tiny if it occurs often in test environments
        if "prajjwal1/bert-tiny" in model_name and isinstance(e, OSError):
            logging.error("This might be due to the model not being available locally or network issues.")
            logging.error("Ensure 'prajjwal1/bert-tiny' is cached or accessible.")
        raise

    all_embeddings: List[np.ndarray] = []
    logging.info(f"Processing {len(texts)} texts in batches of {batch_size} on {device}.")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # logging.info(f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

        try:
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128) # Max length for dummy
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
                sum_embeddings = torch.sum(last_hidden_states * expanded_mask, 1)
                sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9) 
                mean_pooled_embeddings = sum_embeddings / sum_mask

            all_embeddings.extend([emb.cpu().numpy() for emb in mean_pooled_embeddings])
        except Exception as e:
            logging.error(f"Error processing batch {i // batch_size + 1}: {e}")
            raise
    logging.info(f"Successfully extracted embeddings for {len(all_embeddings)} sentences from model {model_name}.")
    return all_embeddings


def save_embeddings(
    embeddings: List[np.ndarray],
    output_path: str,
    output_format: str = "numpy_list"
):
    """
    Saves a list of embeddings to a file.
    """
    logging.info(f"Saving {len(embeddings)} embeddings to {output_path} in {output_format} format.")
    if output_format == "numpy_list":
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logging.info(f"Created directory: {output_dir}")
            np.savez_compressed(output_path, *embeddings) # Saves as arr_0, arr_1, ...
            logging.info(f"Embeddings successfully saved to {output_path}.")
        except Exception as e:
            logging.error(f"Error saving embeddings with np.savez_compressed: {e}")
            raise
    elif output_format == "hdf5":
        try:
            import h5py
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with h5py.File(output_path, 'w') as hf:
                for i, emb in enumerate(embeddings):
                    hf.create_dataset(f"embedding_{i}", data=emb)
            logging.info(f"Embeddings successfully saved to {output_path} in HDF5 format.")
        except ImportError:
            logging.error("h5py library is not installed. Cannot save in HDF5 format.")
            raise
        except Exception as e:
            logging.error(f"Error saving embeddings in HDF5 format: {e}")
            raise
    else:
        logging.error(f"Unsupported output format: {output_format}")
        raise ValueError(f"Unsupported output format: {output_format}")

if __name__ == '__main__':
    logging.info("Starting dummy embedding file generation.")

    # Configuration for dummy file generation
    use_dummy_model_main = True 
    dummy_model_name_main = "prajjwal1/bert-tiny" # Small model for fast dummy embedding generation
    # Ensure this model is available in the environment or Hugging Face cache.
    # If running in a restricted environment without internet, this model needs to be pre-downloaded.

    # Determine device
    main_device = "cuda" if torch.cuda.is_available() else "cpu"
    # For bert-tiny, CPU is fine and avoids CUDA dependencies for this simple generation task.
    # If a GPU is available, it might be faster, but not strictly necessary for bert-tiny.
    if "bert-tiny" in dummy_model_name_main:
        logging.info("Using bert-tiny, forcing device to CPU for dummy generation for wider compatibility.")
        main_device = "cpu" 

    # Define dummy sentences
    texts_corpus_A = [f"This is sentence A number {i} from the primary collection." for i in range(70)]
    texts_corpus_B = [f"Sentence B sample {i}, quite different from A's documents." for i in range(60)]

    # Output paths
    output_dir_main = "draftPY" # Save directly in draftPY
    os.makedirs(output_dir_main, exist_ok=True) # Ensure directory exists

    path_A = os.path.join(output_dir_main, "dummy_corpus_A_embeddings.npz")
    path_B = os.path.join(output_dir_main, "dummy_corpus_B_embeddings.npz")

    # Generate and save embeddings for Corpus A
    logging.info(f"Generating embeddings for Corpus A (70 sentences) using {dummy_model_name_main} on {main_device}.")
    try:
        embeddings_A = extract_ds_r1_sentence_embeddings(
            texts_corpus_A,
            model_name=dummy_model_name_main,
            device=main_device,
            batch_size=16 # Smaller batch size for tiny model if memory is constrained
        )
        save_embeddings(embeddings_A, path_A, output_format="numpy_list")
        logging.info(f"Dummy Corpus A embeddings saved to {path_A}")

        # Verify saved file for A
        loaded_A = np.load(path_A, allow_pickle=True)
        num_embeddings_A = len(loaded_A.files)
        logging.info(f"Verified {path_A}: Contains {num_embeddings_A} embeddings (expected {len(texts_corpus_A)}).")
        assert num_embeddings_A == len(texts_corpus_A)

    except Exception as e:
        logging.error(f"Failed to generate or save Corpus A embeddings: {e}", exc_info=True)
        # If bert-tiny is the issue, provide a specific hint.
        if "bert-tiny" in dummy_model_name_main and "offline" in str(e).lower() or "not found" in str(e).lower():
            logging.error("Ensure 'prajjwal1/bert-tiny' is downloaded and accessible (e.g., via internet or local cache).")


    # Generate and save embeddings for Corpus B
    logging.info(f"Generating embeddings for Corpus B (60 sentences) using {dummy_model_name_main} on {main_device}.")
    try:
        embeddings_B = extract_ds_r1_sentence_embeddings(
            texts_corpus_B,
            model_name=dummy_model_name_main,
            device=main_device,
            batch_size=16
        )
        save_embeddings(embeddings_B, path_B, output_format="numpy_list")
        logging.info(f"Dummy Corpus B embeddings saved to {path_B}")

        # Verify saved file for B
        loaded_B = np.load(path_B, allow_pickle=True)
        num_embeddings_B = len(loaded_B.files)
        logging.info(f"Verified {path_B}: Contains {num_embeddings_B} embeddings (expected {len(texts_corpus_B)}).")
        assert num_embeddings_B == len(texts_corpus_B)

    except Exception as e:
        logging.error(f"Failed to generate or save Corpus B embeddings: {e}", exc_info=True)
        if "bert-tiny" in dummy_model_name_main and "offline" in str(e).lower() or "not found" in str(e).lower():
            logging.error("Ensure 'prajjwal1/bert-tiny' is downloaded and accessible (e.g., via internet or local cache).")

    logging.info("Dummy embedding file generation process finished.")
