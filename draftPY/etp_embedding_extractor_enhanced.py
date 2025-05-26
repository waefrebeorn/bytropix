import argparse
import json
import logging
import os
import random
import sys
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel, AutoConfig

# --- Global H5PY_AVAILABLE flag (set based on import success) ---
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
# --- End Global H5PY_AVAILABLE flag ---

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ETP_Embedding_Extractor_Enhanced")

# Supported pooling strategies
POOLING_STRATEGIES = ["mean", "cls", "last_token"]

# A small, diverse set of example sentences
DIVERSE_SENTENCE_POOL = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is rapidly transforming various industries.",
    "Hyperbolic geometry offers unique advantages for embedding hierarchical data.",
    "Climate change poses a significant threat to global ecosystems.",
    "The history of philosophy is rich with diverse perspectives on reality.",
    "Exploring distant galaxies reveals the vastness of the universe.",
    "Quantum computing promises to solve problems currently intractable for classical computers.",
    "Shakespeare's plays continue to be performed and studied worldwide.",
    "The principles of economics can explain complex market behaviors.",
    "Learning a new language opens up new cultural understanding.",
    "Machine learning models require large datasets for effective training.",
    "The pursuit of happiness is a fundamental human endeavor.",
    "Renewable energy sources are crucial for a sustainable future.",
    "Genetic engineering holds both promise and peril for humanity.",
    "Understanding the human brain remains one of science's greatest challenges.",
    "The novel tells a compelling story of adventure and self-discovery.",
    "Data visualization helps in understanding complex datasets.",
    "Cybersecurity is increasingly important in our interconnected world.",
    "The impact of social media on society is a subject of ongoing debate.",
    "Developing robust and fair AI systems requires careful ethical consideration.",
    "The sunset painted the sky in vibrant hues of orange and purple.",
    "Ancient civilizations left behind remarkable architectural wonders.",
    "Mathematics is the language of the universe.",
    "Bioluminescence in the deep sea creates an ethereal spectacle.",
    "The journey of a thousand miles begins with a single step.",
    "Culinary arts blend creativity with precise techniques.",
    "Music has the power to evoke deep emotions and memories.",
    "Urban planning aims to create functional and livable cities.",
    "The theory of relativity revolutionized our understanding of space and time.",
    "Conservation efforts are vital for protecting endangered species.",
]


def load_dummy_texts(
    num_texts: int = 100,
    corpus_label: str = "A",
    use_diverse_pool: bool = True,
    min_words: int = 5,
    max_words: int = 30
) -> List[str]:
    """
    Generates a list of dummy sentences.
    Can use a predefined diverse pool or generate repetitive placeholder sentences.
    """
    if use_diverse_pool and DIVERSE_SENTENCE_POOL:
        filtered_pool = [
            s for s in DIVERSE_SENTENCE_POOL
            if min_words <= len(s.split()) <= max_words
        ]
        if not filtered_pool:
            logger.warning("Diverse sentence pool became empty after word count filtering. Falling back to placeholder sentences.")
            return [f"Placeholder sentence {corpus_label}{i} for testing embedding extractor." for i in range(num_texts)]

        if num_texts <= len(filtered_pool):
            return random.sample(filtered_pool, num_texts)
        else:
            base_samples = random.choices(filtered_pool, k=num_texts)
            final_samples = []
            counts = {}
            for sample in base_samples:
                if sample in counts:
                    counts[sample] += 1
                    final_samples.append(f"{sample} (Variation {counts[sample]})")
                else:
                    counts[sample] = 0
                    final_samples.append(sample)
            return final_samples
    else:
        return [f"This is dummy sentence {corpus_label}{i} for testing the embedding extractor." for i in range(num_texts)]

def get_device(device_str: str = "auto") -> torch.device:
    """Determines the appropriate torch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # For Apple Silicon
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)

def get_sentence_embedding(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: str = "mean",
    cls_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    input_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes sentence embedding based on the chosen strategy.
    """
    if strategy == "mean":
        expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * expanded_mask, 1)
        sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    elif strategy == "cls":
        if input_ids is None or cls_token_id is None:
            logger.warning("CLS pooling: input_ids or cls_token_id missing. Using first token.")
            return last_hidden_state[:, 0]
        batch_cls_embeddings = []
        for i in range(input_ids.shape[0]):
            cls_indices = (input_ids[i] == cls_token_id).nonzero(as_tuple=True)[0]
            if len(cls_indices) > 0:
                batch_cls_embeddings.append(last_hidden_state[i, cls_indices[0]])
            else:
                logger.warning(f"CLS token ID {cls_token_id} not found for sample {i}. Using first token.")
                batch_cls_embeddings.append(last_hidden_state[i, 0])
        return torch.stack(batch_cls_embeddings)
    elif strategy == "last_token":
        if input_ids is None or eos_token_id is None:
            logger.warning("Last_token pooling: input_ids or eos_token_id missing. Using last non-padding token via attention_mask.")
            sequence_lengths = torch.sum(attention_mask, dim=1) -1
            return last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), sequence_lengths]
        batch_last_token_embeddings = []
        for i in range(input_ids.shape[0]):
            eos_indices = (input_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                batch_last_token_embeddings.append(last_hidden_state[i, eos_indices[-1]])
            else:
                sequence_length = torch.sum(attention_mask[i]) - 1
                batch_last_token_embeddings.append(last_hidden_state[i, sequence_length])
                logger.debug(f"EOS token ID {eos_token_id} not found for sample {i}. Using token at effective seq_len {sequence_length}.")
        return torch.stack(batch_last_token_embeddings)
    else:
        raise ValueError(f"Unsupported pooling strategy: {strategy}. Choose from {POOLING_STRATEGIES}")


def extract_embeddings(
    texts: List[str],
    model_name_or_path: str,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
    pooling_strategy: str = "mean",
    trust_remote_code: bool = True,
    corpus_label_for_logging: str = "N/A" # Added for contextual logging
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Extracts sentence embeddings from a list of texts using a specified model.
    """
    logger.info(f"Corpus {corpus_label_for_logging}: Loading model and tokenizer: {model_name_or_path} (trust_remote_code={trust_remote_code})")
    model_load_start_time = time.time()

    extraction_metadata = {
        "model_name_or_path": model_name_or_path, "device": str(device),
        "batch_size": batch_size, "max_length": max_length,
        "pooling_strategy": pooling_strategy,
        "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "num_texts_input": len(texts),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "corpus_label": corpus_label_for_logging,
    }
    try:
        import transformers
        extraction_metadata["transformers_version"] = transformers.__version__
        model_transformers_version_from_lib = transformers.__version__
    except ImportError:
        extraction_metadata["transformers_version"] = "Not available"
        model_transformers_version_from_lib = "N/A"

    model = None # Initialize model to None
    try:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )

        model_load_kwargs = {
            "trust_remote_code": trust_remote_code,
        }

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model_preferred_dtype_str_config = getattr(config, "torch_dtype", "float32")
        if model_preferred_dtype_str_config is None: model_preferred_dtype_str_config = "float32"

        model_transformers_version_config = getattr(config, "transformers_version", "N/A")
        extraction_metadata["transformers_version_config_from_model"] = model_transformers_version_config

        logger.info(f"Corpus {corpus_label_for_logging}: Model config suggests torch_dtype='{model_preferred_dtype_str_config}' and was saved with transformers_version='{model_transformers_version_config}'.")
        logger.info(f"Corpus {corpus_label_for_logging}: Currently using transformers library version: '{model_transformers_version_from_lib}'.")

        target_torch_dtype_for_load = None
        target_attn_implementation = "auto"

        is_deepseek_or_qwen_model = "deepseek" in model_name_or_path.lower() or "qwen" in model_name_or_path.lower()

        if is_deepseek_or_qwen_model and device.type == 'cuda':
            logger.info(f"Corpus {corpus_label_for_logging}: Model is DeepSeek/Qwen on CUDA. Will prioritize bfloat16/float16 with Flash Attention 2.")
            target_attn_implementation = "flash_attention_2"
            if torch.cuda.is_bf16_supported():
                logger.info(f"Corpus {corpus_label_for_logging}: Device supports bfloat16. Setting torch_dtype=torch.bfloat16 for Flash Attention 2.")
                target_torch_dtype_for_load = torch.bfloat16
            else:
                logger.info(f"Corpus {corpus_label_for_logging}: Device does not report bfloat16 support. Using torch.float16 for Flash Attention 2.")
                target_torch_dtype_for_load = torch.float16
        else:
            logger.info(f"Corpus {corpus_label_for_logging}: Not a DeepSeek/Qwen model on CUDA, or device is {device.type}. Determining dtype based on config preference and 'auto' attention.")
            if model_preferred_dtype_str_config == "bfloat16":
                if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) or \
                   (device.type == 'cpu' and hasattr(torch, 'bfloat16')):
                    target_torch_dtype_for_load = torch.bfloat16
                else: target_torch_dtype_for_load = torch.float32
            elif model_preferred_dtype_str_config == "float16":
                if device.type == 'cuda': target_torch_dtype_for_load = torch.float16
                else: target_torch_dtype_for_load = torch.float32
            else:
                target_torch_dtype_for_load = torch.float32

        if target_torch_dtype_for_load:
            model_load_kwargs["torch_dtype"] = target_torch_dtype_for_load
        if target_attn_implementation != "auto":
            model_load_kwargs["attn_implementation"] = target_attn_implementation

        logger.info(f"Corpus {corpus_label_for_logging}: Attempting model load with primary kwargs: {model_load_kwargs}")

        try:
            model = AutoModel.from_pretrained(
                model_name_or_path,
                **model_load_kwargs
            )
        except Exception as e_load_primary:
            logger.warning(f"Corpus {corpus_label_for_logging}: Primary loading attempt with kwargs {model_load_kwargs} FAILED: {e_load_primary}", exc_info=False)

            if model_load_kwargs.get("attn_implementation") == "flash_attention_2":
                logger.info(f"Corpus {corpus_label_for_logging}: Fallback 1: Flash Attention 2 failed. Retrying with attn_implementation='eager'.")
                model_load_kwargs_fallback1 = model_load_kwargs.copy()
                model_load_kwargs_fallback1["attn_implementation"] = "eager"
                logger.info(f"Corpus {corpus_label_for_logging}: Fallback 1 (eager) kwargs: {model_load_kwargs_fallback1}")
                try:
                    model = AutoModel.from_pretrained(model_name_or_path, **model_load_kwargs_fallback1)
                except Exception as e_fallback1:
                    logger.warning(f"Corpus {corpus_label_for_logging}: Fallback 1 (eager attention) FAILED: {e_fallback1}", exc_info=False)

            if model is None:
                logger.info(f"Corpus {corpus_label_for_logging}: Fallback 2: Attempting load with minimal arguments (trust_remote_code, default dtype float32, auto attention).")
                minimal_kwargs = {"trust_remote_code": trust_remote_code, "torch_dtype": torch.float32}
                logger.info(f"Corpus {corpus_label_for_logging}: Minimal fallback kwargs: {minimal_kwargs}")
                try:
                    model = AutoModel.from_pretrained(model_name_or_path, **minimal_kwargs)
                except Exception as e_fallback2:
                    logger.error(f"Corpus {corpus_label_for_logging}: All loading attempts FAILED. Last error (minimal fallback): {e_fallback2}", exc_info=True)
                    raise e_fallback2

        model.to(device)
        model.eval()

        model_load_end_time = time.time()
        logger.info(f"Corpus {corpus_label_for_logging}: Model and tokenizer loaded in {model_load_end_time - model_load_start_time:.2f}s.")

        extraction_metadata["model_config_class"] = model.config.__class__.__name__
        extraction_metadata["tokenizer_class"] = tokenizer.__class__.__name__
        extraction_metadata["model_hidden_size"] = getattr(model.config, 'hidden_size', 'N/A')
        extraction_metadata["model_num_layers"] = getattr(model.config, 'num_hidden_layers', 'N/A')
        extraction_metadata["vocab_size"] = getattr(model.config,'vocab_size', getattr(tokenizer, 'vocab_size', 'N/A'))
        extraction_metadata["actual_attn_implementation"] = getattr(model.config, '_attn_implementation', 'N/A')
        extraction_metadata["actual_torch_dtype"] = str(model.dtype)
        logger.info(f"Corpus {corpus_label_for_logging}: Model loaded. Actual attention implementation: {extraction_metadata['actual_attn_implementation']}, "
                    f"Actual dtype: {extraction_metadata['actual_torch_dtype']}")

    except Exception as e:
        logger.error(f"FATAL (Corpus {corpus_label_for_logging}): Error loading model or tokenizer '{model_name_or_path}': {e}", exc_info=True)
        if isinstance(e, TypeError) and "argument of type 'NoneType' is not iterable" in str(e) and "ALL_PARALLEL_STYLES" in str(e):
             logger.error("The persistent TypeError related to 'ALL_PARALLEL_STYLES' suggests a core incompatibility. "
                          "Final recommendation: try using transformers version "
                          f"{model_transformers_version_config if model_transformers_version_config != 'N/A' else '4.44.0 (as per model config)'} "
                          "or investigate model-specific loading on Hugging Face discussions for this transformers version.")
        raise

    cls_token_id_for_pooling = getattr(tokenizer, 'cls_token_id', None)
    eos_token_id_for_pooling = getattr(tokenizer, 'eos_token_id', None)
    if pooling_strategy == "cls" and cls_token_id_for_pooling is None:
        logger.warning(f"Corpus {corpus_label_for_logging}: CLS pooling requested but tokenizer ({tokenizer.__class__.__name__}) has no defined 'cls_token_id'. Will use first token of sequence.")
    if pooling_strategy == "last_token" and eos_token_id_for_pooling is None:
        logger.warning(f"Corpus {corpus_label_for_logging}: Last_token pooling requested but tokenizer ({tokenizer.__class__.__name__}) has no defined 'eos_token_id'. Will use last non-padding token via attention_mask.")

    all_embeddings_list: List[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(f"Corpus {corpus_label_for_logging}: Processing {len(texts)} texts in {total_batches} batches of size {batch_size} on {device}.")

    if total_batches == 0:
        logger.warning(f"Corpus {corpus_label_for_logging}: No texts to process, skipping embedding extraction loop.")
        extraction_metadata["num_embeddings_extracted"] = 0
        extraction_metadata["embedding_dimension"] = "N/A (no texts)"
        return all_embeddings_list, extraction_metadata


    # Determine logging interval for batch progress
    if total_batches < 20:
        log_batch_interval = 1
    elif total_batches < 200:
        log_batch_interval = 10
    elif total_batches < 2000:
        log_batch_interval = 50
    else:
        log_batch_interval = 100

    loop_start_time = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        current_batch_num = i // batch_size + 1
        batch_processing_start_time = time.time()

        try:
            inputs = tokenizer(
                batch_texts, return_tensors="pt",
                padding="max_length" if max_length > 0 else True,
                truncation=True, max_length=max_length if max_length > 0 else None
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=False)
                last_hidden_state = outputs.last_hidden_state
                batch_sentence_embeddings = get_sentence_embedding(
                    last_hidden_state, inputs['attention_mask'], strategy=pooling_strategy,
                    cls_token_id=cls_token_id_for_pooling, eos_token_id=eos_token_id_for_pooling,
                    input_ids=inputs.get('input_ids')
                )
            all_embeddings_list.extend([emb.cpu().numpy().astype(np.float32) for emb in batch_sentence_embeddings])
        except Exception as e:
            logger.error(f"Corpus {corpus_label_for_logging}: Error processing batch {current_batch_num}/{total_batches}: {e}", exc_info=True)
            logger.warning(f"Corpus {corpus_label_for_logging}: Skipping batch {current_batch_num} due to error. Check logs for details.")
            continue # Skip to the next batch

        batch_processing_end_time = time.time()
        batch_duration = batch_processing_end_time - batch_processing_start_time

        log_this_batch_progress = (
            current_batch_num == 1 or
            current_batch_num % log_batch_interval == 0 or
            current_batch_num == total_batches
        )

        if log_this_batch_progress:
            percentage_complete = (current_batch_num / total_batches) * 100
            current_loop_elapsed_time = time.time() - loop_start_time
            avg_time_per_batch_so_far = current_loop_elapsed_time / current_batch_num if current_batch_num > 0 else 0

            etr_formatted = "N/A"
            if avg_time_per_batch_so_far > 0 and current_batch_num < total_batches :
                batches_remaining = total_batches - current_batch_num
                etr_seconds = batches_remaining * avg_time_per_batch_so_far
                etr_formatted = time.strftime("%H:%M:%S", time.gmtime(etr_seconds))
            
            logger.info(
                f"Corpus {corpus_label_for_logging}: Batch {current_batch_num}/{total_batches} "
                f"({percentage_complete:.1f}%) processed. "
                f"Last batch: {batch_duration:.2f}s. "
                f"Avg batch time: {avg_time_per_batch_so_far:.2f}s. "
                f"ETR: {etr_formatted}"
            )
        # Optional: More frequent DEBUG logging if INFO is sparse and DEBUG level is enabled
        elif logger.isEnabledFor(logging.DEBUG) and log_batch_interval > 20 and current_batch_num % (log_batch_interval // 5) == 0:
            logger.debug(
                f"Corpus {corpus_label_for_logging}: Processed batch {current_batch_num}/{total_batches}..."
            )

    total_loop_duration = time.time() - loop_start_time
    avg_batch_time_overall = (total_loop_duration / total_batches) if total_batches > 0 else 0
    logger.info(
        f"Corpus {corpus_label_for_logging}: Finished all {total_batches} batches. "
        f"Extracted {len(all_embeddings_list)} embeddings in {total_loop_duration:.2f}s "
        f"(Avg: {avg_batch_time_overall:.2f}s/batch)."
    )

    extraction_metadata["num_embeddings_extracted"] = len(all_embeddings_list)
    if all_embeddings_list:
        extraction_metadata["embedding_dimension"] = all_embeddings_list[0].shape[-1]
    else:
        extraction_metadata["embedding_dimension"] = "N/A (no embeddings extracted)"

    if len(all_embeddings_list) != len(texts) and total_batches > 0 : # Only warn if some processing was expected
        logger.warning(f"Corpus {corpus_label_for_logging}: Number of extracted embeddings ({len(all_embeddings_list)}) "
                       f"does not match number of input texts ({len(texts)}). "
                       "This may be due to errors during batch processing.")

    logger.info(f"Corpus {corpus_label_for_logging}: Successfully extracted embeddings for {len(all_embeddings_list)} out of {len(texts)} input sentences from model {model_name_or_path}.")
    return all_embeddings_list, extraction_metadata

# --- save_embeddings_with_metadata function (no major changes needed for logging here, it's already quite informative) ---
def save_embeddings_with_metadata(
    embeddings: List[np.ndarray], metadata: Dict[str, Any],
    output_path: str, output_format: str = "npz"):
    corpus_label = metadata.get("corpus_label", "N/A") # Get corpus label from metadata
    logger.info(f"Corpus {corpus_label}: Saving {len(embeddings)} embeddings and metadata to {output_path} in {output_format} format.")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Corpus {corpus_label}: Created directory: {output_dir}")
        except OSError as e:
            logger.error(f"Corpus {corpus_label}: Failed to create directory {output_dir}: {e}")
            pass # Proceed, saving might still work if dir exists due to race or perms

    if not embeddings:
        logger.warning(f"Corpus {corpus_label}: No embeddings were extracted to save for {output_path}. Saving metadata only.")
        metadata["num_embeddings_extracted"] = 0
        metadata["embedding_dimension"] = "N/A (no embeddings extracted)"
        if output_format == "npz":
            try:
                metadata_serializable = {}
                for k, v_item in metadata.items():
                    if isinstance(v_item, (dict, list, tuple)):
                        try: metadata_serializable[k] = json.dumps(v_item)
                        except TypeError: metadata_serializable[k] = str(v_item)
                    elif v_item is None: metadata_serializable[k] = "None"
                    else: metadata_serializable[k] = v_item
                
                metadata_str_array = np.array(json.dumps(metadata_serializable, indent=4), dtype=object)
                np.savez_compressed(output_path, metadata=metadata_str_array)
                logger.info(f"Corpus {corpus_label}: Saved metadata (only) to {output_path} (NPZ).")
            except Exception as e: logger.error(f"Corpus {corpus_label}: Error saving metadata-only NPZ to {output_path}: {e}")
        elif output_format == "hdf5" and H5PY_AVAILABLE:
             try:
                with h5py.File(output_path, 'w') as hf:
                    for key, value in metadata.items():
                        try:
                            if isinstance(value, (dict, list, tuple)): hf.attrs[key] = json.dumps(value)
                            elif value is None: hf.attrs[key] = "None"
                            else: hf.attrs[key] = value
                        except TypeError: hf.attrs[key] = str(value)
                logger.info(f"Corpus {corpus_label}: Saved metadata (only) as HDF5 attributes to {output_path}.")
             except Exception as e: logger.error(f"Corpus {corpus_label}: Error saving metadata-only HDF5 to {output_path}: {e}")
        return

    if output_format == "npz":
        try:
            embeddings_dict_to_save = {f'arr_{i}': emb for i, emb in enumerate(embeddings)}
            metadata_str_array = np.array(json.dumps(metadata, indent=4), dtype=object) # Serialize full metadata
            embeddings_dict_to_save['metadata'] = metadata_str_array
            np.savez_compressed(output_path, **embeddings_dict_to_save)
            logger.info(f"Corpus {corpus_label}: Embeddings and metadata successfully saved to {output_path} (NPZ).")
        except Exception as e:
            logger.error(f"Corpus {corpus_label}: Error saving NPZ file to {output_path}: {e}")
            raise
    elif output_format == "hdf5":
        if not H5PY_AVAILABLE:
            logger.error(f"Corpus {corpus_label}: h5py library is not installed, but HDF5 output format was selected. Please install h5py or choose 'npz' format.")
            raise ImportError("h5py is required for HDF5 output format.")
        try:
            with h5py.File(output_path, 'w') as hf:
                for i, emb in enumerate(embeddings):
                    hf.create_dataset(f"embedding_{i}", data=emb, compression="gzip")
                for key, value in metadata.items():
                    try:
                        if isinstance(value, (dict, list, tuple)): hf.attrs[key] = json.dumps(value)
                        elif value is None: hf.attrs[key] = "None" 
                        else: hf.attrs[key] = value
                    except TypeError as te:
                        logger.warning(f"Corpus {corpus_label}: Could not serialize metadata key '{key}' (value: {value}, type: {type(value)}) for HDF5. Storing as string. Error: {te}")
                        hf.attrs[key] = str(value) 
            logger.info(f"Corpus {corpus_label}: Embeddings and metadata successfully saved to {output_path} (HDF5).")
        except Exception as e:
            logger.error(f"Corpus {corpus_label}: Error saving HDF5 file to {output_path}: {e}")
            raise
    else:
        logger.error(f"Corpus {corpus_label}: Unsupported output format: {output_format}. Please choose 'npz' or 'hdf5'.")
        raise ValueError(f"Unsupported output format: {output_format}")


# --- if __name__ == '__main__': block ---
if __name__ == '__main__':
    overall_script_start_time = time.time()
    parser = argparse.ArgumentParser(description="Enhanced ETP Embedding Extractor")
    # ... (Arguments remain the same)
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Hugging Face model name or path.")
    parser.add_argument("--texts_file_A", type=str, default=None, help="Path to .txt file for Corpus A (one sentence per line). If None, uses dummy texts A.")
    parser.add_argument("--texts_file_B", type=str, default=None, help="Path to .txt file for Corpus B. If None and --output_path_B is set, uses dummy texts B.")
    parser.add_argument("--num_dummy_texts_A", type=int, default=70, help="Num dummy texts for corpus A if --texts_file_A is None.")
    parser.add_argument("--num_dummy_texts_B", type=int, default=60, help="Num dummy texts for corpus B if --texts_file_B is None and --output_path_B is set.")
    parser.add_argument("--dummy_text_use_diverse_pool", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use the diverse sentence pool for dummy texts.")
    parser.add_argument("--dummy_text_min_words", type=int, default=5, help="Min words for sentences from diverse pool.")
    parser.add_argument("--dummy_text_max_words", type=int, default=30, help="Max words for sentences from diverse pool.")
    parser.add_argument("--output_path_A", type=str, default="etp_corpus_A_embeddings.npz", help="Output path for Corpus A embeddings (.npz or .h5). Default relative to script execution dir.")
    parser.add_argument("--output_path_B", type=str, default=None, help="Optional output path for Corpus B embeddings. Default relative if only filename given.")
    parser.add_argument("--output_format", type=str, default="npz", choices=["npz", "hdf5"], help="Output file format.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Device to use ('auto', 'cuda', 'cpu', 'mps').")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for tokenizer. 0 for model's default / dynamic padding to batch max.")
    parser.add_argument("--pooling_strategy", type=str, default="mean", choices=POOLING_STRATEGIES, help="Pooling strategy.")
    parser.add_argument("--trust_remote_code", type=lambda x: (str(x).lower() == 'true'), default=True, help="Trust remote code for AutoModel/Tokenizer.")
    parser.add_argument("--use_bert_tiny_for_test", action="store_true", help="Use prajjwal1/bert-tiny for quick testing.")

    args = parser.parse_args()

    logger.info("Starting ETP Embedding Extractor script.")
    logger.info(f"Run arguments: {vars(args)}")


    if args.use_bert_tiny_for_test:
        logger.info("Overriding model to 'prajjwal1/bert-tiny' for quick testing.")
        args.model_name_or_path = "prajjwal1/bert-tiny"
        if args.max_length > 128 or args.max_length == 0 :
             logger.info(f"Adjusting max_length from {args.max_length} to 128 for bert-tiny test.")
             args.max_length = 128

    selected_device = get_device(args.device)
    logger.info(f"Using device: {selected_device}")

    if not H5PY_AVAILABLE and args.output_format == "hdf5":
        logger.error("HDF5 output format selected, but the 'h5py' library is not available. "
                     "Please install it (e.g., 'pip install h5py') or choose 'npz' format.")
        sys.exit(1)

    corpora_to_process = []
    if args.output_path_A:
        corpora_to_process.append(("A", args.texts_file_A, args.num_dummy_texts_A, args.output_path_A))
    else:
        logger.error("--output_path_A is required. Please specify an output path for Corpus A.")
        sys.exit(1)

    if args.output_path_B:
        corpora_to_process.append(("B", args.texts_file_B, args.num_dummy_texts_B, args.output_path_B))

    model_transformers_version_config_for_error_msg = "N/A" # For fallback error logging
    try:
        temp_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
        model_transformers_version_config_for_error_msg = getattr(temp_config, "transformers_version", "N/A")
    except Exception as e_cfg:
        logger.warning(f"Could not pre-fetch model config for {args.model_name_or_path} to get transformers_version: {e_cfg}")


    for corpus_label, texts_file, num_dummy, output_path_val in corpora_to_process:
        corpus_process_start_time = time.time()
        logger.info(f"--- Starting processing for Corpus {corpus_label} ---")
        texts_current_corpus: List[str]
        if texts_file:
            if not os.path.exists(texts_file):
                logger.error(f"Texts file not found for Corpus {corpus_label}: {texts_file}. Skipping this corpus.")
                continue
            logger.info(f"Loading texts for Corpus {corpus_label} from: {texts_file}")
            try:
                with open(texts_file, 'r', encoding='utf-8') as f:
                    texts_current_corpus = [line.strip() for line in f if line.strip()]
                if not texts_current_corpus:
                    logger.warning(f"Text file {texts_file} for Corpus {corpus_label} is empty or contains only whitespace. Skipping this corpus.")
                    continue
                logger.info(f"Loaded {len(texts_current_corpus)} texts for Corpus {corpus_label}.")
            except Exception as e:
                logger.error(f"Error reading text file {texts_file} for Corpus {corpus_label}: {e}. Skipping this corpus.")
                continue
        else:
            logger.info(f"No text file provided for Corpus {corpus_label}. Using {num_dummy} dummy texts.")
            texts_current_corpus = load_dummy_texts(
                num_texts=num_dummy, corpus_label=corpus_label,
                use_diverse_pool=args.dummy_text_use_diverse_pool,
                min_words=args.dummy_text_min_words, max_words=args.dummy_text_max_words
            )
            logger.info(f"Generated {len(texts_current_corpus)} dummy texts for Corpus {corpus_label}.")

        if not texts_current_corpus:
            logger.warning(f"No texts available for Corpus {corpus_label} after loading/generation. Skipping this corpus.")
            continue

        logger.info(f"Extracting embeddings for Corpus {corpus_label} ({len(texts_current_corpus)} texts)...")
        try:
            embeddings_corpus, metadata_corpus = extract_embeddings(
                texts_current_corpus, args.model_name_or_path, selected_device, args.batch_size,
                args.max_length, args.pooling_strategy, args.trust_remote_code,
                corpus_label_for_logging=corpus_label # Pass corpus label here
            )
            # Ensure transformers version from model config is in metadata, if fetched
            if "transformers_version_config_from_model" not in metadata_corpus:
                 metadata_corpus["transformers_version_config_from_model"] = model_transformers_version_config_for_error_msg

            if embeddings_corpus or metadata_corpus.get("num_embeddings_extracted", 0) == 0 :
                save_embeddings_with_metadata(embeddings_corpus, metadata_corpus, output_path_val, args.output_format)
            else:
                logger.warning(f"No embeddings were extracted for Corpus {corpus_label}, and metadata indicates 0 extracted. Nothing saved to {output_path_val}.")
            
            corpus_process_end_time = time.time()
            logger.info(f"--- Corpus {corpus_label} processing finished in {corpus_process_end_time - corpus_process_start_time:.2f}s. Data saved to {output_path_val} ---")

        except Exception as e:
            error_log_transformers_version = model_transformers_version_config_for_error_msg \
                if model_transformers_version_config_for_error_msg != "N/A" else "like 4.44.0"

            logger.error(f"An unhandled error occurred during processing or saving for Corpus {corpus_label}: {e}", exc_info=True)
            if isinstance(e, TypeError) and "argument of type 'NoneType' is not iterable" in str(e) and "ALL_PARALLEL_STYLES" in str(e):
                 logger.error("This specific TypeError often suggests an incompatibility with the model's configuration or current transformers version. "
                              f"Model was saved with transformers: {error_log_transformers_version}. Consider aligning versions or checking model's Hugging Face page for loading issues.")
            logger.warning(f"Processing for Corpus {corpus_label} failed. Check logs.")
            corpus_process_end_time = time.time()
            logger.info(f"--- Corpus {corpus_label} processing attempted and failed in {corpus_process_end_time - corpus_process_start_time:.2f}s ---")


    overall_script_end_time = time.time()
    logger.info(f"All specified embedding extraction tasks finished in {overall_script_end_time - overall_script_start_time:.2f}s.")