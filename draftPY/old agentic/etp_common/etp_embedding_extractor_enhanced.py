import argparse
import json
import logging
import os
import random
import sys
import time
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel

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
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Extracts sentence embeddings from a list of texts using a specified model.
    """
    logger.info(f"Loading model and tokenizer: {model_name_or_path} (trust_remote_code={trust_remote_code})")
    extraction_metadata = {
        "model_name_or_path": model_name_or_path, "device": str(device),
        "batch_size": batch_size, "max_length": max_length,
        "pooling_strategy": pooling_strategy,
        "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "num_texts_input": len(texts)
    }
    try:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model: PreTrainedModel = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model.to(device); model.eval()
        extraction_metadata["model_config_class"] = model.config.__class__.__name__
        extraction_metadata["tokenizer_class"] = tokenizer.__class__.__name__
        # Attempt to get some specific config values if they exist
        extraction_metadata["model_hidden_size"] = getattr(model.config, 'hidden_size', 'N/A')
        extraction_metadata["model_num_layers"] = getattr(model.config, 'num_hidden_layers', 'N/A')

    except Exception as e:
        logger.error(f"Error loading model or tokenizer '{model_name_or_path}': {e}")
        if "RateLimiter" in str(e) or "Connection error" in str(e):
             logger.error("This might be a Hugging Face Hub connection issue. Ensure you are online and not rate-limited.")
        raise

    cls_token_id_for_pooling = getattr(tokenizer, 'cls_token_id', None)
    eos_token_id_for_pooling = getattr(tokenizer, 'eos_token_id', None)
    if pooling_strategy == "cls" and cls_token_id_for_pooling is None:
        logger.warning(f"CLS pooling requested but tokenizer ({tokenizer.__class__.__name__}) has no cls_token_id. Will use first token.")
    if pooling_strategy == "last_token" and eos_token_id_for_pooling is None:
        logger.warning(f"Last_token pooling requested but tokenizer ({tokenizer.__class__.__name__}) has no eos_token_id. Will use last non-padding token via attention_mask.")

    all_embeddings_list: List[np.ndarray] = []
    logger.info(f"Processing {len(texts)} texts in batches of {batch_size} on {device}.")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        logger.debug(f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
        try:
            inputs = tokenizer(batch_texts, return_tensors="pt",
                               padding="max_length" if max_length > 0 else True,
                               truncation=True, max_length=max_length if max_length > 0 else None)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=False)
                last_hidden_state = outputs.last_hidden_state
                batch_sentence_embeddings = get_sentence_embedding(
                    last_hidden_state, inputs['attention_mask'], strategy=pooling_strategy,
                    cls_token_id=cls_token_id_for_pooling, eos_token_id=eos_token_id_for_pooling,
                    input_ids=inputs.get('input_ids'))
            all_embeddings_list.extend([emb.cpu().numpy().astype(np.float32) for emb in batch_sentence_embeddings])
        except Exception as e:
            logger.error(f"Error processing batch {i // batch_size + 1}: {e}", exc_info=True)
            logger.warning(f"Skipping batch {i // batch_size + 1} due to error.")
            continue
    extraction_metadata["num_embeddings_extracted"] = len(all_embeddings_list)
    if all_embeddings_list: extraction_metadata["embedding_dimension"] = all_embeddings_list[0].shape[-1]
    logger.info(f"Successfully extracted embeddings for {len(all_embeddings_list)} sentences from model {model_name_or_path}.")
    return all_embeddings_list, extraction_metadata

def save_embeddings_with_metadata(
    embeddings: List[np.ndarray], metadata: Dict[str, Any],
    output_path: str, output_format: str = "npz"):
    logger.info(f"Saving {len(embeddings)} embeddings and metadata to {output_path} in {output_format} format.")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir); logger.info(f"Created directory: {output_dir}")

    if not embeddings:
        logger.warning(f"No embeddings to save for {output_path}.")
        if output_format == "npz":
            metadata_str = json.dumps(metadata, indent=4)
            np.savez_compressed(output_path, metadata=np.array([metadata_str], dtype=object))
            logger.info(f"Saved metadata (only) to {output_path}.")
        elif output_format == "hdf5" and H5PY_AVAILABLE:
             with h5py.File(output_path, 'w') as hf:
                for key, value in metadata.items():
                    try: hf.attrs[key] = json.dumps(value) if isinstance(value, (dict,list)) else value
                    except TypeError: hf.attrs[key] = str(value)
                logger.info(f"Saved metadata (only) as HDF5 attributes to {output_path}.")
        return

    if output_format == "npz":
        try:
            embeddings_dict_to_save = {f'arr_{i}': emb for i, emb in enumerate(embeddings)}
            metadata_str = json.dumps(metadata, indent=4)
            embeddings_dict_to_save['metadata'] = np.array([metadata_str], dtype=object)
            np.savez_compressed(output_path, **embeddings_dict_to_save)
            logger.info(f"Embeddings and metadata successfully saved to {output_path} (NPZ).")
        except Exception as e: logger.error(f"Error saving NPZ: {e}"); raise
    elif output_format == "hdf5":
        if not H5PY_AVAILABLE:
            logger.error("h5py not installed for HDF5 save."); raise ImportError("h5py required.")
        try:
            with h5py.File(output_path, 'w') as hf:
                for i, emb in enumerate(embeddings): hf.create_dataset(f"embedding_{i}", data=emb)
                for key, value in metadata.items():
                    try:
                        if isinstance(value, (dict, list)): hf.attrs[key] = json.dumps(value)
                        else: hf.attrs[key] = value
                    except TypeError: hf.attrs[key] = str(value)
            logger.info(f"Embeddings and metadata successfully saved to {output_path} (HDF5).")
        except Exception as e: logger.error(f"Error saving HDF5: {e}"); raise
    else:
        logger.error(f"Unsupported output format: {output_format}. Choose 'npz' or 'hdf5'.")
        raise ValueError(f"Unsupported output format: {output_format}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced ETP Embedding Extractor")
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Hugging Face model name or path.")
    parser.add_argument("--texts_file_A", type=str, default=None, help="Path to .txt file for Corpus A (one sentence per line). If None, uses dummy texts A.")
    parser.add_argument("--texts_file_B", type=str, default=None, help="Path to .txt file for Corpus B. If None and --output_path_B is set, uses dummy texts B.")
    parser.add_argument("--num_dummy_texts_A", type=int, default=70, help="Num dummy texts for corpus A if --texts_file_A is None.")
    parser.add_argument("--num_dummy_texts_B", type=int, default=60, help="Num dummy texts for corpus B if --texts_file_B is None and --output_path_B is set.")
    parser.add_argument("--dummy_text_use_diverse_pool", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use the diverse sentence pool for dummy texts.")
    parser.add_argument("--dummy_text_min_words", type=int, default=5, help="Min words for sentences from diverse pool.")
    parser.add_argument("--dummy_text_max_words", type=int, default=30, help="Max words for sentences from diverse pool.")
    parser.add_argument("--output_path_A", type=str, default="draftPY/etp_corpus_A_deepseek_embeddings.npz", help="Output path for Corpus A embeddings (.npz or .h5).")
    parser.add_argument("--output_path_B", type=str, default=None, help="Optional output path for Corpus B embeddings.")
    parser.add_argument("--output_format", type=str, default="npz", choices=["npz", "hdf5"], help="Output file format.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use ('auto', 'cuda', 'cpu').")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for tokenizer. 0 for model's default.")
    parser.add_argument("--pooling_strategy", type=str, default="mean", choices=POOLING_STRATEGIES, help="Pooling strategy.")
    parser.add_argument("--trust_remote_code", type=lambda x: (str(x).lower() == 'true'), default=True, help="Trust remote code for AutoModel/Tokenizer.")
    parser.add_argument("--use_bert_tiny_for_test", action="store_true", help="Use prajjwal1/bert-tiny for quick testing.")
    args = parser.parse_args()

    if args.use_bert_tiny_for_test:
        args.model_name_or_path = "prajjwal1/bert-tiny"; logger.info("Overriding model to 'prajjwal1/bert-tiny' for testing.")
        if args.max_length > 128 : args.max_length = 128

    selected_device = get_device(args.device); logger.info(f"Using device: {selected_device}")
    if not H5PY_AVAILABLE and args.output_format == "hdf5":
        logger.error("HDF5 selected, but h5py unavailable. Install or use 'npz'."); sys.exit(1)

    corpora_to_process = [("A", args.texts_file_A, args.num_dummy_texts_A, args.output_path_A)]
    if args.output_path_B:
        corpora_to_process.append(("B", args.texts_file_B, args.num_dummy_texts_B, args.output_path_B))

    for corpus_label, texts_file, num_dummy, output_path_val in corpora_to_process:
        if not output_path_val: continue # Skip if output path is not defined (e.g. B is optional)
        
        texts_current_corpus: List[str]
        if texts_file:
            logger.info(f"Loading texts for Corpus {corpus_label} from: {texts_file}")
            try:
                with open(texts_file, 'r', encoding='utf-8') as f:
                    texts_current_corpus = [line.strip() for line in f if line.strip()]
                if not texts_current_corpus:
                    logger.error(f"Text file {texts_file} is empty or invalid for Corpus {corpus_label}. Skipping.")
                    continue
            except FileNotFoundError:
                logger.error(f"Texts file not found for Corpus {corpus_label}: {texts_file}. Skipping.")
                continue
        else:
            logger.info(f"Using {num_dummy} dummy texts for Corpus {corpus_label}.")
            texts_current_corpus = load_dummy_texts(
                num_dummy, corpus_label,
                use_diverse_pool=args.dummy_text_use_diverse_pool,
                min_words=args.dummy_text_min_words,
                max_words=args.dummy_text_max_words
            )
        logger.info(f"Extracting embeddings for Corpus {corpus_label} ({len(texts_current_corpus)} texts)...")
        embeddings_corpus, metadata_corpus = extract_embeddings(
            texts_current_corpus, args.model_name_or_path, selected_device, args.batch_size,
            args.max_length, args.pooling_strategy, args.trust_remote_code
        )
        if embeddings_corpus:
            save_embeddings_with_metadata(embeddings_corpus, metadata_corpus, output_path_val, args.output_format)
            logger.info(f"Corpus {corpus_label} embeddings and metadata saved to {output_path_val}")
        else:
            logger.warning(f"No embeddings extracted for Corpus {corpus_label}. Nothing saved to {output_path_val} (except maybe metadata).")
    logger.info("Embedding extraction process finished.")