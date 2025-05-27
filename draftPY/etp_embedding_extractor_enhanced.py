import argparse
import json
import logging
import os
import random
import sys
import time
from typing import List, Dict, Any, Optional, Tuple, Union # Added Union
from pathlib import Path # Added Path

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
POOLING_STRATEGIES = ["mean", "cls", "last_token", "none"] # Added "none" for layerwise

# A small, diverse set of example sentences (keep as is)
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

def get_sentence_embedding( # This function might be less relevant if extracting all layers
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: str = "mean",
    cls_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    input_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes sentence embedding based on the chosen strategy.
    If strategy is "none", it returns the last_hidden_state itself (for layerwise).
    """
    if strategy == "none": # New case for layerwise
        return last_hidden_state
    elif strategy == "mean":
        expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * expanded_mask, 1)
        sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    elif strategy == "cls":
        # ... (CLS pooling logic remains the same)
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
        # ... (Last token pooling logic remains the same)
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


import argparse
import json
import logging
import os
import random
import sys
import time
from typing import List, Dict, Any, Optional, Tuple, Union # Added Union
from pathlib import Path # Added Path

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
POOLING_STRATEGIES = ["mean", "cls", "last_token", "none"] # Added "none" for layerwise

# A small, diverse set of example sentences (keep as is)
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

def get_sentence_embedding( # This function might be less relevant if extracting all layers
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: str = "mean",
    cls_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    input_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes sentence embedding based on the chosen strategy.
    If strategy is "none", it returns the last_hidden_state itself (for layerwise).
    """
    if strategy == "none": # New case for layerwise
        return last_hidden_state
    elif strategy == "mean":
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


import argparse
import json
import logging
import os
import random
import sys
import time
from typing import List, Dict, Any, Optional, Tuple, Union # Added Union
from pathlib import Path # Added Path

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
POOLING_STRATEGIES = ["mean", "cls", "last_token", "none"] # Added "none" for layerwise

# A small, diverse set of example sentences (keep as is)
DIVERSE_SENTENCE_POOL = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is rapidly transforming various industries.",
    "Hyperbolic geometry offers unique advantages for embedding hierarchical data.",
    # ... (rest of DIVERSE_SENTENCE_POOL remains the same)
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
    # ... (load_dummy_texts function remains the same)
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
    # ... (get_device function remains the same)
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # For Apple Silicon
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def get_sentence_embedding( # This function might be less relevant if extracting all layers
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: str = "mean",
    cls_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    input_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes sentence embedding based on the chosen strategy.
    If strategy is "none", it returns the last_hidden_state itself (for layerwise).
    """
    # ... (get_sentence_embedding function remains the same)
    if strategy == "none": # New case for layerwise
        return last_hidden_state
    elif strategy == "mean":
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
    corpus_label_for_logging: str = "N/A",
    extract_all_hidden_states: bool = False,
    extract_attentions: bool = False
) -> Tuple[Union[List[np.ndarray], Dict[str, List[np.ndarray]]], Dict[str, Any]]:
    """
    Extracts embeddings from texts. Can extract pooled sentence embeddings,
    all hidden layer states, and/or attention weights.
    """
    logger.info(f"Corpus {corpus_label_for_logging}: Loading model and tokenizer: {model_name_or_path} (trust_remote_code={trust_remote_code})")
    model_load_start_time = time.time()

    extraction_metadata = {
        "model_name_or_path": model_name_or_path, "device": str(device),
        "batch_size": batch_size, "max_length": max_length,
        "pooling_strategy": pooling_strategy,
        "extract_all_hidden_states": extract_all_hidden_states,
        "extract_attentions": extract_attentions,
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

    model: Optional[PreTrainedModel] = None
    try:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )

        model_load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "output_hidden_states": extract_all_hidden_states,
            "output_attentions": extract_attentions,
        }

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model_preferred_dtype_str_config = getattr(config, "torch_dtype", "float32")
        if model_preferred_dtype_str_config is None: model_preferred_dtype_str_config = "float32"

        model_transformers_version_config = getattr(config, "transformers_version", "N/A")
        extraction_metadata["transformers_version_config_from_model"] = model_transformers_version_config

        logger.info(f"Corpus {corpus_label_for_logging}: Model config suggests torch_dtype='{model_preferred_dtype_str_config}' and was saved with transformers_version='{model_transformers_version_config}'.")
        logger.info(f"Corpus {corpus_label_for_logging}: Currently using transformers library version: '{model_transformers_version_from_lib}'.")

        target_torch_dtype_for_load = None
        chosen_attn_implementation = "auto"


        is_deepseek_or_qwen_model = "deepseek" in model_name_or_path.lower() or "qwen" in model_name_or_path.lower()

        if device.type == 'cuda':
            if is_deepseek_or_qwen_model:
                logger.info(f"Corpus {corpus_label_for_logging}: Model is DeepSeek/Qwen on CUDA.")
                if extract_attentions:
                    chosen_attn_implementation = "eager"
                    logger.info(f"Corpus {corpus_label_for_logging}: Extracting attentions, setting attn_implementation='eager' for compatibility.")
                elif hasattr(config, "_attn_implementation") and config._attn_implementation == "flash_attention_2":
                     chosen_attn_implementation = "flash_attention_2"
                     logger.info(f"Corpus {corpus_label_for_logging}: Model config supports Flash Attention 2, and not extracting attentions. Will attempt to use 'flash_attention_2'.")
                elif hasattr(config, "_attn_implementation") and config._attn_implementation == "sdpa":
                    chosen_attn_implementation = "sdpa"
                    logger.info(f"Corpus {corpus_label_for_logging}: Model config supports SDPA, and not extracting attentions. Will attempt to use 'sdpa'.")
                else: 
                    chosen_attn_implementation = "sdpa" 
                    logger.info(f"Corpus {corpus_label_for_logging}: Not extracting attentions. Defaulting to 'sdpa' for DeepSeek/Qwen on CUDA.")
            elif extract_attentions: 
                chosen_attn_implementation = "eager"
                logger.info(f"Corpus {corpus_label_for_logging}: Extracting attentions on CUDA for non-DeepSeek/Qwen model. Setting attn_implementation='eager'.")
            else: 
                chosen_attn_implementation = "sdpa"
                logger.info(f"Corpus {corpus_label_for_logging}: Not extracting attentions on CUDA for non-DeepSeek/Qwen model. Defaulting to 'sdpa'.")

            if torch.cuda.is_bf16_supported():
                logger.info(f"Corpus {corpus_label_for_logging}: Device supports bfloat16. Setting torch_dtype=torch.bfloat16.")
                target_torch_dtype_for_load = torch.bfloat16
            else:
                logger.info(f"Corpus {corpus_label_for_logging}: Device does not report bfloat16 support. Using torch.float16.")
                target_torch_dtype_for_load = torch.float16
        else: 
            logger.info(f"Corpus {corpus_label_for_logging}: Device is {device.type}. Using 'eager' attention if attentions extracted, else 'auto'. Defaulting to float32.")
            if extract_attentions:
                chosen_attn_implementation = "eager"
            if model_preferred_dtype_str_config == "bfloat16" and hasattr(torch, 'bfloat16'): 
                 target_torch_dtype_for_load = torch.bfloat16
            else:
                 target_torch_dtype_for_load = torch.float32

        if target_torch_dtype_for_load:
            model_load_kwargs["torch_dtype"] = target_torch_dtype_for_load
        
        if chosen_attn_implementation != "auto":
            model_load_kwargs["attn_implementation"] = chosen_attn_implementation

        logger.info(f"Corpus {corpus_label_for_logging}: Attempting model load with primary kwargs: {model_load_kwargs}")

        try:
            model = AutoModel.from_pretrained(
                model_name_or_path,
                **model_load_kwargs
            )
        except Exception as e_load_primary:
            logger.warning(f"Corpus {corpus_label_for_logging}: Primary loading attempt with kwargs {model_load_kwargs} FAILED: {e_load_primary}", exc_info=False)
            
            if model_load_kwargs.get("attn_implementation") != "eager":
                logger.info(f"Corpus {corpus_label_for_logging}: Fallback 1: Retrying with attn_implementation='eager'.")
                model_load_kwargs_fallback1 = model_load_kwargs.copy()
                model_load_kwargs_fallback1["attn_implementation"] = "eager"
                logger.info(f"Corpus {corpus_label_for_logging}: Fallback 1 (eager) kwargs: {model_load_kwargs_fallback1}")
                try:
                    model = AutoModel.from_pretrained(model_name_or_path, **model_load_kwargs_fallback1)
                except Exception as e_fallback1:
                    logger.warning(f"Corpus {corpus_label_for_logging}: Fallback 1 (eager attention) FAILED: {e_fallback1}", exc_info=False)

            if model is None and "torch_dtype" in model_load_kwargs:
                logger.info(f"Corpus {corpus_label_for_logging}: Fallback 2: Retrying without explicit torch_dtype.")
                model_load_kwargs_fallback2 = model_load_kwargs.copy()
                del model_load_kwargs_fallback2["torch_dtype"]
                if model_load_kwargs_fallback2.get("attn_implementation") != "eager" and \
                   (extract_attentions or model_load_kwargs.get("attn_implementation") != "auto"):
                    model_load_kwargs_fallback2["attn_implementation"] = "eager"
                
                logger.info(f"Corpus {corpus_label_for_logging}: Fallback 2 kwargs: {model_load_kwargs_fallback2}")
                try:
                    model = AutoModel.from_pretrained(model_name_or_path, **model_load_kwargs_fallback2)
                except Exception as e_fallback2:
                    logger.warning(f"Corpus {corpus_label_for_logging}: Fallback 2 (no explicit dtype) FAILED: {e_fallback2}", exc_info=False)

            if model is None:
                logger.info(f"Corpus {corpus_label_for_logging}: Fallback 3: Attempting load with minimal safe arguments.")
                minimal_kwargs = {
                    "trust_remote_code": trust_remote_code,
                    "torch_dtype": torch.float32, 
                    "output_hidden_states": extract_all_hidden_states,
                    "output_attentions": extract_attentions
                }
                if extract_attentions: 
                    minimal_kwargs["attn_implementation"] = "eager"
                else: 
                    minimal_kwargs["attn_implementation"] = "sdpa" if device.type == 'cuda' else "auto"

                logger.info(f"Corpus {corpus_label_for_logging}: Minimal fallback kwargs: {minimal_kwargs}")
                try:
                    model = AutoModel.from_pretrained(model_name_or_path, **minimal_kwargs)
                except Exception as e_fallback_final:
                    logger.error(f"Corpus {corpus_label_for_logging}: All loading attempts FAILED. Last error (minimal fallback): {e_fallback_final}", exc_info=True)
                    raise e_fallback_final
        
        if model is None:
            raise RuntimeError(f"Corpus {corpus_label_for_logging}: Model could not be loaded after multiple fallbacks.")

        model.to(device)
        model.eval()

        model_load_end_time = time.time()
        logger.info(f"Corpus {corpus_label_for_logging}: Model and tokenizer loaded in {model_load_end_time - model_load_start_time:.2f}s.")
        
        extraction_metadata["model_config_class"] = model.config.__class__.__name__
        extraction_metadata["tokenizer_class"] = tokenizer.__class__.__name__
        extraction_metadata["model_hidden_size"] = getattr(model.config, 'hidden_size', 'N/A')
        num_hidden_layers = getattr(model.config, 'num_hidden_layers', 0)
        extraction_metadata["model_num_layers"] = num_hidden_layers
        extraction_metadata["vocab_size"] = getattr(model.config,'vocab_size', getattr(tokenizer, 'vocab_size', 'N/A'))
        actual_attn_impl_from_config = getattr(model.config, '_attn_implementation', 'Not Specified in Config')
        extraction_metadata["actual_attn_implementation"] = actual_attn_impl_from_config
        extraction_metadata["actual_torch_dtype"] = str(model.dtype)
        logger.info(f"Corpus {corpus_label_for_logging}: Model loaded. Actual attention implementation (from config): {actual_attn_impl_from_config}, "
                    f"Actual dtype: {extraction_metadata['actual_torch_dtype']}, Num layers: {num_hidden_layers}")

    except Exception as e:
        logger.error(f"FATAL (Corpus {corpus_label_for_logging}): Error loading model or tokenizer '{model_name_or_path}': {e}", exc_info=True)
        if isinstance(e, TypeError) and "argument of type 'NoneType' is not iterable" in str(e) and "ALL_PARALLEL_STYLES" in str(e):
             logger.error("The persistent TypeError related to 'ALL_PARALLEL_STYLES' suggests a core incompatibility. "
                          "Final recommendation: try using transformers version "
                          f"{model_transformers_version_config if model_transformers_version_config != 'N/A' else 'similar to model save version'} "
                          "or investigate model-specific loading on Hugging Face discussions for this transformers version.")
        raise

    cls_token_id_for_pooling = getattr(tokenizer, 'cls_token_id', None)
    eos_token_id_for_pooling = getattr(tokenizer, 'eos_token_id', None)
    if pooling_strategy == "cls" and cls_token_id_for_pooling is None:
        logger.warning(f"Corpus {corpus_label_for_logging}: CLS pooling requested but tokenizer ({tokenizer.__class__.__name__}) has no defined 'cls_token_id'. Will use first token of sequence.")
    if pooling_strategy == "last_token" and eos_token_id_for_pooling is None:
        logger.warning(f"Corpus {corpus_label_for_logging}: Last_token pooling requested but tokenizer ({tokenizer.__class__.__name__}) has no defined 'eos_token_id'. Will use last non-padding token via attention_mask.")

    extracted_data: Union[List[np.ndarray], Dict[str, List[np.ndarray]]]
    if extract_all_hidden_states:
        # Ensure num_hidden_layers is valid (it could be 0 if model config is minimal)
        actual_num_layers_for_dict = num_hidden_layers if num_hidden_layers > 0 else 1 
        extracted_data = {f"hidden_state_layer_{i}": [] for i in range(actual_num_layers_for_dict + 1)} # +1 for embeddings
    else:
        extracted_data = [] 

    extracted_attention_data: Optional[Dict[str, List[np.ndarray]]] = None
    if extract_attentions:
        actual_num_layers_for_attn = num_hidden_layers if num_hidden_layers > 0 else 1
        extracted_attention_data = {f"attention_layer_{i}": [] for i in range(actual_num_layers_for_attn)}


    total_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(f"Corpus {corpus_label_for_logging}: Processing {len(texts)} texts in {total_batches} batches of size {batch_size} on {device}.")
    
    if total_batches == 0:
        logger.warning(f"Corpus {corpus_label_for_logging}: No texts to process, skipping embedding extraction loop.")
        extraction_metadata["num_embeddings_extracted"] = 0
        extraction_metadata["embedding_dimension"] = "N/A (no texts)"
        return extracted_data, extraction_metadata


    if total_batches < 20: log_batch_interval = 1
    elif total_batches < 200: log_batch_interval = 10
    elif total_batches < 2000: log_batch_interval = 50
    else: log_batch_interval = 100

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
            inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs_on_device) 

                if extract_all_hidden_states:
                    if not outputs.hidden_states:
                        logger.error(f"Corpus {corpus_label_for_logging}: Model outputs.hidden_states is None despite 'output_hidden_states=True'. This should not happen. Skipping hidden state extraction for this batch.")
                    else:
                        for layer_idx, layer_hidden_states_batch in enumerate(outputs.hidden_states):
                            # layer_hidden_states_batch is (batch_size, seq_len, hidden_size)
                            # Ensure key exists, especially if num_hidden_layers was 0 initially
                            dict_key = f"hidden_state_layer_{layer_idx}"
                            if dict_key not in extracted_data: # Should not happen with pre-initialization
                                extracted_data[dict_key] = []
                                logger.warning(f"Dynamically added key {dict_key} to extracted_data.")

                            for k_item_idx in range(layer_hidden_states_batch.shape[0]): # Iterate over items in batch
                                # Convert to float32 ON THE SAME DEVICE first, then CPU, then NumPy
                                tensor_to_save = layer_hidden_states_batch[k_item_idx].to(torch.float32).cpu().numpy()
                                extracted_data[dict_key].append(tensor_to_save)
                else: 
                    last_hidden_state = outputs.last_hidden_state
                    batch_sentence_embeddings = get_sentence_embedding(
                        last_hidden_state, inputs_on_device['attention_mask'], strategy=pooling_strategy,
                        cls_token_id=cls_token_id_for_pooling, eos_token_id=eos_token_id_for_pooling,
                        input_ids=inputs_on_device.get('input_ids')
                    )
                    if isinstance(extracted_data, list):
                        # Ensure conversion to float32 before numpy if original was bfloat16
                        processed_embeddings = [
                            emb.to(torch.float32).cpu().numpy() for emb in batch_sentence_embeddings
                        ]
                        extracted_data.extend(processed_embeddings)

                if extract_attentions and extracted_attention_data is not None:
                    if not outputs.attentions:
                        logger.error(f"Corpus {corpus_label_for_logging}: Model outputs.attentions is None despite 'output_attentions=True'. Skipping attention extraction for this batch.")
                    else:
                        for layer_idx, layer_attentions_batch in enumerate(outputs.attentions):
                            # layer_attentions_batch is (batch_size, num_heads, seq_len, seq_len)
                            dict_key = f"attention_layer_{layer_idx}"
                            if dict_key not in extracted_attention_data: # Should not happen
                                extracted_attention_data[dict_key] = []
                                logger.warning(f"Dynamically added key {dict_key} to extracted_attention_data.")

                            for k_item_idx in range(layer_attentions_batch.shape[0]): # Iterate over items in batch
                                # Convert to float32 ON THE SAME DEVICE first, then CPU, then NumPy
                                tensor_to_save = layer_attentions_batch[k_item_idx].to(torch.float32).cpu().numpy()
                                extracted_attention_data[dict_key].append(tensor_to_save)

        except Exception as e:
            logger.error(f"Corpus {corpus_label_for_logging}: Error processing batch {current_batch_num}/{total_batches}: {e}", exc_info=True)
            logger.warning(f"Corpus {corpus_label_for_logging}: Skipping batch {current_batch_num} due to error. Check logs for details.")
            continue

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
        elif logger.isEnabledFor(logging.DEBUG) and log_batch_interval > 20 and current_batch_num % (log_batch_interval // 5) == 0:
            logger.debug(
                f"Corpus {corpus_label_for_logging}: Processed batch {current_batch_num}/{total_batches}..."
            )

    total_loop_duration = time.time() - loop_start_time
    avg_batch_time_overall = (total_loop_duration / total_batches) if total_batches > 0 else 0
    
    num_actually_extracted = 0
    if isinstance(extracted_data, list):
        num_actually_extracted = len(extracted_data)
    elif isinstance(extracted_data, dict) and extracted_data:
        # Use the count from the first layer's list (should be consistent across layers for hidden states)
        # For robustness, find a key that actually has data if some layers might be skipped (though unlikely with current loop)
        first_valid_key = None
        for k_iter in extracted_data:
            if extracted_data[k_iter]: # Check if the list for this key is not empty
                first_valid_key = k_iter
                break
        if first_valid_key:
            num_actually_extracted = len(extracted_data[first_valid_key])
        else: # All lists in dict might be empty if all batches failed
            num_actually_extracted = 0


    logger.info(
        f"Corpus {corpus_label_for_logging}: Finished all {total_batches} batches. "
        f"Extracted data for {num_actually_extracted} items in {total_loop_duration:.2f}s "
        f"(Avg: {avg_batch_time_overall:.2f}s/batch)."
    )

    extraction_metadata["num_embeddings_extracted"] = num_actually_extracted
    if isinstance(extracted_data, list) and extracted_data:
        extraction_metadata["embedding_dimension"] = extracted_data[0].shape[-1]
    elif isinstance(extracted_data, dict) and extracted_data and model: 
        extraction_metadata["embedding_dimension"] = getattr(model.config, 'hidden_size', 'N/A')
    else:
        extraction_metadata["embedding_dimension"] = "N/A (no embeddings extracted or model info missing)"

    if num_actually_extracted != len(texts) and total_batches > 0 :
        logger.warning(f"Corpus {corpus_label_for_logging}: Number of extracted items ({num_actually_extracted}) "
                       f"does not match number of input texts ({len(texts)}). "
                       "This may be due to errors during batch processing.")

    logger.info(f"Corpus {corpus_label_for_logging}: Successfully extracted data for {num_actually_extracted} out of {len(texts)} input sentences from model {model_name_or_path}.")
    
    final_data_to_save = extracted_data
    if extracted_attention_data:
        if isinstance(final_data_to_save, list): # This case should be rare if attentions are on
            logger.warning("Attentions extracted but main data is a list (pooled). Attentions will be saved under 'attentions' key.")
            final_data_to_save = {"pooled_embeddings": final_data_to_save, "attentions": extracted_attention_data}
        elif isinstance(final_data_to_save, dict):
            final_data_to_save["attentions"] = extracted_attention_data
            
    return final_data_to_save, extraction_metadata



def save_embeddings_with_metadata(
    data_to_save: Union[List[np.ndarray], Dict[str, Union[List[np.ndarray], Dict[str, List[np.ndarray]]]]],
    metadata: Dict[str, Any],
    output_path: str, output_format: str = "npz"):
    corpus_label = metadata.get("corpus_label", "N/A")
    logger.info(f"Corpus {corpus_label}: Saving extracted data and metadata to {output_path} in {output_format} format.")
    # ... (output_dir creation logic remains the same) ...
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Corpus {corpus_label}: Created directory: {output_dir}")
        except OSError as e:
            logger.error(f"Corpus {corpus_label}: Failed to create directory {output_dir}: {e}")
            pass


    # Check if data_to_save is empty (either empty list or empty dict, or dict with empty lists)
    is_data_empty = False
    if isinstance(data_to_save, list) and not data_to_save:
        is_data_empty = True
    elif isinstance(data_to_save, dict):
        if not data_to_save:
            is_data_empty = True
        else:
            # Check if all lists within the dict are empty
            all_internal_lists_empty = True
            for key, value_list_or_dict in data_to_save.items():
                if isinstance(value_list_or_dict, list) and value_list_or_dict:
                    all_internal_lists_empty = False
                    break
                elif isinstance(value_list_or_dict, dict): # For nested dict like attentions
                    for sub_key, sub_list in value_list_or_dict.items():
                        if isinstance(sub_list, list) and sub_list:
                            all_internal_lists_empty = False
                            break
                    if not all_internal_lists_empty: break # break outer loop
            if all_internal_lists_empty:
                is_data_empty = True

    if is_data_empty:
        logger.warning(f"Corpus {corpus_label}: No actual data (embeddings/states/attentions) extracted to save for {output_path}. Saving metadata only.")
        # ... (metadata-only saving logic remains the same) ...
        metadata["num_embeddings_extracted"] = 0 # ensure this is set correctly
        metadata["embedding_dimension"] = "N/A (no data extracted)"
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
            data_to_npz = {}
            if isinstance(data_to_save, list): # Pooled embeddings
                # Stack if possible (all same shape, common for pooled)
                if data_to_save and all(isinstance(e, np.ndarray) and e.shape == data_to_save[0].shape for e in data_to_save):
                    data_to_npz['pooled_embeddings'] = np.stack(data_to_save, axis=0)
                else: # Save as individual arrays if shapes differ or not all ndarray
                    for i, emb in enumerate(data_to_save):
                        data_to_npz[f'pooled_embedding_{i}'] = emb
            elif isinstance(data_to_save, dict): # Layer-wise hidden states and/or attentions
                for key, list_of_arrays in data_to_save.items():
                    if key == "attentions" and isinstance(list_of_arrays, dict): # Nested dict for attentions
                        for attention_layer_key, attention_list in list_of_arrays.items():
                            if attention_list and all(isinstance(a, np.ndarray) for a in attention_list):
                                # Cannot easily stack attentions due to varying seq_len per batch item
                                # Save them as a list of arrays under one key, or individual arrays
                                # For simplicity, let's save list of arrays (might need allow_pickle=True on load)
                                # Or better, save each item in the list of arrays as arr_0, arr_1 within a "group"
                                # np.savez doesn't directly support groups. So, flatten the key.
                                for idx, arr_item in enumerate(attention_list):
                                     data_to_npz[f"{key}_{attention_layer_key}_item_{idx}"] = arr_item
                            elif attention_list: # mixed types or other issue
                                logger.warning(f"Could not directly save {key}_{attention_layer_key} as stacked array. Saving as object or skipping.")
                                data_to_npz[f"{key}_{attention_layer_key}_obj"] = np.array(attention_list, dtype=object)

                    elif isinstance(list_of_arrays, list) and list_of_arrays:
                        # For hidden states, each item in list_of_arrays is (seq_len, hidden_size)
                        # These can't be stacked directly if seq_len varies per input text.
                        # So, save each text's layer representation individually.
                        for idx, arr_item in enumerate(list_of_arrays):
                             data_to_npz[f"{key}_item_{idx}"] = arr_item
            
            metadata_str_array = np.array(json.dumps(metadata, indent=4), dtype=object)
            data_to_npz['metadata'] = metadata_str_array
            np.savez_compressed(output_path, **data_to_npz)
            logger.info(f"Corpus {corpus_label}: Data and metadata successfully saved to {output_path} (NPZ).")
        except Exception as e:
            logger.error(f"Corpus {corpus_label}: Error saving NPZ file to {output_path}: {e}", exc_info=True)
            raise
    elif output_format == "hdf5":
        if not H5PY_AVAILABLE:
            # ... (H5PY not available error) ...
            logger.error(f"Corpus {corpus_label}: h5py library is not installed, but HDF5 output format was selected. Please install h5py or choose 'npz' format.")
            raise ImportError("h5py is required for HDF5 output format.")
        try:
            with h5py.File(output_path, 'w') as hf:
                if isinstance(data_to_save, list): # Pooled embeddings
                    if data_to_save and all(isinstance(e, np.ndarray) and e.shape == data_to_save[0].shape for e in data_to_save):
                        hf.create_dataset('pooled_embeddings', data=np.stack(data_to_save, axis=0), compression="gzip")
                    else: # Save individually
                        pooled_grp = hf.create_group("pooled_embeddings")
                        for i, emb in enumerate(data_to_save):
                            pooled_grp.create_dataset(f"embedding_{i}", data=emb, compression="gzip")
                elif isinstance(data_to_save, dict): # Layer-wise
                    for key, list_of_arrays_or_dict in data_to_save.items():
                        grp = hf.create_group(key)
                        if key == "attentions" and isinstance(list_of_arrays_or_dict, dict):
                            for attention_layer_key, attention_list in list_of_arrays_or_dict.items():
                                attention_grp = grp.create_group(attention_layer_key)
                                if attention_list:
                                    # Store as a list of datasets (variable length for seq_len)
                                    for idx, arr_item in enumerate(attention_list):
                                        attention_grp.create_dataset(f"item_{idx}", data=arr_item, compression="gzip")
                        elif isinstance(list_of_arrays_or_dict, list) and list_of_arrays_or_dict:
                            # For hidden states, list_of_arrays is a list of (seq_len, hidden_dim)
                            for idx, arr_item in enumerate(list_of_arrays_or_dict):
                                grp.create_dataset(f"item_{idx}", data=arr_item, compression="gzip")
                
                # Save metadata as attributes
                for meta_key, meta_value in metadata.items():
                    try:
                        if isinstance(meta_value, (dict, list, tuple)): hf.attrs[meta_key] = json.dumps(meta_value)
                        elif meta_value is None: hf.attrs[meta_key] = "None" 
                        else: hf.attrs[meta_key] = meta_value
                    except TypeError as te:
                        logger.warning(f"Corpus {corpus_label}: Could not serialize metadata key '{meta_key}' for HDF5. Storing as string. Error: {te}")
                        hf.attrs[meta_key] = str(meta_value)
            logger.info(f"Corpus {corpus_label}: Data and metadata successfully saved to {output_path} (HDF5).")
        except Exception as e:
            logger.error(f"Corpus {corpus_label}: Error saving HDF5 file to {output_path}: {e}", exc_info=True)
            raise
    else:
        logger.error(f"Corpus {corpus_label}: Unsupported output format: {output_format}. Please choose 'npz' or 'hdf5'.")
        raise ValueError(f"Unsupported output format: {output_format}")

# --- if __name__ == '__main__': block ---
if __name__ == '__main__':
    overall_script_start_time = time.time()
    parser = argparse.ArgumentParser(description="Enhanced ETP Embedding Extractor")
    # ... (previous arguments)
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
    parser.add_argument("--pooling_strategy", type=str, default="mean", choices=POOLING_STRATEGIES, help="Pooling strategy. Use 'none' for all_hidden_states.")
    parser.add_argument("--trust_remote_code", type=lambda x: (str(x).lower() == 'true'), default=True, help="Trust remote code for AutoModel/Tokenizer.")
    parser.add_argument("--use_bert_tiny_for_test", action="store_true", help="Use prajjwal1/bert-tiny for quick testing.")
    parser.add_argument("--skip_if_output_exists", type=lambda x: (str(x).lower() == 'true'), default=True, help="Skip processing if output file already exists.") # New
    parser.add_argument("--extract_all_hidden_states", type=lambda x: (str(x).lower() == 'true'), default=False, help="Extract hidden states from all layers instead of just a pooled embedding.") # New
    parser.add_argument("--extract_attentions", type=lambda x: (str(x).lower() == 'true'), default=False, help="Extract attention weights from all layers.") # New

    args = parser.parse_args()
    # ... (rest of arg parsing, device selection, H5PY check) ...
    logger.info("Starting ETP Embedding Extractor script.")
    logger.info(f"Run arguments: {vars(args)}")


    if args.use_bert_tiny_for_test:
        logger.info("Overriding model to 'prajjwal1/bert-tiny' for quick testing.")
        args.model_name_or_path = "prajjwal1/bert-tiny"
        if args.max_length > 128 or args.max_length == 0 :
             logger.info(f"Adjusting max_length from {args.max_length} to 128 for bert-tiny test.")
             args.max_length = 128
        # For bert-tiny, all_hidden_states and attentions are usually fine
        # but pooling might need adjustment if 'none' isn't default for these flags.
        if args.extract_all_hidden_states and args.pooling_strategy != "none":
            logger.info(f"extract_all_hidden_states is True, setting pooling_strategy to 'none'. Was: {args.pooling_strategy}")
            args.pooling_strategy = "none"


    selected_device = get_device(args.device)
    logger.info(f"Using device: {selected_device}")

    if not H5PY_AVAILABLE and args.output_format == "hdf5":
        logger.error("HDF5 output format selected, but the 'h5py' library is not available. "
                     "Please install it (e.g., 'pip install h5py') or choose 'npz' format.")
        sys.exit(1)

    # Ensure pooling strategy is 'none' if extracting all hidden states
    if args.extract_all_hidden_states and args.pooling_strategy != "none":
        logger.warning(f"Extracting all hidden states, but pooling strategy is '{args.pooling_strategy}'. "
                       "Changing pooling strategy to 'none' to save full hidden states.")
        args.pooling_strategy = "none"


    corpora_to_process = []
    if args.output_path_A:
        corpora_to_process.append(("A", args.texts_file_A, args.num_dummy_texts_A, args.output_path_A))
    else:
        logger.error("--output_path_A is required. Please specify an output path for Corpus A.")
        sys.exit(1)

    if args.output_path_B:
        corpora_to_process.append(("B", args.texts_file_B, args.num_dummy_texts_B, args.output_path_B))

    model_transformers_version_config_for_error_msg = "N/A" 
    try:
        temp_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
        model_transformers_version_config_for_error_msg = getattr(temp_config, "transformers_version", "N/A")
    except Exception as e_cfg:
        logger.warning(f"Could not pre-fetch model config for {args.model_name_or_path} to get transformers_version: {e_cfg}")

    for corpus_label, texts_file, num_dummy, output_path_val in corpora_to_process:
        corpus_process_start_time = time.time()
        logger.info(f"--- Starting processing for Corpus {corpus_label} ---")

        output_file = Path(output_path_val)
        if args.skip_if_output_exists and output_file.exists() and output_file.is_file() and output_file.stat().st_size > 0:
            logger.info(f"Output file {output_file} already exists and is not empty. Skipping Corpus {corpus_label} processing.")
            continue
        
        # ... (rest of text loading logic: texts_current_corpus) ...
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


        logger.info(f"Extracting data for Corpus {corpus_label} ({len(texts_current_corpus)} texts)... "
                    f"All Hidden States: {args.extract_all_hidden_states}, Attentions: {args.extract_attentions}")
        try:
            extracted_data_corpus, metadata_corpus = extract_embeddings(
                texts_current_corpus, args.model_name_or_path, selected_device, args.batch_size,
                args.max_length, args.pooling_strategy, args.trust_remote_code,
                corpus_label_for_logging=corpus_label,
                extract_all_hidden_states=args.extract_all_hidden_states, # Pass flag
                extract_attentions=args.extract_attentions             # Pass flag
            )
            # ... (metadata update and saving logic, same as before) ...
            if "transformers_version_config_from_model" not in metadata_corpus:
                 metadata_corpus["transformers_version_config_from_model"] = model_transformers_version_config_for_error_msg

            # Check if anything was actually extracted before saving
            data_is_present = False
            if isinstance(extracted_data_corpus, list) and extracted_data_corpus:
                data_is_present = True
            elif isinstance(extracted_data_corpus, dict):
                for _key, _val_list_or_dict in extracted_data_corpus.items():
                    if isinstance(_val_list_or_dict, list) and _val_list_or_dict:
                        data_is_present = True; break
                    elif isinstance(_val_list_or_dict, dict): # for nested attentions
                        for _sub_key, _sub_val_list in _val_list_or_dict.items():
                            if isinstance(_sub_val_list, list) and _sub_val_list:
                                data_is_present = True; break
                        if data_is_present: break
            
            if data_is_present:
                save_embeddings_with_metadata(extracted_data_corpus, metadata_corpus, output_path_val, args.output_format)
            else:
                logger.warning(f"No actual data (embeddings/states/attentions) were extracted for Corpus {corpus_label}. Only metadata might be saved if applicable, or skipping save for {output_path_val}.")
                # Optionally save metadata only, as the save function handles this
                save_embeddings_with_metadata({}, metadata_corpus, output_path_val, args.output_format)


            corpus_process_end_time = time.time()
            logger.info(f"--- Corpus {corpus_label} processing finished in {corpus_process_end_time - corpus_process_start_time:.2f}s. Data potentially saved to {output_path_val} ---")

        except Exception as e:
            # ... (error handling logic, same as before) ...
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