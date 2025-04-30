# =====================================================================
# WuBu Nesting Model Trainer (Nucleotide Adaptation with BioPython)
# =====================================================================
# Description:
# This script trains a sequence model based on the WuBu Nesting
# architecture, adapted for nucleotide sequences (e.g., mRNA).
# It uses a hyperbolic geometry core with tangent space bridges
# for encoder/decoder components and Riemannian optimization.
# Includes automatic dataset download/processing for select datasets
# (RNAcentral FASTA, Rfam Stockholm via BioPython) and options for
# combining datasets and optimizing batching for large VRAM.
# =====================================================================

# =====================================================================
# Python Imports and Setup
# =====================================================================
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, DistributedSampler
import numpy as np
orig_np_load = np.load

# Define a custom load function that handles the allow_pickle parameter
def custom_np_load(*args, **kwargs):
    allow_pickle = kwargs.pop('allow_pickle', False)
    mmap_mode = kwargs.pop('mmap_mode', None)
    
    # If mmap_mode is specified, use memmap
    if mmap_mode is not None:
        kwargs['mode'] = mmap_mode  # open_memmap uses 'mode' instead of 'mmap_mode'
        return np.lib.format.open_memmap(*args, **kwargs)
    else:
        # Use the original np.load with allow_pickle
        return orig_np_load(*args, allow_pickle=allow_pickle, **kwargs)

# Replace np.load with our custom function
np.load = custom_np_load
import math
import random
import argparse
import logging
import time
import contextlib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any, Iterable
from collections import deque, defaultdict
import gc
import os
import socket
import platform
from torch.nn.parallel import DistributedDataParallel as DDP # Use alias for clarity
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank, get_world_size
from torch import amp
from dataclasses import dataclass, field
import itertools
from tqdm import tqdm
import inspect
import functools
import requests # For downloading data
import gzip # For handling .gz files
import shutil # For file operations
import json # For combined dataset metadata

# --- BioPython Import ---
try:
    from Bio import SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    SeqIO = None
    BIOPYTHON_AVAILABLE = False
    # Logger not configured yet, print warning directly
    print("WARNING: BioPython not found (`pip install biopython`). Stockholm format parsing will not be available.")

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Setup logger - Initial basic config, will be refined in main
logger = logging.getLogger("WuBuNestingTrainer") # Use WuBu naming
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True)

# Constants
EPS = 1e-7 # Small epsilon for numerical stability
DATA_DIR = "data" # Directory to store downloaded and processed data
NUCLEOTIDE_MAP = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4, '-': 4} # Treat gaps/unknown as index 4
NUCLEOTIDE_VOCAB_SIZE = 5 # A, U, G, C, N(-) - Ensure this matches model layers
COMBINED_DATA_INFO_FILE = "combined_rna_dataset_info.json" # Metadata filename for combined dataset


# =====================================================================
# Data Preparation Utilities
# =====================================================================

def download_file(url: str, dest_path: str, chunk_size=8192):
    """Downloads a file from a URL to a destination path with progress."""
    temp_dest_path = dest_path + ".part" # Download to temporary file
    try:
        logger.info(f"Downloading {url} to {dest_path}...")
        response = requests.get(url, stream=True, timeout=600) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        total_size = int(response.headers.get('content-length', 0))
        with open(temp_dest_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path), total=total_size, unit='iB', unit_scale=True, unit_divisor=1024, leave=False
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk: # filter out keep-alive new chunks
                    size = f.write(chunk)
                    bar.update(size)
        shutil.move(temp_dest_path, dest_path) # Rename to final destination atomically
        logger.info(f"Download complete: {dest_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed for {url}: {e}")
        if os.path.exists(temp_dest_path):
            try:
                os.remove(temp_dest_path)
            except OSError:
                logger.warning(f"Could not remove partial download {temp_dest_path}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during download of {url}: {e}")
        if os.path.exists(temp_dest_path):
            try:
                os.remove(temp_dest_path)
            except OSError:
                logger.warning(f"Could not remove partial download {temp_dest_path}")
        return False

def parse_fasta_to_indices(fasta_path: str, nucleotide_map: Dict[str, int], unknown_idx: int) -> List[int]:
    """Parses a FASTA file (potentially gzipped) and returns a single list of nucleotide indices."""
    all_indices = []
    open_func = gzip.open if fasta_path.endswith(".gz") else open
    logger.info(f"Parsing FASTA {fasta_path}...")
    try:
        with open_func(fasta_path, 'rt', errors='ignore') as f: # Read as text, ignore decoding errors
            current_seq_indices = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_seq_indices:
                        all_indices.extend(current_seq_indices)
                    current_seq_indices = []
                else:
                    line_clean = line.upper().replace("T", "U") # Ensure uppercase RNA
                    line_indices = [nucleotide_map.get(nt, unknown_idx) for nt in line_clean] # Map or use unknown_idx
                    current_seq_indices.extend(line_indices)
            if current_seq_indices:
                all_indices.extend(current_seq_indices) # Add the last sequence
        logger.info(f"Parsed {len(all_indices):,} total nucleotide indices from FASTA.")
        return all_indices
    except FileNotFoundError:
        logger.error(f"FASTA file not found during parsing: {fasta_path}")
        raise
    except Exception as e:
        logger.error(f"Error parsing FASTA file {fasta_path}: {e}", exc_info=True)
        raise

def parse_stockholm_biopython(stockholm_path: str, nucleotide_map: Dict[str, int], unknown_idx: int) -> List[int]:
    """
    Parses a Stockholm file (potentially gzipped) using BioPython SeqIO
    and returns a single list of nucleotide indices, concatenating sequences.
    """
    if not BIOPYTHON_AVAILABLE:
        raise ImportError("BioPython is required for parsing Stockholm files. Please install it (`pip install biopython`).")

    all_indices = []
    open_func = gzip.open if stockholm_path.endswith(".gz") else open
    logger.info(f"Parsing Stockholm {stockholm_path} using BioPython...")
    try:
        # Open with gzip if needed, read as text
        with open_func(stockholm_path, 'rt', errors='ignore') as handle:
            # Parse records one by one
            for record in SeqIO.parse(handle, "stockholm"):
                seq_str = str(record.seq).upper().replace("T", "U") # Get sequence, ensure uppercase RNA
                record_indices = [nucleotide_map.get(nt, unknown_idx) for nt in seq_str]
                all_indices.extend(record_indices)
                # Optional: Add separator between sequences if desired
        logger.info(f"Parsed {len(all_indices):,} total nucleotide indices from Stockholm (BioPython).")
        return all_indices
    except FileNotFoundError:
        logger.error(f"Stockholm file not found during parsing: {stockholm_path}")
        raise
    except Exception as e:
        logger.error(f"Error parsing Stockholm file {stockholm_path} with BioPython: {e}", exc_info=True)
        raise

def prepare_dataset(dataset_name: str, data_dir: str) -> Optional[Tuple[str, int]]:
    """
    Checks if processed dataset exists, otherwise downloads raw data,
    processes it into nucleotide indices (with memory-aware FASTA parsing),
    and saves to the final .npy path using np.save.

    Returns:
        Tuple (path_to_npy, num_indices) or None if preparation fails.
    """
    os.makedirs(data_dir, exist_ok=True)
    processed_npy_path = os.path.join(data_dir, f"{dataset_name}_indices.npy")
    unknown_idx = NUCLEOTIDE_MAP.get('N', NUCLEOTIDE_VOCAB_SIZE - 1)

    # --- Check if already processed and valid ---
    if os.path.exists(processed_npy_path):
        mmap_obj = None
        try:
            mmap_obj = np.lib.format.open_memmap(processed_npy_path, mode='r')
            size = mmap_obj.shape[0]
            if size > 0:
                logger.info(f"Processed dataset found: {processed_npy_path} (Size: {size:,})")
                return processed_npy_path, size
            else:
                logger.warning(f"Existing processed file is empty: {processed_npy_path}. Re-processing.")
                del mmap_obj; gc.collect() # Release handle before remove
                _try_remove_file(processed_npy_path)
        except ValueError as ve: # Handle potential header error on check
             if "magic string" in str(ve):
                 logger.warning(f"Existing file '{processed_npy_path}' has invalid header. Re-processing.")
             else:
                 logger.warning(f"Error checking existing file {processed_npy_path} (ValueError): {ve}. Re-processing.")
             if mmap_obj is not None: del mmap_obj; gc.collect()
             _try_remove_file(processed_npy_path)
        except Exception as e:
            logger.warning(f"Error checking existing file {processed_npy_path}: {e}. Re-processing.")
            if mmap_obj is not None: del mmap_obj; gc.collect()
            _try_remove_file(processed_npy_path)
        finally:
            if mmap_obj is not None: del mmap_obj; gc.collect()


    logger.info(f"Processed dataset '{processed_npy_path}' not found or invalid. Attempting to download and process...")

    # --- Dataset Definitions (ensure ENSEMBL_RELEASE is correct) ---
    ENSEMBL_RELEASE = "113"
    datasets_info = {
        "rfam_seed": {
            "url": "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz",
            "raw_filename": "Rfam.seed.gz", "format": "stockholm",
            "parser": parse_stockholm_biopython, "requires_biopython": True },
#        "rnacentral_active": {
#            "url": "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz",
#            "raw_filename": "rnacentral_active.fasta.gz", "format": "fasta",
#            "warning": "This dataset is very large (~50GB compressed)!" },
        "gencode_human_cdna": {
            "url": f"https://ftp.ensembl.org/pub/release-{ENSEMBL_RELEASE}/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz",
            "raw_filename": f"Homo_sapiens.GRCh38.cdna.all.release{ENSEMBL_RELEASE}.fa.gz",
            "format": "fasta", "warning": "Large cDNA dataset." },
        "refseq_human_rna": {
            "url": "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_rna.fna.gz",
            "raw_filename": "GRCh38_latest_rna.fna.gz", "format": "fasta",
            "warning": "Various RNA types from RefSeq." },
    }

    if dataset_name not in datasets_info:
        logger.error(f"Unknown dataset name: '{dataset_name}'. Available: {list(datasets_info.keys())}")
        return None

    info = datasets_info[dataset_name]
    raw_file_path = os.path.join(data_dir, info["raw_filename"])

    if info.get("requires_biopython", False) and not BIOPYTHON_AVAILABLE:
         logger.error(f"Dataset '{dataset_name}' requires BioPython, but it was not found.")
         return None
    if "warning" in info: logger.warning(info["warning"])

    if not os.path.exists(raw_file_path):
        if not download_file(info["url"], raw_file_path):
            logger.error(f"Failed download raw data for {dataset_name}")
            return None

    index_array = None
    num_indices = 0
    temp_chunk_file_path = processed_npy_path + ".tmp_parser" # Define here for cleanup

    try:
        parser_func = info.get("parser", None)
        if parser_func:
            # Assumes parser returns a list
            all_indices = parser_func(raw_file_path, NUCLEOTIDE_MAP, unknown_idx)
            if not all_indices:
                logger.error(f"Parser function for {dataset_name} returned no indices.")
                return None
            logger.info("Converting parsed indices list to NumPy array...")
            final_dtype = np.uint8 if NUCLEOTIDE_VOCAB_SIZE <= 256 else np.int16
            index_array = np.array(all_indices, dtype=final_dtype)
            num_indices = index_array.shape[0]
            logger.info(f"NumPy array created from list (Shape: {index_array.shape}, Size: {index_array.nbytes / (1024*1024):.2f} MB, DType: {final_dtype}).")
            del all_indices; gc.collect()

        elif info["format"] == "fasta":
            # --- Memory-Aware FASTA Parsing using temp file ---
            final_dtype = np.uint8 if NUCLEOTIDE_VOCAB_SIZE <= 256 else np.int16
            total_parsed_indices = 0
            chunk_list = []
            MAX_CHUNK_LIST_MEM_MB = 500 # Adjust as needed

            logger.info(f"Parsing FASTA {raw_file_path} (Memory-Aware)...")
            open_func = gzip.open if raw_file_path.endswith(".gz") else open

            # Ensure temp file doesn't exist from previous failed run
            if os.path.exists(temp_chunk_file_path):
                 _try_remove_file(temp_chunk_file_path)

            try:
                with open_func(raw_file_path, 'rt', errors='ignore') as f:
                    current_seq_indices = []
                    processed_lines = 0
                    for line in f:
                        processed_lines += 1
                        line = line.strip()
                        if not line: continue
                        if line.startswith(">"):
                            if current_seq_indices:
                                chunk_list.extend(current_seq_indices)
                                current_mem_bytes = sys.getsizeof(chunk_list) # Rough estimate
                                if current_mem_bytes > MAX_CHUNK_LIST_MEM_MB * 1024 * 1024:
                                     logger.info(f"Writing intermediate chunk ({len(chunk_list):,} indices, ~{current_mem_bytes / (1024*1024):.1f}MB list) to disk...")
                                     chunk_array = np.array(chunk_list, dtype=final_dtype)
                                     mode = 'ab' if os.path.exists(temp_chunk_file_path) else 'wb'
                                     with open(temp_chunk_file_path, mode) as chunk_f: chunk_array.tofile(chunk_f)
                                     total_parsed_indices += len(chunk_list)
                                     del chunk_list[:] # Clear list
                                     del chunk_array; gc.collect()
                            current_seq_indices = []
                        else:
                            line_clean = line.upper().replace("T", "U")
                            line_indices = [NUCLEOTIDE_MAP.get(nt, unknown_idx) for nt in line_clean]
                            current_seq_indices.extend(line_indices)
                        # Log progress occasionally
                        if processed_lines % 5000000 == 0:
                            logger.debug(f" ...parsed {processed_lines:,} lines...")

                    if current_seq_indices:
                        chunk_list.extend(current_seq_indices)

                if chunk_list:
                    logger.info(f"Writing final chunk ({len(chunk_list):,} indices) to disk...")
                    chunk_array = np.array(chunk_list, dtype=final_dtype)
                    mode = 'ab' if os.path.exists(temp_chunk_file_path) else 'wb'
                    with open(temp_chunk_file_path, mode) as chunk_f: chunk_array.tofile(chunk_f)
                    total_parsed_indices += len(chunk_list)
                    del chunk_list[:]
                    del chunk_array; gc.collect()

                logger.info(f"Finished parsing FASTA. Total indices parsed to temp file: {total_parsed_indices:,}")

                if total_parsed_indices == 0:
                    logger.error(f"Parsing resulted in zero indices for {dataset_name}.")
                    _try_remove_file(temp_chunk_file_path)
                    return None

                logger.info(f"Loading indices from temp file '{temp_chunk_file_path}'...")
                index_array = np.fromfile(temp_chunk_file_path, dtype=final_dtype)
                num_indices = index_array.shape[0]
                if num_indices != total_parsed_indices:
                     logger.warning(f"Size mismatch between parsed count ({total_parsed_indices:,}) and loaded array ({num_indices:,})")
                logger.info(f"Loaded array (Shape: {index_array.shape}, Size: {index_array.nbytes / (1024*1024):.2f} MB)")

                _try_remove_file(temp_chunk_file_path) # Clean up temp file

            except MemoryError as me_fasta:
                logger.error(f"MemoryError occurred during memory-aware FASTA parsing for {dataset_name}: {me_fasta}. File may be too large or chunk limit too high.", exc_info=True)
                _try_remove_file(temp_chunk_file_path)
                return None
            except Exception as e_fasta:
                logger.error(f"Error during memory-aware FASTA parsing {raw_file_path}: {e_fasta}", exc_info=True)
                _try_remove_file(temp_chunk_file_path)
                raise # Re-raise other errors
            # --- End Memory-Aware FASTA Parsing ---
        else:
            logger.error(f"No parser defined for format: {info['format']}")
            return None

        # --- Saving Step (Common path) ---
        if index_array is None or index_array.size == 0:
             logger.error(f"Processing resulted in zero nucleotide indices for {dataset_name}.")
             return None
        num_indices = index_array.shape[0] # Get size before potential deletion

        if os.path.exists(processed_npy_path):
            logger.warning(f"Target file {processed_npy_path} exists. Deleting before save.")
            if not _try_remove_file(processed_npy_path):
                 logger.error(f"Could not delete existing target file {processed_npy_path}. Cannot proceed.")
                 del index_array; gc.collect(); return None

        logger.info(f"Saving final array using np.save to: {processed_npy_path} (Size: {num_indices:,})")
        np.save(processed_npy_path, index_array)
        logger.info(f"Successfully completed np.save call.")

        # Verification
        mmap_verify = None
        try:
             mmap_verify = np.lib.format.open_memmap(processed_npy_path, mode='r')
             saved_size = mmap_verify.shape[0]
             if saved_size != num_indices:
                 raise ValueError(f"Saved file size {saved_size:,} != expected {num_indices:,}")
             logger.info(f"Direct save successful and verified. Dataset available at: {processed_npy_path}")
        except Exception as verify_err:
             logger.error(f"CRITICAL FAILURE: Error verifying saved file {processed_npy_path}: {verify_err}")
             if mmap_verify is not None: del mmap_verify; gc.collect()
             _try_remove_file(processed_npy_path)
             del index_array; gc.collect(); return None
        finally:
             if mmap_verify is not None: del mmap_verify; gc.collect()

        del index_array # Free memory after successful save and verification
        gc.collect()
        return processed_npy_path, num_indices

    except Exception as e:
        logger.error(f"Outer error during processing/save for {dataset_name} to {processed_npy_path}: {e}", exc_info=True)
        if index_array is not None: del index_array; gc.collect()
        _try_remove_file(temp_chunk_file_path) # Ensure parser temp file is cleaned
        _try_remove_file(processed_npy_path) # Ensure final file is cleaned on error
        return None


def _try_remove_file(filepath: str, max_retries: int = 5, initial_delay: float = 0.2):
    """Tries to remove a file with delays and retries, logging results."""
    if not os.path.exists(filepath):
        logger.debug(f"File already removed or does not exist: {filepath}")
        return True

    retry_delay = initial_delay
    for attempt in range(max_retries):
        try:
            os.remove(filepath)
            logger.info(f"Successfully removed file '{filepath}' on attempt {attempt + 1}.")
            return True
        except PermissionError as pe:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to remove '{filepath}': {pe}. Retrying after {retry_delay:.2f}s...")
            time.sleep(retry_delay)
            retry_delay *= 1.5 # Exponential backoff (optional)
        except FileNotFoundError:
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: File '{filepath}' not found during removal attempt.")
            return True # Treat as success if not found anymore
        except Exception as remove_err:
            logger.error(f"Unexpected error removing '{filepath}' on attempt {attempt + 1}: {remove_err}", exc_info=True)
            return False # Non-permission error, stop retrying

    logger.error(f"Failed to remove '{filepath}' after {max_retries} attempts due to persistent lock or other error.")
    return False

def prepare_combined_dataset(data_dir: str, max_datasets: Optional[int] = None) -> Optional[str]:
    """
    Prepares a combined dataset using memory mapping. Determines total size first,
    creates the final file once, then appends data chunk by chunk.

    Args:
        data_dir: Directory for data storage and processing.
        max_datasets: Maximum number of datasets to include. None for all.

    Returns:
        Path to the combined dataset (.npy) or None if preparation fails.
    """
    os.makedirs(data_dir, exist_ok=True)
    combined_npy_path = os.path.join(data_dir, "combined_rna_indices.npy")
    metadata_path = os.path.join(data_dir, COMBINED_DATA_INFO_FILE)
    temp_combined_path = combined_npy_path + ".building" # Use a temp name during build

    # --- Clean up previous temp build file ---
    if os.path.exists(temp_combined_path):
        logger.warning(f"Removing previous temporary build file: {temp_combined_path}")
        if not _try_remove_file(temp_combined_path): return None

    # --- Check if final combined dataset and metadata already exist and are valid ---
    if os.path.exists(combined_npy_path) and os.path.exists(metadata_path):
        logger.info(f"Combined dataset found: {combined_npy_path}")
        logger.info(f"Metadata found: {metadata_path}")
        mmap_obj = None
        try:
            with open(metadata_path, 'r') as f: meta = json.load(f)
            mmap_obj = np.lib.format.open_memmap(combined_npy_path, mode='r')
            if meta.get('total_indices') == mmap_obj.shape[0]:
                logger.info("Metadata size matches NPY file size.")
                return combined_npy_path
            else:
                logger.warning(f"Metadata/NPY size mismatch. Rebuilding.")
                if mmap_obj is not None: del mmap_obj; gc.collect() # Close before removing
                if os.path.exists(combined_npy_path): _try_remove_file(combined_npy_path)
                if os.path.exists(metadata_path): _try_remove_file(metadata_path)
        except ValueError as ve: # Handle potential header error on check
             if "magic string" in str(ve):
                 logger.warning(f"Existing file '{combined_npy_path}' has invalid header. Rebuilding.")
             else:
                 logger.warning(f"Error checking existing combined data (ValueError): {ve}. Rebuilding.")
             if mmap_obj is not None: del mmap_obj; gc.collect()
             if os.path.exists(combined_npy_path): _try_remove_file(combined_npy_path)
             if os.path.exists(metadata_path): _try_remove_file(metadata_path)
        except Exception as check_err:
            logger.warning(f"Error checking existing combined data: {check_err}. Rebuilding.")
            if mmap_obj is not None: del mmap_obj; gc.collect()
            if os.path.exists(combined_npy_path): _try_remove_file(combined_npy_path)
            if os.path.exists(metadata_path): _try_remove_file(metadata_path)
        finally:
            if mmap_obj is not None: del mmap_obj; gc.collect()


    logger.info(f"Combined dataset '{combined_npy_path}' or metadata not found/invalid. Attempting creation...")

    # --- List of datasets to combine ---
    dataset_names_to_process = ["rfam_seed", "gencode_human_cdna", "refseq_human_rna"]
    if max_datasets is not None and max_datasets > 0:
        dataset_names_to_process = dataset_names_to_process[:max_datasets]
        logger.warning(f"Limiting combined dataset to the first {max_datasets} sources: {dataset_names_to_process}")

    # --- Stage 1: Prepare individual datasets and calculate total size ---
    prepared_datasets_info = [] # List of tuples: (name, path, size)
    total_expected_indices = 0
    logger.info("--- Stage 1: Preparing individual datasets and calculating total size ---")
    all_datasets_prepared = True
    for dataset_name in dataset_names_to_process:
        logger.info(f"Preparing '{dataset_name}'...")
        prep_result = prepare_dataset(dataset_name, data_dir)
        if prep_result is not None:
            path, size = prep_result
            if size > 0:
                prepared_datasets_info.append((dataset_name, path, size))
                total_expected_indices += size
                logger.info(f"'{dataset_name}' ready (Size: {size:,}). Cumulative size: {total_expected_indices:,}")
            else:
                logger.warning(f"Dataset '{dataset_name}' prepared but is empty. Skipping.")
        else:
            logger.warning(f"Failed to prepare dataset '{dataset_name}'. Skipping.")
            # Mark as failed only if it was critical? For now, allow skipping non-critical ones.
            # Example: If rnacentral fails due to memory, maybe proceed with others?
            # Let's proceed but log the failure.
            all_datasets_prepared = False # Indicate at least one failed

    if not prepared_datasets_info or total_expected_indices == 0:
        logger.error("No valid datasets could be prepared or all were empty. Cannot create combined dataset.")
        return None
    elif not all_datasets_prepared:
         logger.warning("One or more datasets failed preparation. Combined dataset will be incomplete.")


    logger.info(f"--- Stage 1 Complete: Total expected indices: {total_expected_indices:,} from {len(prepared_datasets_info)} dataset(s) ---")

    # --- Stage 2: Create final memmap file and append data ---
    logger.info(f"--- Stage 2: Creating final file '{temp_combined_path}' and appending data ---")
    final_dtype = np.uint8 if NUCLEOTIDE_VOCAB_SIZE <= 256 else np.int16
    combined_mmap = None
    dataset_boundaries = []
    current_offset = 0

    try:
        logger.info(f"Creating target memmap file: {temp_combined_path} (Size: {total_expected_indices:,}, DType: {final_dtype})")
        combined_mmap = np.lib.format.open_memmap(temp_combined_path, mode='w+', dtype=final_dtype, shape=(total_expected_indices,))

        chunk_size = 10_000_000
        for name, path, size in prepared_datasets_info:
            logger.info(f"Appending '{name}' ({size:,} indices) from {path}...")
            individual_mmap = None
            try:
                individual_mmap = np.load(path, mmap_mode='r')
                if individual_mmap.shape[0] != size:
                     logger.warning(f"Size mismatch for '{name}': Expected {size:,}, Found {individual_mmap.shape[0]:,}. Using found size.")
                     size = individual_mmap.shape[0] # Adjust size if it changed

                append_start_index = current_offset
                append_end_index = current_offset + size
                if append_end_index > total_expected_indices:
                    logger.error(f"Append error for '{name}': Would write past end of allocated file ({append_end_index} > {total_expected_indices}). Skipping.")
                    continue # Skip this dataset if something went wrong with size calc

                for i in tqdm(range(0, size, chunk_size), desc=f"Appending {name}", leave=False, unit="chunk"):
                    start = i
                    end = min(i + chunk_size, size)
                    data_chunk = individual_mmap[start:end]
                    combined_mmap[append_start_index + start : append_start_index + end] = data_chunk.astype(final_dtype)

                dataset_boundaries.append((name, current_offset, append_end_index))
                current_offset = append_end_index
                logger.info(f"Finished appending '{name}'. Current offset: {current_offset:,}")
                combined_mmap.flush()

            except Exception as append_err:
                 logger.error(f"Error appending data from '{name}': {append_err}", exc_info=True)
                 # Continue to next dataset? Or abort? Let's abort for safety.
                 raise append_err
            finally:
                 if individual_mmap is not None: del individual_mmap; gc.collect()

        logger.info("All datasets appended. Performing final flush.")
        combined_mmap.flush()
        final_written_size = combined_mmap.shape[0]
        # Close handle before rename
        del combined_mmap; combined_mmap = None; gc.collect()

        if final_written_size != total_expected_indices:
             logger.error(f"Final written size ({final_written_size:,}) != expected size ({total_expected_indices:,}). Aborting.")
             _try_remove_file(temp_combined_path)
             return None
        if current_offset != total_expected_indices:
             logger.error(f"Final write offset ({current_offset:,}) != expected size ({total_expected_indices:,}). Potential issue. Aborting.")
             _try_remove_file(temp_combined_path)
             return None


        logger.info(f"Renaming temporary file {temp_combined_path} to final path {combined_npy_path}")
        os.replace(temp_combined_path, combined_npy_path) # Atomic rename

        # --- Verification step after rename ---
        verify_mmap = None
        try:
            verify_mmap = np.lib.format.open_memmap(combined_npy_path, mode='r')
            if verify_mmap.shape[0] != total_expected_indices:
                 raise ValueError(f"Verification failed: Final file size {verify_mmap.shape[0]:,} != expected {total_expected_indices:,}")
            logger.info("Final file verified successfully.")
        except Exception as final_verify_err:
            logger.error(f"CRITICAL FAILURE: Error verifying final combined file {combined_npy_path}: {final_verify_err}")
            if verify_mmap is not None: del verify_mmap; gc.collect()
            _try_remove_file(combined_npy_path) # Clean up potentially corrupted final file
            return None
        finally:
            if verify_mmap is not None: del verify_mmap; gc.collect()
        # --- End Verification ---


        # Save metadata
        metadata = {
            "combined_npy_path": os.path.basename(combined_npy_path),
            "dataset_boundaries": dataset_boundaries,
            "total_indices": total_expected_indices,
            "creation_time": datetime.now().isoformat(),
            "included_datasets": [name for name, _, _ in dataset_boundaries]
        }
        logger.debug(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Combined dataset saved successfully: {combined_npy_path}")
        logger.info(f"Metadata saved: {metadata_path}")
        return combined_npy_path

    except Exception as build_err:
        logger.error(f"Error during Stage 2 (building combined file): {build_err}", exc_info=True)
        if combined_mmap is not None: del combined_mmap; combined_mmap = None; gc.collect()
        _try_remove_file(temp_combined_path)
        if os.path.exists(combined_npy_path): _try_remove_file(combined_npy_path)
        if os.path.exists(metadata_path): _try_remove_file(metadata_path)
        return None


# =====================================================================
# Dataset Classes
# =====================================================================
class WuBuNestingDataset(IterableDataset):
    """ IterableDataset for reading overlapping nucleotide sequences from a processed NumPy (.npy) file. """
    def __init__(self, npy_file_path: str, context_size: int = 256):
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"Dataset NPY file not found: {npy_file_path}")
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        self.npy_file_path = npy_file_path
        self.context_size = context_size
        self.data_size = 0
        self.num_possible_samples = 0
        self.data_dtype = np.uint8
        self.seed = None
        self.epoch = 0
        data = None # Initialize to None
        try:
            # Use memory mapping to read metadata without loading the whole file
            data = np.load(self.npy_file_path, mmap_mode='r')
            if len(data.shape) != 1:
                raise ValueError(f"Dataset NPY must be 1D, found shape {data.shape}")
            self.data_size = data.shape[0]
            self.data_dtype = data.dtype
            if self.data_dtype not in [np.uint8, np.int16, np.int32, np.int64]:
                logger.warning(f"Dataset dtype is {self.data_dtype}, expected integer.")

            if self.data_size == 0:
                raise ValueError("Dataset NPY file contains no data.")
            self.num_possible_samples = max(0, self.data_size - self.context_size)
            if self.num_possible_samples == 0:
                raise ValueError(f"No samples possible: Ctx={self.context_size:,} DataSize={self.data_size:,}.")
            logger.info(f"WuBuNestingDataset '{os.path.basename(npy_file_path)}': Size={self.data_size:,}, Samples={self.num_possible_samples:,}, DType={self.data_dtype}")

        except ImportError:
            logger.error("NumPy required.")
            raise
        except AttributeError as ae:
            logger.error(f"AttributeError during NPY metadata read {npy_file_path}: {ae}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error reading NPY metadata {npy_file_path}: {e}", exc_info=True)
            raise
        finally:
            if data is not None:
                del data
                gc.collect()

    def __len__(self):
        if self.num_possible_samples == 0:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        world_size = get_world_size() if is_initialized() else 1
        total_effective_workers = max(1, num_workers * world_size)
        # Calculate samples per worker, ensuring all samples are covered
        base_samples = self.num_possible_samples // total_effective_workers
        remainder = self.num_possible_samples % total_effective_workers
        worker_id = worker_info.id if worker_info else 0
        rank = get_rank() if is_initialized() else 0
        global_worker_id = rank * num_workers + worker_id
        num_samples_this_worker = base_samples + (1 if global_worker_id < remainder else 0)
        return num_samples_this_worker

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rank = get_rank() if is_initialized() else 0
        world_size = get_world_size() if is_initialized() else 1
        if self.num_possible_samples == 0:
            return iter([])

        total_effective_workers = max(1, num_workers * world_size)
        global_worker_id = rank * num_workers + worker_id

        # Calculate start and end index for this specific worker
        base_samples = self.num_possible_samples // total_effective_workers
        remainder = self.num_possible_samples % total_effective_workers
        start_sample_idx = global_worker_id * base_samples + min(global_worker_id, remainder)
        num_samples_this_worker = base_samples + (1 if global_worker_id < remainder else 0)
        end_sample_idx = start_sample_idx + num_samples_this_worker
        end_sample_idx = min(end_sample_idx, self.num_possible_samples)

        mmap_handle = None
        index_data = None
        try:
            index_data = np.load(self.npy_file_path, mmap_mode='r', allow_pickle=True)
            if hasattr(index_data, 'base') and isinstance(index_data.base, np.memmap):
                mmap_handle = index_data.base._mmap
            elif hasattr(index_data, '_mmap') and index_data._mmap is not None:
                mmap_handle = index_data._mmap
            else:
                logger.warning(f"W:{worker_id} R:{rank}: Could not get mmap handle, data loaded in memory or non-standard memmap.")

            if index_data is None or index_data.size != self.data_size:
                logger.error(f"W:{worker_id} R:{rank}: Failed load NPY or size mismatch.")
                return iter([])

            if start_sample_idx >= end_sample_idx:
                logger.debug(f"W:{worker_id} R:{rank}: No samples assigned (Start:{start_sample_idx}, End:{end_sample_idx}).")
                return iter([])

            # Generate the indices this worker is responsible for
            worker_indices = np.arange(start_sample_idx, end_sample_idx, dtype=np.int64)

            # Seed the random number generator for this worker/epoch
            base_seed = self.seed if self.seed is not None else int(time.time())
            seed_for_worker = (base_seed + global_worker_id + self.epoch * total_effective_workers) % (2**32)
            rng = np.random.default_rng(seed=seed_for_worker)
            rng.shuffle(worker_indices) # Shuffle only the indices this worker will process

            logger.debug(f"W:{worker_id} R:{rank}: Processing {len(worker_indices)} indices from {start_sample_idx} to {end_sample_idx-1} (Seed {seed_for_worker}, Epoch {self.epoch})")

            for idx in worker_indices:
                start_ctx = idx
                end_ctx = idx + self.context_size
                end_tgt = end_ctx + 1 # Target is shifted by one position

                if end_tgt > self.data_size:
                    logger.warning(f"W:{worker_id} R:{rank}: Index {idx} leads to out-of-bounds access (max_data_idx={self.data_size-1}, needed={end_tgt-1}). Skip.")
                    continue

                try:
                    context_slice = index_data[start_ctx : end_ctx]
                    target_slice = index_data[start_ctx + 1 : end_tgt] # Shifted target

                    if len(context_slice) != self.context_size or len(target_slice) != self.context_size:
                        logger.warning(f"W:{worker_id} R:{rank}: Slice length mismatch for index {idx} (ctx:{len(context_slice)}, tgt:{len(target_slice)}). Expected {self.context_size}. Skip.")
                        continue

                    context_tensor = torch.tensor(context_slice, dtype=torch.long)
                    target_tensor = torch.tensor(target_slice, dtype=torch.long)
                    yield context_tensor, target_tensor

                except IndexError:
                    logger.warning(f"W:{worker_id} R:{rank}: IndexError encountered for index {idx}. Skip.")
                    continue
                except Exception as e:
                    logger.error(f"W:{worker_id} R:{rank}: Error processing index {idx}: {e}")
                    continue

        except FileNotFoundError:
            logger.error(f"W:{worker_id} R:{rank}: NPY file not found: {self.npy_file_path}")
        except ValueError as ve:
            logger.error(f"W:{worker_id} R:{rank}: ValueError during NPY load/setup: {ve}")
        except Exception as e:
            logger.error(f"W:{worker_id} R:{rank}: Iterator failed unexpectedly: {e}", exc_info=True)
        finally:
            if mmap_handle is not None:
                try:
                    if hasattr(mmap_handle, 'close') and callable(mmap_handle.close):
                        mmap_handle.close()
                        logger.debug(f"W:{worker_id} R:{rank}: Closed mmap handle.")
                    else:
                        logger.debug(f"W:{worker_id} R:{rank}: mmap handle object does not have a close() method.")
                except Exception as close_err:
                    logger.warning(f"W:{worker_id} R:{rank}: Error closing mmap handle: {close_err}")
            if index_data is not None:
                del index_data
                index_data = None
            mmap_handle = None
            gc.collect()

    def set_seed(self, seed: int):
        self.seed = seed

    def set_epoch(self, epoch: int):
        self.epoch = epoch

class BalancedWuBuNestingDataset(IterableDataset):
    """
    IterableDataset for reading overlapping nucleotide sequences from a combined
    NumPy (.npy) file, with optional balanced sampling across source datasets.
    """
    def __init__(self, npy_file_path: str, context_size: int = 256,
                 balanced_sampling: bool = True, metadata_path: Optional[str] = None):
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"Dataset NPY file not found: {npy_file_path}")
        if context_size <= 0:
            raise ValueError("context_size must be positive")

        self.npy_file_path = npy_file_path
        self.context_size = context_size
        self.balanced_sampling = balanced_sampling
        self.data_size = 0
        self.num_possible_samples = 0
        self.data_dtype = np.uint8
        self.seed = None
        self.epoch = 0
        self.dataset_boundaries = [] # List of (name, start, end) tuples

        # --- Read metadata from NPY file ---
        mmap_obj = None
        try:
            # Use memory mapping to read metadata without loading the whole file
            mmap_obj = np.load(self.npy_file_path, mmap_mode='r', allow_pickle=True)
            if len(mmap_obj.shape) != 1:
                raise ValueError(f"Dataset NPY must be 1D, found shape {mmap_obj.shape}")
            self.data_size = mmap_obj.shape[0]
            self.data_dtype = mmap_obj.dtype
            if self.data_dtype not in [np.uint8, np.int16, np.int32, np.int64]:
                logger.warning(f"Dataset dtype is {self.data_dtype}, expected integer.")
            if self.data_size == 0:
                raise ValueError("Dataset NPY file contains no data.")
            self.num_possible_samples = max(0, self.data_size - self.context_size)
            if self.num_possible_samples == 0:
                raise ValueError(f"No samples possible: Ctx={self.context_size:,} DataSize={self.data_size:,}.")
        except Exception as e:
            logger.error(f"Error reading NPY metadata {self.npy_file_path}: {e}", exc_info=True)
            raise
        finally:
            if mmap_obj is not None:
                del mmap_obj
                gc.collect()

        # --- Load dataset boundaries for balanced sampling ---
        if metadata_path is None:
            metadata_path = os.path.join(os.path.dirname(self.npy_file_path), COMBINED_DATA_INFO_FILE)

        if self.balanced_sampling:
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                    self.dataset_boundaries = meta.get("dataset_boundaries", [])
                    # Basic validation of boundaries against data size
                    if self.dataset_boundaries and self.dataset_boundaries[-1][2] != self.data_size:
                         logger.warning(f"Metadata boundary end ({self.dataset_boundaries[-1][2]:,}) doesn't match data size ({self.data_size:,}). Balanced sampling might be inaccurate.")
                    if not self.dataset_boundaries:
                         logger.warning(f"Metadata file found but no boundaries listed. Disabling balanced sampling.")
                         self.balanced_sampling = False
                    else:
                         logger.info(f"Loaded boundaries for {len(self.dataset_boundaries)} datasets for balanced sampling.")
                except Exception as e:
                    logger.warning(f"Failed to load or parse dataset boundaries from {metadata_path}: {e}. Disabling balanced sampling.")
                    self.balanced_sampling = False
            else:
                logger.warning(f"Metadata file '{metadata_path}' not found. Disabling balanced sampling.")
                self.balanced_sampling = False

        # Log final configuration
        logger.info(f"BalancedWuBuNestingDataset '{os.path.basename(self.npy_file_path)}': Size={self.data_size:,}, Samples={self.num_possible_samples:,}, DType={self.data_dtype}, Balanced={self.balanced_sampling}")

    def __len__(self):
        # Estimate length per worker for progress bars etc.
        if self.num_possible_samples == 0:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        world_size = get_world_size() if is_initialized() else 1
        total_effective_workers = max(1, num_workers * world_size)
        # Return the approximate number of samples this worker will yield
        return max(1, self.num_possible_samples // total_effective_workers)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rank = get_rank() if is_initialized() else 0
        world_size = get_world_size() if is_initialized() else 1
        if self.num_possible_samples == 0:
            return iter([])

        total_effective_workers = max(1, num_workers * world_size)
        global_worker_id = rank * num_workers + worker_id

        # Calculate the *target* number of samples for this worker
        target_samples_this_worker = self.__len__() # Use estimated length

        # Create deterministic RNG for this worker/epoch
        base_seed = self.seed if self.seed is not None else int(time.time())
        seed_for_worker = (base_seed + global_worker_id + self.epoch * total_effective_workers) % (2**32)
        rng = np.random.default_rng(seed=seed_for_worker)

        mmap_handle = None
        index_data = None

        try:
            # Memory-map the dataset (read-only)
            index_data = np.load(self.npy_file_path, mmap_mode='r', allow_pickle=True)
            if hasattr(index_data, '_mmap') and index_data._mmap is not None:
                mmap_handle = index_data._mmap # Keep handle for closing

            # --- Generate sample indices ---
            worker_indices = []
            if self.balanced_sampling and self.dataset_boundaries:
                # Balanced sampling logic
                num_source_datasets = len(self.dataset_boundaries)
                if num_source_datasets == 0: # Fallback if boundaries somehow failed after init check
                     logger.warning(f"W:{worker_id} R:{rank}: No boundaries for balanced sampling. Falling back to simple sampling.")
                     indices_to_sample = rng.choice(self.num_possible_samples, size=target_samples_this_worker, replace=False)
                     worker_indices = indices_to_sample.tolist()
                else:
                     samples_per_dataset_target = max(1, target_samples_this_worker // num_source_datasets)
                     logger.debug(f"W:{worker_id} R:{rank}: Target {target_samples_this_worker} samples, ~{samples_per_dataset_target} per dataset.")

                     temp_indices_list = []
                     # Sample from each dataset proportionally
                     for i, (name, start_idx, end_idx) in enumerate(self.dataset_boundaries):
                         possible_starts_in_ds = max(0, (end_idx - start_idx) - self.context_size)
                         if possible_starts_in_ds == 0:
                            continue # Skip empty or too small datasets

                         # Determine how many samples to take from this dataset
                         num_to_sample = min(samples_per_dataset_target, possible_starts_in_ds)
                         # Simple remainder handling (distribute extra samples to first few datasets)
                         if i < (target_samples_this_worker % num_source_datasets):
                              num_to_sample = min(num_to_sample + 1, possible_starts_in_ds)

                         if num_to_sample > 0:
                              # Sample indices *within* the range of possible starts for this dataset
                              sampled_offsets = rng.choice(possible_starts_in_ds, size=num_to_sample, replace=True) 
                              # Convert offsets to absolute indices
                              absolute_indices = start_idx + sampled_offsets
                              temp_indices_list.extend(absolute_indices.tolist())

                     # Shuffle the combined list of indices from all datasets
                     rng.shuffle(temp_indices_list)
                     worker_indices = temp_indices_list
                     logger.debug(f"W:{worker_id} R:{rank}: Generated {len(worker_indices)} balanced indices.")

            else:
                # Simple random sampling across the entire dataset for this worker
                indices_to_sample = rng.choice(self.num_possible_samples, size=target_samples_this_worker, replace=False)
                worker_indices = indices_to_sample.tolist()
                logger.debug(f"W:{worker_id} R:{rank}: Generated {len(worker_indices)} simple random indices.")

            # --- Yield data ---
            yielded_count = 0
            for idx in worker_indices:
                # Optional: Check if enough samples have been yielded already
                # if yielded_count >= target_samples_this_worker: break

                # Calculate context and target slice indices
                start_ctx = int(idx) # Ensure integer index
                end_ctx = start_ctx + self.context_size
                end_tgt = end_ctx + 1

                # Check bounds (should be inherently correct if num_possible_samples was calculated right)
                if end_tgt > self.data_size:
                    logger.warning(f"W:{worker_id} R:{rank}: Calculated index {idx} leads to out-of-bounds access ({end_tgt} > {self.data_size}). Skipping.")
                    continue

                try:
                    context_slice = index_data[start_ctx : end_ctx]
                    target_slice = index_data[start_ctx + 1 : end_tgt] # Shifted target

                    if len(context_slice) != self.context_size or len(target_slice) != self.context_size:
                        logger.warning(f"W:{worker_id} R:{rank}: Slice length mismatch for index {idx}. Expected {self.context_size}. Skipping.")
                        continue

                    context_tensor = torch.tensor(context_slice, dtype=torch.long)
                    target_tensor = torch.tensor(target_slice, dtype=torch.long)
                    yield context_tensor, target_tensor
                    yielded_count += 1

                except IndexError:
                    logger.warning(f"W:{worker_id} R:{rank}: IndexError encountered for index {idx}. Skipping.")
                    continue
                except Exception as e:
                    logger.error(f"W:{worker_id} R:{rank}: Error processing index {idx}: {e}")
                    continue

        except FileNotFoundError:
            logger.error(f"W:{worker_id} R:{rank}: NPY file not found during iteration: {self.npy_file_path}")
        except Exception as e:
            logger.error(f"W:{worker_id} R:{rank}: Iterator failed unexpectedly: {e}", exc_info=True)
        finally:
            if mmap_handle is not None:
                try:
                    if hasattr(mmap_handle, 'close') and callable(mmap_handle.close):
                         mmap_handle.close()
                         logger.debug(f"W:{worker_id} R:{rank}: Closed mmap handle.")
                except Exception as close_err:
                    logger.warning(f"W:{worker_id} R:{rank}: Error closing mmap handle: {close_err}")
            if index_data is not None:
                del index_data
            mmap_handle = None
            gc.collect()

    def set_seed(self, seed: int):
        self.seed = seed

    def set_epoch(self, epoch: int):
        self.epoch = epoch

def seed_worker(worker_id: int, base_seed: int, rank_offset: int):
    """Sets the seed for a dataloader worker."""
    worker_seed = base_seed + rank_offset + worker_id
    worker_seed = worker_seed % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)
    logger.debug(f"Worker {worker_id} (Rank Offset {rank_offset}) seeded with {worker_seed}")

# =====================================================================
# HAKMEM Components
# =====================================================================
@dataclass
class SamplerConfig:
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

class GradientStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_gradients=0
        self.clipped_gradients=0
        self.max_gradient_norm=0.0
        self.sum_clip_ratios=0.0
        self.non_finite_grads_in_step=0
        self.step_stats={}

    def record_gradient(self, original_norm: float, clipped: bool, clip_ratio: Optional[float] = None):
        if np.isfinite(original_norm):
            self.total_gradients += 1
            self.max_gradient_norm = max(self.max_gradient_norm, original_norm)
            if clipped:
                self.clipped_gradients += 1
                self.sum_clip_ratios += (clip_ratio if clip_ratio is not None else 0.0)
        else:
            self.non_finite_grads_in_step += 1

    def get_step_stats(self) -> dict:
        if self.total_gradients == 0 and self.non_finite_grads_in_step == 0:
            return {"gradients_clipped":0,"total_gradients":0,"clip_ratio_avg":0.0,"max_gradient":0.0,"clip_percentage":0.0,"non_finite_grads":0}
        clip_percentage = (self.clipped_gradients / self.total_gradients) * 100 if self.total_gradients > 0 else 0.0
        avg_clip_ratio = self.sum_clip_ratios / self.clipped_gradients if self.clipped_gradients > 0 else 0.0
        return {"gradients_clipped":self.clipped_gradients, "total_gradients":self.total_gradients, "non_finite_grads":self.non_finite_grads_in_step, "clip_ratio_avg":avg_clip_ratio, "max_gradient":self.max_gradient_norm, "clip_percentage":clip_percentage}

    def record_step(self, step: int, skipped: bool = False) -> dict:
        stats = self.get_step_stats()
        stats['step_skipped'] = skipped
        self.step_stats[step] = stats
        self.reset()
        return stats

class HAKMEMEntropyHelper:
    def __init__(self, max_cache_size: int = 50000):
        self.entropy_cache={}
        self.max_cache_size = max_cache_size

    def _clean_cache(self):
        if len(self.entropy_cache) > self.max_cache_size:
            remove_count = len(self.entropy_cache) - (self.max_cache_size*4//5)
            keys_to_remove = list(itertools.islice(self.entropy_cache.keys(), remove_count))
            for k in keys_to_remove:
                if k in self.entropy_cache:
                    del self.entropy_cache[k]

    def compute_entropy(self, data_window: Union[np.ndarray, Tuple[int, ...], List[int], bytes, torch.Tensor], vocab_size: int = NUCLEOTIDE_VOCAB_SIZE) -> float: # Use NUCLEOTIDE_VOCAB_SIZE
        cache_key = None
        item_list = []
        prefix = f"v{vocab_size}_"
        if isinstance(data_window, tuple):
            item_list = list(data_window)
        elif isinstance(data_window, list):
            item_list = data_window
        elif isinstance(data_window, bytes):
            item_list = list(data_window)
        elif isinstance(data_window, np.ndarray):
            item_list = data_window.tolist()
        elif isinstance(data_window, torch.Tensor):
            item_list = data_window.cpu().long().tolist()
        else:
            logger.debug(f"Unknown type for entropy: {type(data_window)}")
            return 0.0
        item_list_int = [int(b) for b in item_list if isinstance(b, (int, float)) and b >= 0]
        if not item_list_int:
            return 0.0
        item_tuple = tuple(item_list_int)
        cache_key = prefix + str(item_tuple)
        if cache_key is not None and cache_key in self.entropy_cache:
            return self.entropy_cache[cache_key]
        try:
            counts = np.bincount(np.array(item_list_int, dtype=np.int64), minlength=vocab_size)
            total_items = counts.sum()
            if total_items == 0:
                return 0.0
            probs = counts[counts > 0] / total_items
            entropy = float(-np.sum(probs * np.log2(probs + EPS)))
            result = max(0.0, entropy)
            if cache_key is not None:
                self.entropy_cache[cache_key] = result
                self._clean_cache()
            return result
        except Exception as e:
            logger.warning(f"Entropy calculation failed for key {cache_key}: {e}")
            return 0.0

class HAKMEMBabylonIndex: # Unused for Nucleotides
    def __init__(self, max_cache_size: int = 50000):
        self.entropy_helper = HAKMEMEntropyHelper(max_cache_size)
        self.whitespace_chars = set(' \t\n\r\f\v')
        self.punctuation_chars = set('.,;:!?-()[]{}<>"\'`~@#$%^&*_=+|\\/')
        logger.info("HAKMEMBabylonIndex initialized (Word/Punctuation Patching - UNUSED).")

    def create_patches(self, byte_seq_tensor: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
        return []

    @torch.no_grad()
    def reset_context(self):
        self.entropy_helper.entropy_cache = {}

class HAKMEMCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_size>0
        valid_heads=[h for h in range(max(1,num_heads),0,-1) if hidden_size%h==0]
        original_num_heads=num_heads
        if not valid_heads:
            num_heads=1
            logger.warning(f"HAKMEMCrossAttn: No valid heads size {hidden_size}. Using 1 (Req: {original_num_heads}).")
        elif hidden_size%num_heads!=0:
            num_heads=valid_heads[0]
            logger.warning(f"HAKMEMCrossAttn: Adj heads {original_num_heads}->{num_heads} for size {hidden_size}.")
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.head_dim=hidden_size//self.num_heads
        self.scale=1./math.sqrt(max(1,self.head_dim))
        self.norm_q=nn.LayerNorm(hidden_size,eps=1e-6)
        self.norm_kv=nn.LayerNorm(hidden_size,eps=1e-6)
        self.q_proj=nn.Linear(hidden_size,hidden_size,bias=False)
        self.k_proj=nn.Linear(hidden_size,hidden_size,bias=False)
        self.v_proj=nn.Linear(hidden_size,hidden_size,bias=False)
        self.out_proj=nn.Linear(hidden_size,hidden_size,bias=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Nq, _ = queries.size()
        _, Nk, KVD = keys_values.size()
        assert KVD==self.hidden_size
        assert Nq > 0
        if Nk==0:
            return torch.zeros_like(queries)
        Qn=self.norm_q(queries)
        KVn=self.norm_kv(keys_values)
        Q=self.q_proj(Qn).view(B,Nq,self.num_heads,self.head_dim).transpose(1,2)
        K=self.k_proj(KVn).view(B,Nk,self.num_heads,self.head_dim).transpose(1,2)
        V=self.v_proj(KVn).view(B,Nk,self.num_heads,self.head_dim).transpose(1,2)
        attn_mask_sdpa=None
        attn_mask_manual=None
        if attention_mask is not None:
            if attention_mask.dim() == 2 and attention_mask.shape == (B, Nk):
                mask_bool = attention_mask.to(dtype=torch.bool, device=queries.device)
                attn_mask_manual = mask_bool.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, Nq, -1)
                attn_mask_sdpa = mask_bool.unsqueeze(1).expand(-1, Nq, -1)
            elif attention_mask.dim() == 4 and attention_mask.shape == (B, self.num_heads, Nq, Nk):
                 attn_mask_manual = attention_mask.to(dtype=torch.bool, device=queries.device)
                 attn_mask_sdpa = attention_mask
            else:
                logger.warning(f"HAKMEMCrossAttentionBlock unexpected mask shape {attention_mask.shape}. Ignoring.")
                attention_mask = None

        use_flash=hasattr(F,'scaled_dot_product_attention') and queries.device.type == 'cuda'
        output=None
        if use_flash:
            dp=self.dropout.p if self.training else 0.0
            try:
                output=F.scaled_dot_product_attention(Q,K,V,attn_mask=attn_mask_sdpa,dropout_p=dp, is_causal=False)
                if not torch.isfinite(output).all():
                    raise ValueError("Flash Attention output contained NaN/Inf")
            except Exception as flash_ex:
                logger.debug(f"Flash Attention failed: {flash_ex}. Fallback.",exc_info=False)
                use_flash=False
                output=None
        if output is None:
            scores=torch.matmul(Q,K.transpose(-2,-1))*self.scale
            scores=torch.clamp(scores,min=-30.,max=30.)
            if attn_mask_manual is not None:
                scores=scores.masked_fill(attn_mask_manual,float('-inf'))
            attn_probs=torch.softmax(scores.float(),dim=-1).to(scores.dtype)
            attn_probs=torch.nan_to_num(attn_probs)
            attn_probs=self.dropout(attn_probs)
            output=torch.matmul(attn_probs,V)
        output=output.transpose(1,2).contiguous().view(B,Nq,self.hidden_size)
        output=self.out_proj(output)
        if not torch.isfinite(output).all():
            num_non_finite = (~torch.isfinite(output)).sum().item()
            logger.warning(f"NaN/Inf ({num_non_finite}) in CrossAttention output. Replacing.")
            output=torch.nan_to_num(output,nan=0.0, posinf=0.0, neginf=0.0)
        return output

class HAKMEMQController:
    def __init__(self, learning_rate: float=0.01, discount: float=0.95, epsilon: float=0.2, epsilon_decay: float=0.9998, min_epsilon: float=0.01, lr_scale_options: Optional[List[float]]=None, momentum_scale_options: Optional[List[float]]=None, max_q_table_size: int=10000):
        self.q_table: Dict[Tuple, Dict[str, np.ndarray]]={}
        self.alpha=learning_rate
        self.gamma=discount
        self.epsilon=epsilon
        self.min_epsilon=min_epsilon
        self.epsilon_decay=epsilon_decay
        self.prev_loss: Optional[float]=None
        self.prev_state: Optional[Tuple]=None
        self.prev_action: Optional[Dict[str, float]]=None
        if lr_scale_options is None:
            lr_scale_options=[0.9,0.95,1.0,1.05,1.1]
        if momentum_scale_options is None:
            momentum_scale_options=[0.95,0.98,1.0,1.01,1.02]
        self.action_ranges={'lr_scale':np.array(lr_scale_options,dtype=np.float32),'momentum_scale':np.array(momentum_scale_options,dtype=np.float32)}
        self.num_actions={p:len(s) for p,s in self.action_ranges.items()}
        self.loss_window=deque(maxlen=20)
        self.grad_norm_window=deque(maxlen=20)
        self.lr_window=deque(maxlen=10)
        self.momentum_window=deque(maxlen=10)
        self.performance_window=deque(maxlen=50)
        self.stable_steps=0
        self.oscillation_counter=0
        self.prev_actions_log=deque(maxlen=10)
        self.max_q_table_size=max(100,max_q_table_size)
        self.q_table_access_count: Dict[Tuple, int]=defaultdict(int)
        self.q_table_creation_time: Dict[Tuple, float]={}
        self.flow_coefficient=0.05
        self.oscillation_penalty=0.15
        self.stability_reward_bonus=0.05
        self.large_grad_penalty_factor=0.1
        logger.info(f"QController init: a={self.alpha:.3f},g={self.gamma:.3f},e={self.epsilon:.3f},decay={self.epsilon_decay:.5f},min_e={self.min_epsilon:.3f}")
        logger.info(f"QController actions: LR={self.action_ranges['lr_scale']}, Mom={self.action_ranges['momentum_scale']}")

    def get_state(self,lr: float, momentum: float, grad_norm: Optional[float], loss: Optional[float])->Optional[Tuple]:
        if loss is None or grad_norm is None or not np.isfinite(loss) or not np.isfinite(grad_norm):
            logger.debug("QState skip: Invalid input")
            return None
        self.loss_window.append(loss)
        self.grad_norm_window.append(grad_norm)
        self.lr_window.append(lr)
        self.momentum_window.append(momentum)
        if len(self.loss_window)<5 or len(self.grad_norm_window)<5:
            logger.debug("QState skip: Insufficient history.")
            return None
        loss_trend_bin,grad_norm_level_bin,lr_level_bin,momentum_level_bin,oscillation_bin=2,2,2,1,0
        try:
            y_loss=np.array(list(self.loss_window)[-10:],dtype=np.float32)
            x_loss=np.arange(len(y_loss))
            slope_loss=np.polyfit(x_loss,y_loss,1)[0] if len(y_loss)>=3 and len(np.unique(y_loss))>1 else 0.
            avg_loss=np.median(y_loss)
            normalized_slope=slope_loss/(abs(avg_loss)+EPS)
            loss_trend_bin=np.digitize(normalized_slope,bins=[-0.05,-0.005,0.005,0.05]).item() if len(y_loss)>=3 else 2
            median_grad_norm=np.median(list(self.grad_norm_window))
            grad_norm_level_bin=np.digitize(median_grad_norm,bins=[0.1,0.5,1.5,5.0]).item()
            lr_level_bin=np.digitize(lr/1e-4,bins=[0.5,2.0,10.0,50.0]).item()
            momentum_level_bin=np.digitize(momentum,bins=[0.85,0.92,0.97]).item()
            if len(self.performance_window)>=5:
                recent_rewards=np.sign([r for r in list(self.performance_window)[-5:] if r!=0])
                sign_changes=np.sum(np.abs(np.diff(recent_rewards)))/2.0
                self.oscillation_counter=min(self.oscillation_counter+1,5) if sign_changes>=2 else max(0,self.oscillation_counter-1)
            oscillation_bin=1 if self.oscillation_counter>=3 else 0
        except Exception as e_state:
            logger.error(f"QState calc error: {e_state}",exc_info=True)
            return None
        state=(loss_trend_bin,grad_norm_level_bin,oscillation_bin,lr_level_bin,momentum_level_bin)
        self.q_table_access_count[state]+=1
        return state

    def compute_reward(self,current_loss: Optional[float],prev_loss: Optional[float],grad_norm: Optional[float])->float:
        if current_loss is None or prev_loss is None or grad_norm is None or not np.isfinite(current_loss) or not np.isfinite(prev_loss) or not np.isfinite(grad_norm):
            logger.debug("Reward skip: Invalid input")
            return 0.0
        loss_change=prev_loss-current_loss
        median_prev_loss=np.median(list(self.loss_window)[:-1]) if len(self.loss_window)>1 else prev_loss
        loss_change_ratio=loss_change/(abs(median_prev_loss)+EPS)
        base_reward=np.tanh(loss_change_ratio*10.0)
        grad_penalty=-self.large_grad_penalty_factor*min(1.0,max(0.0,(grad_norm-5.0)/10.0)) if grad_norm>5.0 else 0.0
        osc_penalty=-self.oscillation_penalty if self.oscillation_counter>=3 else 0.0
        current_perf_reward=base_reward+grad_penalty+osc_penalty
        self.performance_window.append(current_perf_reward)
        stab_bonus=0.0
        if current_perf_reward>0.01:
            self.stable_steps+=1
            stab_bonus=min(0.15,self.stability_reward_bonus*math.log1p(self.stable_steps/5.0))
        else:
            self.stable_steps=0
        reward=base_reward+grad_penalty+osc_penalty+stab_bonus
        return float(np.clip(reward,-1.0,1.0))

    def choose_action(self,state: Optional[Tuple])->Dict[str, float]:
        if state is None:
            return {'lr_scale':1.0,'momentum_scale':1.0}
        if state not in self.q_table:
            self.q_table[state]={p:np.zeros(self.num_actions[p],dtype=np.float32) for p in self.action_ranges.keys()}
            self.q_table_creation_time[state]=time.time()
            self.q_table_access_count[state]=1
            self._manage_q_table_size()
        self.epsilon=max(self.min_epsilon,self.epsilon*self.epsilon_decay)
        chosen_actions={}
        for param,q_values in self.q_table[state].items():
            action_space=self.action_ranges[param]
            if random.random()<self.epsilon:
                chosen_idx=random.randrange(len(action_space))
            else:
                finite_q_mask=np.isfinite(q_values)
                if not np.any(finite_q_mask):
                    chosen_idx=random.randrange(len(action_space))
                    logger.warning(f"QCtrl Warning: All Q-values non-finite {state}/{param}. Random action.")
                else:
                    finite_q_values=q_values[finite_q_mask]
                    max_q=np.max(finite_q_values)
                    finite_indices = np.where(finite_q_mask)[0]
                    best_indices = finite_indices[np.isclose(q_values[finite_q_mask], max_q)]
                    if len(best_indices)>0:
                        chosen_idx=np.random.choice(best_indices)
                    else:
                        logger.warning(f"QCtrl Warning: Cannot find best action index among finite Qs {state}/{param}. Random.")
                        chosen_idx=random.randrange(len(action_space))
            chosen_actions[param]=float(action_space[chosen_idx])
        self.prev_actions_log.append(chosen_actions.copy())
        return chosen_actions

    def update(self,state: Optional[Tuple],action: Optional[Dict[str, float]],reward: float,next_state: Optional[Tuple]):
        if state is None or next_state is None or action is None:
            logger.debug("QUpdate skip: Invalid state/action.")
            return
        if state not in self.q_table:
            logger.warning(f"QCtrl Warning: State {state} not found in update.")
            return
        if next_state not in self.q_table:
            self.q_table[next_state]={p:np.zeros(self.num_actions[p],dtype=np.float32) for p in self.action_ranges.keys()}
            self.q_table_creation_time[next_state]=time.time()
            self.q_table_access_count[next_state]=0
            self._manage_q_table_size()
        for param,chosen_value in action.items():
            if param not in self.q_table[state]:
                logger.warning(f"QCtrl Warning: Param {param} not in Q-table {state}. Skip update.")
                continue
            action_space=self.action_ranges[param]
            action_indices=np.where(np.isclose(action_space,chosen_value))[0]
            if len(action_indices)==0:
                logger.warning(f"QCtrl Warning: Cannot find action index {param}={chosen_value}. Skip update.")
                continue
            action_idx=action_indices[0]
            current_q=self.q_table[state][param][action_idx]
            next_q_values=self.q_table[next_state][param]
            finite_next_q=next_q_values[np.isfinite(next_q_values)]
            max_future_q=np.max(finite_next_q) if len(finite_next_q)>0 else 0.0
            if not np.isfinite(max_future_q):
                max_future_q=0.0
            td_target=reward+self.gamma*max_future_q
            td_error=td_target-current_q
            adaptive_alpha=min(0.5,max(0.001,self.alpha*(1.0+self.flow_coefficient*np.tanh(abs(td_error)*0.5))))
            new_q=current_q+adaptive_alpha*td_error
            if np.isfinite(new_q):
                self.q_table[state][param][action_idx]=np.clip(new_q,-1e4,1e4)
            else:
                logger.warning(f"QCtrl Warning: Non-finite new Q ({new_q:.2f}) for {state}/{param}/{action_idx}. Reset to 0.")
                self.q_table[state][param][action_idx]=0.0

    def _manage_q_table_size(self):
        if len(self.q_table)>self.max_q_table_size:
            num_to_remove=len(self.q_table)-self.max_q_table_size
            logger.info(f"QTable prune: size {len(self.q_table)} > max {self.max_q_table_size}. Removing {num_to_remove}.")
            try:
                if not self.q_table_access_count or not self.q_table_creation_time or len(self.q_table_access_count)<len(self.q_table)//2 or len(self.q_table_creation_time)<len(self.q_table)//2:
                    logger.warning("QTable prune: Incomplete metadata. Random removal.")
                    current_keys = list(self.q_table.keys())
                    states_to_remove = random.sample(current_keys, min(num_to_remove, len(current_keys)))
                else:
                    sorted_states=sorted(self.q_table.keys(),key=lambda s:(self.q_table_access_count.get(s,0),self.q_table_creation_time.get(s,float('inf'))))
                    states_to_remove=sorted_states[:num_to_remove]
                removed_count=0
                for state_to_remove in states_to_remove:
                    if state_to_remove in self.q_table:
                        self.q_table.pop(state_to_remove, None)
                        self.q_table_access_count.pop(state_to_remove, None)
                        self.q_table_creation_time.pop(state_to_remove, None)
                        removed_count += 1
                    else:
                        logger.debug(f"QTable prune: State {state_to_remove} already removed.")
                logger.info(f"QTable pruned {removed_count} states. New size: {len(self.q_table)}")
            except Exception as e:
                logger.warning(f"QTable prune error: {e}. Fallback random removal.")
                current_keys=list(self.q_table.keys())
                num_to_remove_fallback=max(0,len(current_keys)-self.max_q_table_size)
                if num_to_remove_fallback>0:
                    states_to_remove_fb=random.sample(current_keys,min(num_to_remove_fallback,len(current_keys)))
                    removed_count_fb=0
                    for state_to_remove_fb in states_to_remove_fb:
                        self.q_table.pop(state_to_remove_fb,None)
                        self.q_table_access_count.pop(state_to_remove_fb,None)
                        self.q_table_creation_time.pop(state_to_remove_fb,None)
                        removed_count_fb+=1
                    logger.info(f"QTable fallback pruned {removed_count_fb} random states. New size: {len(self.q_table)}")

    def get_info(self) -> Dict:
        last_action=self.prev_actions_log[-1] if self.prev_actions_log else None
        avg_reward=np.mean(list(self.performance_window)) if self.performance_window else 0.0
        q_table_mem_approx = 0
        try:
            q_table_mem_approx = sys.getsizeof(self.q_table)
            for k, v_dict in self.q_table.items():
                q_table_mem_approx += sys.getsizeof(k) + sys.getsizeof(v_dict)
                for param_key, arr in v_dict.items():
                    q_table_mem_approx += sys.getsizeof(param_key) + (arr.nbytes if isinstance(arr, np.ndarray) else sys.getsizeof(arr))
            q_table_mem_approx /= (1024**2)
        except Exception as e:
            logger.debug(f"Could not estimate Q-table memory: {e}")
            q_table_mem_approx = -1.0
        return {"epsilon":self.epsilon,"stable_steps":self.stable_steps,"oscillation_counter":self.oscillation_counter,"q_table_size":len(self.q_table),"q_table_mem_mb_approx":round(q_table_mem_approx,2),"last_action":last_action,f"avg_reward_last_{self.performance_window.maxlen}":avg_reward}


# =====================================================================
# Hyperbolic Geometry Implementation
# =====================================================================
class Manifold:
    """Abstract base class for manifolds."""
    def __init__(self):
        pass
    def dist(self, x, y, keepdim=False):
        raise NotImplementedError
    def sqdist(self, x, y, keepdim=False):
        raise NotImplementedError
    def egrad2rgrad(self, p, dp):
        raise NotImplementedError
    def proj(self, p, dp): # Project vector dp onto tangent space at p
        raise NotImplementedError
    def proju(self, p): # Project point p onto the manifold
        raise NotImplementedError
    def expmap(self, p, dp): # Exponential map
        raise NotImplementedError
    def logmap(self, p, y): # Logarithmic map
        raise NotImplementedError
    def expmap0(self, dp): # Exp map from origin
        raise NotImplementedError
    def logmap0(self, p): # Log map to origin
        raise NotImplementedError
    def mobius_add(self, x, y):
        raise NotImplementedError
    def mobius_matvec(self, m, x):
        raise NotImplementedError
    def init_weights(self, w, irange=1e-5):
        raise NotImplementedError
    def zero_grad(self, p):
        p.grad.data.zero_()
    def normalize(self, p):
        return self.proju(p) # Default normalize is projection
    def check_point_on_manifold(self, p, atol=1e-5):
        raise NotImplementedError
    def check_vector_on_tangent(self, p, dp, atol=1e-5):
        raise NotImplementedError
    @property
    def name(self):
        return self.__class__.__name__

class PoincareBall(Manifold):
    """Poincaré Ball manifold class with curvature c."""
    def __init__(self, c=1.0):
        super().__init__()
        if isinstance(c, torch.Tensor):
            self.c = c.item()
        elif not isinstance(c, (float, int)):
            raise TypeError(f"Curvature c must be float/int/scalar tensor, got {type(c)}")
        else:
            self.c = float(c)
        if self.c <= 0:
            logger.warning(f"PoincareBall init c={self.c:.3g} <= 0. Ops may act Euclidean.")
            self.k=0.0
            self.sqrt_c=0.0
            self.radius=float('inf')
        else:
            self.k = -self.c
            self.sqrt_c = math.sqrt(self.c)
            self.radius = 1.0 / self.sqrt_c
        self.max_norm = self.radius * (1.0 - EPS) if self.c > 0 else float('inf')
        self.min_norm = EPS
        self._name = f'PoincareBall(c={self.c:.3g})'
    @property
    def name(self):
        return self._name
    def _check_c(self, require_positive=True):
        assert not require_positive or self.c > 0, f"{self.name}: Op requires c>0 (is {self.c})."
    def lambda_x(self, x, keepdim=False):
        if self.c <= 0:
            return torch.ones_like(x[..., 0:1]) * 2.0
        x_f = x.float()
        c_f = float(self.c)
        x_norm_sq = torch.sum(x_f.pow(2), dim=-1, keepdim=True)
        denominator = torch.clamp(1. - c_f * x_norm_sq, min=EPS)
        res = 2. / denominator
        return res.to(x.dtype)
    def sqdist(self, x, y, keepdim=False):
        if self.c <= 0:
            return torch.clamp(torch.sum((x - y).pow(2), dim=-1, keepdim=keepdim), min=0.0)
        compute_dtype = torch.float32 if x.dtype != torch.float64 else torch.float64
        with torch.enable_grad() if x.requires_grad or y.requires_grad else torch.no_grad():
            x_proj=self.proju(x)
            y_proj=self.proju(y)
            x_proj_f = x_proj.to(compute_dtype)
            y_proj_f = y_proj.to(compute_dtype)
            c_f = float(self.c)
            diff_norm_sq = torch.sum((x_proj_f - y_proj_f).pow(2), dim=-1, keepdim=True)
            x_norm_sq = torch.sum(x_proj_f.pow(2), dim=-1, keepdim=True).clamp(min=0)
            y_norm_sq = torch.sum(y_proj_f.pow(2), dim=-1, keepdim=True).clamp(min=0)
            denom_x = torch.clamp(1. - c_f * x_norm_sq, min=EPS)
            denom_y = torch.clamp(1. - c_f * y_norm_sq, min=EPS)
            denominator_product = denom_x * denom_y
            numerator = 2. * c_f * diff_norm_sq
            arcosh_arg = 1. + numerator / (denominator_product + EPS)
            arcosh_arg_clamped = torch.clamp(arcosh_arg, min=1.0 + EPS)
            acosh_val = torch.acosh(arcosh_arg_clamped)
            sq_dist_val = (1.0 / c_f) * acosh_val.pow(2)
        if not keepdim:
            sq_dist_val = sq_dist_val.squeeze(-1)
        return sq_dist_val.to(x.dtype)
    def dist(self, x, y, keepdim=False):
        if self.c <= 0:
            return torch.clamp(torch.norm(x - y, p=2, dim=-1, keepdim=keepdim), min=0.0)
        sq_dist = self.sqdist(x, y, keepdim=True)
        dist_val = torch.sqrt(sq_dist + EPS).clamp(min=0.0)
        if not keepdim:
            dist_val = dist_val.squeeze(-1)
        return dist_val
    def proju(self, x):
        if self.c <= 0:
            return x
        if not torch.is_tensor(x):
            logger.warning(f"proju received non-tensor type {type(x)}. Returning as is.")
            return x
        if not torch.isfinite(x).all():
            num_non_finite = (~torch.isfinite(x)).sum().item()
            logger.warning(f"proju input contains {num_non_finite} non-finite values. Clamping/replacing with 0.")
            safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            safe_x = x
        x_norm = torch.norm(safe_x, p=2, dim=-1, keepdim=True)
        scale = torch.clamp_max(self.max_norm / (x_norm + EPS), 1.0)
        projected_x = safe_x * scale
        if not torch.isfinite(projected_x).all():
             logger.error("NaN detected *after* proju scaling. Check EPS or input values.")
             projected_x = torch.nan_to_num(projected_x, nan=0.0)
        return projected_x
    def proj(self, p, dp):
        return dp # Tangent space is R^n
    def expmap(self, p, dp):
        if self.c <= 0:
            return p + dp
        p = self.proju(p)
        compute_dtype = torch.float32 if p.dtype != torch.float64 else torch.float64
        p_f = p.to(compute_dtype)
        dp_f = dp.to(compute_dtype)
        dp_norm_f = torch.norm(dp_f, p=2, dim=-1, keepdim=True).clamp(min=EPS)
        if torch.all(dp_norm_f < EPS):
            return p
        c_f = float(self.c)
        sqrt_c_f = math.sqrt(c_f)
        lambda_p_f = self.lambda_x(p_f, keepdim=True).to(compute_dtype)
        tanh_arg = sqrt_c_f * lambda_p_f * dp_norm_f / 2.
        tanh_arg_clamped = torch.clamp(tanh_arg, min=-15., max=15.)
        factor = torch.tanh(tanh_arg_clamped) / (sqrt_c_f * dp_norm_f + EPS)
        exp_res_f = self.mobius_add(p_f, factor * dp_f)
        return self.proju(exp_res_f.to(p.dtype))
    def logmap(self, p, y):
        if self.c <= 0:
            return y - p
        p = self.proju(p)
        y = self.proju(y)
        if torch.allclose(p, y, atol=EPS):
            return torch.zeros_like(p)
        compute_dtype = torch.float32 if p.dtype != torch.float64 else torch.float64
        p_f = p.to(compute_dtype)
        y_f = y.to(compute_dtype)
        c_f = float(self.c)
        sqrt_c_f = math.sqrt(c_f)
        sub_f = self.mobius_add(-p_f, y_f)
        sub_norm_f = torch.norm(sub_f, p=2, dim=-1, keepdim=True).clamp(min=EPS)
        lambda_p_f = self.lambda_x(p_f, keepdim=True).to(compute_dtype)
        atanh_arg = sqrt_c_f * sub_norm_f
        atanh_arg_clamped = torch.clamp(atanh_arg, min=-1. + EPS, max=1. - EPS)
        factor = (2. / (sqrt_c_f * lambda_p_f + EPS)) * torch.atanh(atanh_arg_clamped) / (sub_norm_f + EPS)
        return (factor * sub_f).to(p.dtype)
    def expmap0(self, dp):
        if self.c <= 0:
            return dp
        compute_dtype = torch.float32 if dp.dtype != torch.float64 else torch.float64
        dp_f = dp.to(compute_dtype)
        dp_norm_f = torch.norm(dp_f, p=2, dim=-1, keepdim=True).clamp(min=EPS)
        if torch.all(dp_norm_f < EPS):
            return torch.zeros_like(dp)
        c_f = float(self.c)
        sqrt_c_f = math.sqrt(c_f)
        tanh_arg = sqrt_c_f * dp_norm_f
        tanh_arg_clamped = torch.clamp(tanh_arg, min=-15., max=15.)
        factor = torch.tanh(tanh_arg_clamped) / (sqrt_c_f * dp_norm_f + EPS)
        exp0_res_f = factor * dp_f
        return self.proju(exp0_res_f.to(dp.dtype))
    def logmap0(self, p):
        if self.c <= 0:
            return p
        p = self.proju(p)
        compute_dtype = torch.float32 if p.dtype != torch.float64 else torch.float64
        p_f = p.to(compute_dtype)
        p_norm_f = torch.norm(p_f, p=2, dim=-1, keepdim=True).clamp(min=EPS)
        if torch.all(p_norm_f < EPS):
            return torch.zeros_like(p)
        c_f = float(self.c)
        sqrt_c_f = math.sqrt(c_f)
        atanh_arg = sqrt_c_f * p_norm_f
        atanh_arg_clamped = torch.clamp(atanh_arg, min=-1. + EPS, max=1. - EPS)
        factor = torch.atanh(atanh_arg_clamped) / (sqrt_c_f * p_norm_f + EPS)
        return (factor * p_f).to(p.dtype)
    def mobius_add(self, x, y):
        if self.c <= 0:
            return x + y
        compute_dtype = torch.float32 if x.dtype != torch.float64 else torch.float64
        with torch.enable_grad() if x.requires_grad or y.requires_grad else torch.no_grad():
            x = self.proju(x)
            y = self.proju(y)
            x_f = x.to(compute_dtype)
            y_f = y.to(compute_dtype)
            c_f = float(self.c)
            x_norm_sq = torch.sum(x_f.pow(2), dim=-1, keepdim=True).clamp(min=0)
            y_norm_sq = torch.sum(y_f.pow(2), dim=-1, keepdim=True).clamp(min=0)
            xy_dot = torch.sum(x_f * y_f, dim=-1, keepdim=True)
            num_factor_x = 1. + 2. * c_f * xy_dot + c_f * y_norm_sq
            num_factor_y = torch.clamp(1. - c_f * x_norm_sq, min=EPS)
            numerator = num_factor_x * x_f + num_factor_y * y_f
            denominator = 1. + 2. * c_f * xy_dot + c_f**2 * x_norm_sq * y_norm_sq
            denominator = torch.clamp(denominator, min=EPS)
            res_f = numerator / denominator
        return self.proju(res_f.to(x.dtype))
    def mobius_scalar_mul(self, r: Union[float, torch.Tensor], x):
        if self.c <= 0:
            return r * x
        x = self.proju(x)
        compute_dtype = torch.float32 if x.dtype != torch.float64 else torch.float64
        x_f = x.to(compute_dtype)
        x_norm_f = torch.norm(x_f, p=2, dim=-1, keepdim=True).clamp(min=EPS)
        if torch.all(x_norm_f < EPS):
            return torch.zeros_like(x)
        c_f = float(self.c)
        sqrt_c_f = math.sqrt(c_f)
        r_f = r.float() if isinstance(r, torch.Tensor) else float(r)
        atanh_arg = torch.clamp(sqrt_c_f * x_norm_f, min=-1. + EPS, max=1. - EPS)
        tanh_inner_arg = r_f * torch.atanh(atanh_arg)
        tanh_term = torch.tanh(torch.clamp(tanh_inner_arg, min=-15., max=15.))
        direction = x_f / (x_norm_f + EPS)
        magnitude = tanh_term / (sqrt_c_f + EPS)
        res_f = magnitude * direction
        return self.proju(res_f.to(x.dtype))
    def mobius_matvec(self, M: nn.Parameter, x):
        if self.c <= 0:
            return F.linear(x, M)
        x = self.proju(x)
        if torch.allclose(x, torch.zeros_like(x), atol=EPS):
            return torch.zeros_like(x)
        x_log = self.logmap0(x)
        Mx_log = F.linear(x_log, M)
        Mx_hyp = self.expmap0(Mx_log)
        return self.proju(Mx_hyp)
    def egrad2rgrad(self, p, dp):
        if self.c <= 0:
            return dp
        p = self.proju(p)
        compute_dtype = torch.float32 if p.dtype != torch.float64 else torch.float64
        p_f = p.to(compute_dtype)
        lambda_p_sq = self.lambda_x(p_f, keepdim=True).pow(2)
        factor = (1.0 / torch.clamp(lambda_p_sq, min=EPS)).to(p.dtype)
        rgrad = factor * dp
        if not torch.isfinite(rgrad).all():
            num_non_finite = (~torch.isfinite(rgrad)).sum().item()
            logger.warning(f"egrad2rgrad produced {num_non_finite} non-finite values. Check inputs/lambda_x.")
            rgrad = torch.nan_to_num(rgrad, nan=0.0, posinf=0.0, neginf=0.0)
        return rgrad
    def init_weights(self, w: nn.Parameter, irange=1e-5):
        if not hasattr(w,'manifold') or w.manifold!=self:
             logger.warning("init_weights called on param not assigned to this manifold. Using uniform near origin.")
        with torch.no_grad():
            w.data.uniform_(-irange, irange)
            w.data = self.proju(w.data)
            w.manifold = self
    def check_point_on_manifold(self, p, atol=1e-5):
        if self.c <= 0:
            return torch.ones_like(p[...,0],dtype=torch.bool)
        norm_sq = torch.sum(p.pow(2), dim=-1)
        on_manifold = norm_sq <= (self.radius**2 + atol)
        if not torch.all(on_manifold):
            max_norm_found = torch.sqrt(norm_sq.max()).item()
            logger.debug(f"Point check fail: Max norm {max_norm_found:.4f} vs radius {self.radius:.4f} (atol={atol})")
        return on_manifold
    def check_vector_on_tangent(self, p, dp, atol=1e-5):
        if p.shape != dp.shape:
            logger.warning(f"Tangent vector shape {dp.shape} mismatch point shape {p.shape}")
            return False
        return True

def get_manifold(name="poincare", curvature=1.0) -> Manifold:
    name_lower = name.lower()
    if name_lower == "poincare":
        return PoincareBall(c=curvature)
    else:
        raise ValueError(f"Unknown manifold: {name}")

# =====================================================================
# Helper Function for Weight Initialization
# =====================================================================
def init_weights(module):
    """Initializes weights for common layer types, skipping specific hyperbolic ones."""
    if isinstance(module, nn.Linear):
        is_gyro_linear = isinstance(module, GyroLinear)
        is_gyro_bias = hasattr(module, 'bias') and module.bias is not None and hasattr(module.bias, 'manifold') and isinstance(getattr(module.bias, 'manifold', None), Manifold)
        if not is_gyro_linear:
            torch.nn.init.xavier_uniform_(module.weight, gain=math.sqrt(2.0))
            if module.bias is not None and not is_gyro_bias:
                torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
         is_hyperbolic_embedding = isinstance(module, HyperbolicEmbedding) or (hasattr(module.weight, 'manifold') and isinstance(getattr(module.weight, 'manifold', None), Manifold))
         if not is_hyperbolic_embedding:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        is_riemannian_ln = isinstance(module, RiemannianLayerNorm) or (hasattr(module, 'manifold') and isinstance(getattr(module, 'manifold', None), Manifold))
        if not is_riemannian_ln:
             if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.ones_(module.weight)
             if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

# =====================================================================
# Hyperbolic Layers Implementation
# =====================================================================
class HyperbolicEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, manifold: PoincareBall, sparse: bool = False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        assert isinstance(manifold, PoincareBall), "HyperbolicEmbedding requires PoincareBall manifold"
        self.manifold = manifold
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.weight.manifold = manifold
        self.sparse = sparse
        self.reset_parameters()
        logger.debug(f"HypEmb init: {num_embeddings}x{embedding_dim} on {manifold.name}")

    def reset_parameters(self):
        with torch.no_grad():
            self.manifold.init_weights(self.weight, irange=1e-5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_clamped = torch.clamp(input, 0, self.num_embeddings - 1)
        embeddings = F.embedding(input_clamped, self.weight, sparse=self.sparse)
        return self.manifold.proju(embeddings)

    def extra_repr(self):
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, manifold={self.manifold.name}, sparse={self.sparse}'

class GyroLinear(nn.Module):
    """Hyperbolic Linear Layer using Gyrovector space operations."""
    def __init__(self, in_features: int, out_features: int, manifold: PoincareBall, bias: bool = True):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        assert isinstance(manifold,PoincareBall), "GyroLinear requires PoincareBall manifold"
        self.manifold=manifold
        self.use_bias=bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias=nn.Parameter(torch.Tensor(out_features))
            self.bias.manifold=manifold
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        logger.debug(f"GyroLinear init: {in_features}->{out_features}, bias={self.use_bias}, manifold={manifold.name}")

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.use_bias:
            with torch.no_grad():
                self.manifold.init_weights(self.bias, irange=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj=self.manifold.proju(x)
        output_hyperbolic=self.manifold.mobius_matvec(self.weight,x_proj)
        if self.use_bias and self.bias is not None:
            bias_proj=self.manifold.proju(self.bias)
            output_hyperbolic=self.manifold.mobius_add(output_hyperbolic,bias_proj)
        return self.manifold.proju(output_hyperbolic)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}, manifold={self.manifold.name}'

class RiemannianLayerNorm(nn.Module):
    """Layer Normalization adapted for Riemannian manifolds (specifically Poincare Ball)."""
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], manifold: PoincareBall, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        assert isinstance(manifold,PoincareBall), "RiemannianLayerNorm requires PoincareBall"
        self.manifold=manifold
        self.eps=eps
        self.elementwise_affine=elementwise_affine
        if self.elementwise_affine:
            self.gamma=nn.Parameter(torch.Tensor(*self.normalized_shape))
            self.beta=nn.Parameter(torch.Tensor(*self.normalized_shape))
        else:
            self.register_parameter('gamma',None)
            self.register_parameter('beta',None)
        self.reset_parameters()
        logger.debug(f"RiemLayerNorm init: shape={self.normalized_shape}, affine={elementwise_affine}, manifold={manifold.name}")

    def reset_parameters(self):
        if self.elementwise_affine:
            if self.gamma is not None:
                nn.init.ones_(self.gamma)
            if self.beta is not None:
                nn.init.zeros_(self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj=self.manifold.proju(x)
        x_tangent=self.manifold.logmap0(x_proj)
        dims_to_normalize=tuple(range(x_tangent.dim()-len(self.normalized_shape),x_tangent.dim()))
        mean=torch.mean(x_tangent,dim=dims_to_normalize,keepdim=True)
        variance=torch.var(x_tangent,dim=dims_to_normalize,keepdim=True,unbiased=False)
        x_tangent_normalized=(x_tangent-mean)/torch.sqrt(variance+self.eps)
        if self.elementwise_affine:
             if self.gamma is not None:
                x_tangent_normalized = self.gamma * x_tangent_normalized
             if self.beta is not None:
                x_tangent_normalized = x_tangent_normalized + self.beta
        output_hyperbolic = self.manifold.expmap0(x_tangent_normalized)
        return self.manifold.proju(output_hyperbolic)

    def extra_repr(self):
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, manifold={self.manifold.name}'

class HyperbolicDistanceAttention(nn.Module): # Experimental
    def __init__(self, embed_dim: int, num_heads: int, manifold: PoincareBall, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.embed_dim=embed_dim
        assert embed_dim > 0
        assert num_heads > 0
        assert embed_dim % num_heads == 0
        self.num_heads=num_heads
        self.head_dim=embed_dim//self.num_heads
        assert isinstance(manifold,PoincareBall), "HyperbolicDistanceAttention requires PoincareBall"
        self.manifold=manifold
        self.q_proj=GyroLinear(embed_dim,embed_dim,manifold,bias=bias)
        self.k_proj=GyroLinear(embed_dim,embed_dim,manifold,bias=bias)
        self.v_proj=GyroLinear(embed_dim,embed_dim,manifold,bias=bias)
        self.out_proj=GyroLinear(embed_dim,embed_dim,manifold,bias=bias)
        self.dropout=nn.Dropout(dropout)
        self.log_neg_dist_scale=nn.Parameter(torch.tensor(0.0))
        logger.debug(f"HypDistAttn init: dim={embed_dim}, heads={num_heads} on {manifold.name}")

    def get_neg_dist_scale(self)->torch.Tensor:
        return torch.exp(self.log_neg_dist_scale)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None, attn_mask: Optional[torch.Tensor]=None):
        B,T,C = query.size()
        S = key.size(1)
        dev = query.device
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q_exp = q.unsqueeze(3)
        k_exp = k.unsqueeze(2)
        sq_dist = self.manifold.sqdist(q_exp, k_exp, keepdim=True)
        dist = torch.sqrt(sq_dist.clamp(min=0.) + EPS).squeeze(-1)
        scale = self.get_neg_dist_scale()
        scores = -scale * dist
        scores = torch.clamp(scores, min=-30., max=30.)
        if attn_mask is not None:
             if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
             elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
             scores = scores.masked_fill(attn_mask.to(dtype=torch.bool, device=dev), float('-inf'))
        if key_padding_mask is not None:
             pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.bool, device=dev)
             scores = scores.masked_fill(pad_mask, float('-inf'))
        probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        probs = torch.nan_to_num(probs, nan=0.)
        probs = self.dropout(probs)
        v_tan = self.manifold.logmap0(v)
        attn_tan = torch.matmul(probs, v_tan)
        if not torch.isfinite(attn_tan).all():
            num_non_finite = (~torch.isfinite(attn_tan)).sum().item()
            logger.warning(f"HypDistAttn: Non-finite tangent space attn ({num_non_finite}). Replacing with 0.")
            attn_tan = torch.nan_to_num(attn_tan, nan=0.0, posinf=0.0, neginf=0.0)
        attn_tan_r = attn_tan.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        attn_hyp_pre = self.manifold.expmap0(attn_tan_r)
        output = self.out_proj(attn_hyp_pre)
        return output

# =====================================================================
# WuBu Nesting Core
# =====================================================================
class BoundaryManifoldHyperbolic(nn.Module):
    """Represents boundary points for a single level on a hyperbolic manifold."""
    def __init__(self, level_idx: int, num_points: int, point_dim: int, initial_manifold: PoincareBall):
        super().__init__()
        self.level_idx=level_idx
        self.num_points=num_points
        self.point_dim=point_dim
        self.initial_manifold=initial_manifold
        self.current_manifold:PoincareBall=initial_manifold
        if num_points>0 and point_dim>0:
            self.hyperbolic_points=nn.Parameter(torch.Tensor(num_points,point_dim))
            self.hyperbolic_points.manifold=self.current_manifold
            self.reset_parameters()
            logger.info(f"BoundaryManifoldHyp L{level_idx}: {num_points} pts {point_dim}D ({initial_manifold.name}).")
        else:
            self.register_parameter('hyperbolic_points',None)
            logger.info(f"BoundaryManifoldHyp L{level_idx}: No boundary points.")

    def reset_parameters(self):
        if self.hyperbolic_points is not None:
            with torch.no_grad():
                self.initial_manifold.init_weights(self.hyperbolic_points,irange=1e-5)

    def set_current_manifold(self, manifold:PoincareBall):
        self.current_manifold=manifold
        if self.hyperbolic_points is not None:
            self.hyperbolic_points.manifold=manifold

    def get_points(self)->Optional[torch.Tensor]:
        if self.hyperbolic_points is None:
            return None
        return self.current_manifold.proju(self.hyperbolic_points)

class HyperbolicInterLevelTransform(nn.Module):
    """Transforms points and descriptors between adjacent hyperbolic levels."""
    def __init__(self, in_dim:int, out_dim:int, manifold_in:PoincareBall, manifold_out:PoincareBall, transform_type:str, hidden_dim:Optional[int]=None, dropout:float=0.1):
        super().__init__()
        assert isinstance(manifold_in,PoincareBall) and isinstance(manifold_out,PoincareBall), "Transforms require PoincareBall manifolds"
        self.manifold_in_init=manifold_in
        self.manifold_out_init=manifold_out
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.transform_type=transform_type.lower()
        if self.transform_type=='mlp':
            mlp_hidden_dim=hidden_dim if hidden_dim is not None and hidden_dim>0 else max(16,(in_dim+out_dim)//2)
            self.tangent_transform=nn.Sequential(nn.Linear(in_dim,mlp_hidden_dim), nn.LayerNorm(mlp_hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden_dim,out_dim))
            logger.info(f"HypInterLevelTr ({in_dim}->{out_dim}, {manifold_in.name}->{manifold_out.name}): MLP({mlp_hidden_dim}) T_0")
        elif self.transform_type=='linear':
            self.tangent_transform=nn.Linear(in_dim,out_dim)
            logger.info(f"HypInterLevelTr ({in_dim}->{out_dim}, {manifold_in.name}->{manifold_out.name}): Linear T_0")
        else:
            raise ValueError(f"Unsupported transform_type: {transform_type}")
        self.apply(init_weights)

    def forward(self, point_in:torch.Tensor, boundaries_in:Optional[torch.Tensor], descriptor_in:Optional[torch.Tensor], manifold_in_current:PoincareBall, manifold_out_current:PoincareBall)->Tuple[torch.Tensor,Optional[torch.Tensor],Optional[torch.Tensor]]:
        tan_main = manifold_in_current.logmap0(point_in)
        tan_bound = manifold_in_current.logmap0(boundaries_in) if boundaries_in is not None else None
        tan_desc = manifold_in_current.logmap0(descriptor_in) if descriptor_in is not None else None
        tan_main_out = self.tangent_transform(tan_main)
        tan_bound_out = self.tangent_transform(tan_bound) if tan_bound is not None else None
        tan_desc_out = self.tangent_transform(tan_desc) if tan_desc is not None else None
        point_out = manifold_out_current.expmap0(tan_main_out)
        bound_out = manifold_out_current.expmap0(tan_bound_out) if tan_bound_out is not None else None
        desc_out = manifold_out_current.expmap0(tan_desc_out) if tan_desc_out is not None else None
        outputs = [point_out, bound_out, desc_out]
        final_outputs = []
        names = ["point", "boundaries", "descriptor"]
        for name, out in zip(names, outputs):
            if out is not None and not torch.isfinite(out).all():
                num_non_finite = (~torch.isfinite(out)).sum().item()
                logger.warning(f"NaN/Inf ({num_non_finite}) in HypInterLevelTr output '{name}'. Replacing with 0.")
                final_outputs.append(torch.nan_to_num(out, nan=0., posinf=0., neginf=0.))
            else:
                final_outputs.append(out)
        return tuple(final_outputs)

class HyperbolicWuBuNestingLevel(nn.Module):
    """Represents a single level in the Hyperbolic WuBu Nesting architecture."""
    def __init__(self, level_idx:int, dim:int, config:Dict, initial_curvature:float):
        super().__init__()
        self.level_idx=level_idx
        self.dim=dim
        self.config=config
        self.initial_curvature=initial_curvature
        self.use_ld=config.get("use_level_descriptors",True)
        self.use_spread=config.get("use_level_spread",True)
        self.dropout=config.get("dropout",0.1)
        self.ld_init_scale=config.get("level_descriptor_init_scale",1e-5)
        self.relative_vector_aggregation=config.get("relative_vector_aggregation","mean")
        self.min_curvature=max(EPS,config.get("curvature_min_value",EPS))
        self.min_scale=max(EPS,config.get("scale_min_value",EPS))
        self.min_spread=max(EPS,config.get("spread_min_value",EPS))
        def _init_unconstrained(value,min_val):
            value_clamped=max(float(value),min_val+EPS)
            y = value_clamped - min_val
            try:
                if y < 1e-6:
                    unconstrained_val = math.log(y + EPS)
                else:
                    unconstrained_val = math.log(math.expm1(y))
            except (OverflowError, ValueError) as e:
                logger.error(f"Error initializing unconstrained param (val={value}, min={min_val}, y={y}): {e}. Fallback.")
                unconstrained_val=math.log(EPS)
            return torch.tensor(unconstrained_val,dtype=torch.float)

        learn_c=config.get("learnable_curvature",True)
        init_c=self.initial_curvature
        if learn_c:
            self.log_curvature_unconstrained=nn.Parameter(_init_unconstrained(init_c,self.min_curvature))
        else:
            self.register_buffer('log_curvature_unconstrained',_init_unconstrained(init_c,self.min_curvature))

        learn_s=config.get("learnable_scales",True)
        init_scales_list = config.get("initial_scales", [1.0] * (level_idx + 1))
        init_s = init_scales_list[level_idx] if level_idx < len(init_scales_list) else 1.0
        if learn_s:
            self.log_scale_unconstrained=nn.Parameter(_init_unconstrained(init_s,self.min_scale))
        else:
            self.register_buffer('log_scale_unconstrained',_init_unconstrained(init_s,self.min_scale))

        learn_spread=config.get("learnable_spread",True) and self.use_spread
        init_spread_list = config.get("initial_spread_values", [0.1] * (level_idx + 1))
        init_spread = init_spread_list[level_idx] if level_idx < len(init_spread_list) else 0.1
        spread_param_value=_init_unconstrained(init_spread,self.min_spread)
        if learn_spread:
            self.log_spread_unconstrained=nn.Parameter(spread_param_value)
        elif self.use_spread:
            self.register_buffer('log_spread_unconstrained',spread_param_value)
        else:
            self.register_parameter('log_spread_unconstrained', None)

        initial_manifold=PoincareBall(c=self.initial_curvature)
        if self.use_ld:
            self.level_descriptor_param=nn.Parameter(torch.Tensor(dim))
            self.level_descriptor_param.manifold=initial_manifold
            with torch.no_grad():
                initial_manifold.init_weights(self.level_descriptor_param,irange=self.ld_init_scale)
        else:
            self.register_parameter('level_descriptor_param',None)

        num_boundaries_list = config.get("boundary_points_per_level", [8] * (level_idx + 1))
        self.num_boundaries = num_boundaries_list[level_idx] if level_idx < len(num_boundaries_list) else 8
        self.boundary_manifold=BoundaryManifoldHyperbolic(level_idx,self.num_boundaries,dim,initial_manifold=initial_manifold)

        comb_in_dim = self.dim
        if self.relative_vector_aggregation not in ['none', None]:
            comb_in_dim += self.dim
        if self.use_ld:
            comb_in_dim += self.dim
        self.tangent_combiner=nn.Sequential(nn.Linear(comb_in_dim,self.dim))
        logger.info(f"Level {level_idx} TgtCombiner: Linear {comb_in_dim}->{self.dim}")

        self.use_flow=config.get("use_tangent_flow",True)
        self.tangent_flow=None
        self.flow_scale=0.
        if self.use_flow:
            flow_h_dim=max(16,int(dim*config.get("tangent_flow_hidden_dim_ratio",0.5)))
            flow_type=config.get("tangent_flow_type","mlp").lower()
            if flow_type=='mlp':
                self.tangent_flow=nn.Sequential(nn.Linear(dim,flow_h_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(flow_h_dim,dim))
                logger.info(f"Level {level_idx} TangentFlow: MLP({flow_h_dim})")
            elif flow_type=='linear':
                self.tangent_flow=nn.Linear(dim,dim)
                logger.info(f"Level {level_idx} TangentFlow: Linear")
            elif flow_type not in ['none', None]:
                logger.warning(f"L{level_idx}: Unsupported tangent_flow_type '{flow_type}'. Disabling.")
                self.use_flow=False
            if self.use_flow and self.tangent_flow is not None:
                self.flow_scale=config.get("tangent_flow_scale",1.0)
                self.tangent_flow.apply(init_weights)
        self.tangent_combiner.apply(init_weights)

    def get_curvature(self)->torch.Tensor:
        return F.softplus(self.log_curvature_unconstrained)+self.min_curvature

    def get_scale(self)->torch.Tensor:
        return F.softplus(self.log_scale_unconstrained)+self.min_scale

    def get_spread(self)->torch.Tensor:
        if self.use_spread and self.log_spread_unconstrained is not None:
            return F.softplus(self.log_spread_unconstrained)+self.min_spread
        else:
            return torch.tensor(self.min_spread, device=self.log_curvature_unconstrained.device, dtype=self.log_curvature_unconstrained.dtype)

    def forward(self,point_in:torch.Tensor,relative_vectors_tangent_in:Optional[torch.Tensor],descriptor_point_in:Optional[torch.Tensor],sigma_in:Optional[torch.Tensor])->Tuple[torch.Tensor,torch.Tensor,Optional[torch.Tensor],Optional[torch.Tensor],torch.Tensor]:
        if point_in.dim() == 3:
            B, S, Din = point_in.shape
        elif point_in.dim() == 2:
            B, Din = point_in.shape
            S = 1
        else:
            raise ValueError(f"Level {self.level_idx}: Invalid input point dimensions: {point_in.shape}")
        dev = point_in.device
        dtype = next(iter(self.parameters())).dtype if len(list(self.parameters())) > 0 else torch.float32
        assert Din==self.dim, f"Level {self.level_idx}: Input dimension {Din} != level dimension {self.dim}"

        cur_c=self.get_curvature().to(dev)
        cur_s=self.get_scale().to(dev)
        cur_spread=self.get_spread().to(dev)
        cur_m=PoincareBall(c=cur_c)
        if self.level_descriptor_param is not None:
            self.level_descriptor_param.manifold=cur_m
        self.boundary_manifold.set_current_manifold(cur_m)

        point_in_proj=cur_m.proju(point_in)
        tan_main=cur_m.logmap0(point_in_proj)
        tan_rel=relative_vectors_tangent_in if relative_vectors_tangent_in is not None else torch.zeros_like(tan_main)
        tan_ld_comb = torch.zeros_like(tan_main)
        ld_point_self = None
        if self.use_ld and self.level_descriptor_param is not None:
            ld_point_self = cur_m.proju(self.level_descriptor_param)
            tan_ld_self = cur_m.logmap0(ld_point_self)
            if tan_ld_self.dim() == 1 and tan_main.dim() == 3:
                tan_ld_comb = tan_ld_self.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
            elif tan_ld_self.dim() == 1 and tan_main.dim() == 2:
                tan_ld_comb = tan_ld_self.unsqueeze(0).expand(tan_main.shape[0], -1)
            else:
                tan_ld_comb = tan_ld_self
        if descriptor_point_in is not None:
            desc_in_proj=cur_m.proju(descriptor_point_in)
            tan_ld_in=cur_m.logmap0(desc_in_proj)
            tan_ld_comb=tan_ld_comb+tan_ld_in

        inputs_comb=[tan_main.to(dtype)]
        if self.relative_vector_aggregation not in ['none', None]:
            inputs_comb.append(tan_rel.to(dtype))
        if self.use_ld:
            inputs_comb.append(tan_ld_comb.to(dtype))

        assert inputs_comb, f"Level {self.level_idx}: No inputs to tangent combiner!"
        comb_input=torch.cat(inputs_comb,dim=-1)
        exp_comb_dim=self.tangent_combiner[0].in_features
        assert comb_input.shape[-1]==exp_comb_dim, f"Level {self.level_idx} Combiner input dim mismatch: expected {exp_comb_dim}, got {comb_input.shape[-1]}"

        v_comb_tan=self.tangent_combiner(comb_input)
        if self.use_flow and self.tangent_flow is not None:
            flow_disp=self.tangent_flow(v_comb_tan)
            v_comb_tan=v_comb_tan+flow_disp*self.flow_scale

        scaled_tan_out = v_comb_tan * cur_s
        point_out = cur_m.expmap0(scaled_tan_out)
        tan_out = cur_m.logmap0(point_out)
        bound_pts_out = self.boundary_manifold.get_points()
        ld_out = None
        if ld_point_self is not None:
            if ld_point_self.dim() == 1 and point_out.dim() == 3:
                ld_out = ld_point_self.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
            elif ld_point_self.dim() == 1 and point_out.dim() == 2:
                ld_out = ld_point_self.unsqueeze(0).expand(point_out.shape[0], -1)
            else:
                ld_out = ld_point_self
        sigma_out = cur_spread

        final_outputs=[]
        out_vars=[point_out,tan_out,ld_out,bound_pts_out,sigma_out]
        names=["point_out","tan_out","ld_point","boundaries","sigma"]
        for name,out_tensor in zip(names,out_vars):
            if out_tensor is not None:
                if not torch.isfinite(out_tensor).all():
                    num_non_finite = (~torch.isfinite(out_tensor)).sum().item()
                    logger.warning(f"NaN/Inf ({num_non_finite}) in Level {self.level_idx} output '{name}'. Replacing with 0.")
                    out_tensor=torch.nan_to_num(out_tensor,nan=0., posinf=0., neginf=0.)
                final_outputs.append(out_tensor.to(dtype))
            else:
                final_outputs.append(None)

        # Make sure the tuple length matches the return signature even if items are None
        while len(final_outputs) < 5:
            final_outputs.append(None)

        # Ensure returned tensors have the correct shapes if they exist
        pt_ret = final_outputs[0] if final_outputs[0] is not None else torch.zeros((B,S,self.dim), device=dev, dtype=dtype)
        tan_ret = final_outputs[1] if final_outputs[1] is not None else torch.zeros((B,S,self.dim), device=dev, dtype=dtype)
        ld_ret = final_outputs[2] # Can be None
        bnd_ret = final_outputs[3] # Can be None
        sig_ret = final_outputs[4] if final_outputs[4] is not None else torch.tensor(self.min_spread, device=dev, dtype=dtype)

        return pt_ret, tan_ret, ld_ret, bnd_ret, sig_ret


class FullyHyperbolicWuBuNestingModel(nn.Module):
    """The core WuBu Nesting model operating fully in hyperbolic space."""
    def __init__(self, input_dim: int, output_dim: int, config: Dict):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.config=config
        self.num_levels=config.get("num_levels",3)
        assert self.num_levels>0
        self.hyperbolic_dims=config.get("hyperbolic_dims",[128]*self.num_levels)
        self.initial_curvatures=config.get("initial_curvatures",[1.0]*self.num_levels)
        self.dropout=config.get("dropout",0.1)
        self.relative_vector_aggregation=config.get("relative_vector_aggregation","mean")
        self.aggregation_method=config.get("aggregation_method","concat_tangent")

        list_args={'hyperbolic_dims':self.num_levels,'initial_curvatures':self.num_levels,'initial_scales':self.num_levels,'boundary_points_per_level':self.num_levels,'initial_spread_values':self.num_levels}
        num_trans=max(0,self.num_levels-1)
        trans_list_args={'transform_types':num_trans,'transform_hidden_dims':num_trans}

        for k,L in list_args.items():
            if k not in config or len(config[k])!=L:
                raise ValueError(f"Config '{k}' missing or needs length {L} for {self.num_levels} levels. Got: {config.get(k)}")
        for k,L in trans_list_args.items():
            if k not in config or len(config[k])!=L:
                raise ValueError(f"Config '{k}' missing or needs length {L} for {num_trans} transforms. Got: {config.get(k)}")

        self.input_tangent_to_H0_tangent=nn.Linear(input_dim,self.hyperbolic_dims[0])
        self.levels=nn.ModuleList()
        for i in range(self.num_levels):
            self.levels.append(HyperbolicWuBuNestingLevel(level_idx=i, dim=self.hyperbolic_dims[i], config=self.config, initial_curvature=self.initial_curvatures[i]))

        self.transforms=nn.ModuleList()
        trans_types=config.get("transform_types",[])
        trans_hdims=config.get("transform_hidden_dims",[])
        for i in range(num_trans):
            m_in=PoincareBall(c=self.initial_curvatures[i])
            m_out=PoincareBall(c=self.initial_curvatures[i+1])
            self.transforms.append(HyperbolicInterLevelTransform(in_dim=self.hyperbolic_dims[i], out_dim=self.hyperbolic_dims[i+1], manifold_in=m_in, manifold_out=m_out, transform_type=trans_types[i], hidden_dim=trans_hdims[i], dropout=self.dropout))

        assert self.aggregation_method=="concat_tangent", f"Aggregation method '{self.aggregation_method}' not supported. Use 'concat_tangent'."
        comb_tan_dim=sum(self.hyperbolic_dims)
        self.tangent_to_output=nn.Linear(comb_tan_dim,output_dim)

        self.input_tangent_to_H0_tangent.apply(init_weights)
        self.tangent_to_output.apply(init_weights)

        total_p=sum(p.numel() for p in self.parameters())
        train_p=sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"FullyHypWuBuModel init: {self.num_levels} levels.")
        logger.info(f" Architecture: InTgt({input_dim}) -> H0Tgt({self.hyperbolic_dims[0]}) | Levels Dims:{self.hyperbolic_dims} | Agg:{self.aggregation_method} -> OutTgt({output_dim})")
        logger.info(f" Total Params:{total_p:,} | Trainable Params:{train_p:,}")

    def forward(self, x_tangent_in: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_tangent_in.dim() == 2:
            x_tangent_in = x_tangent_in.unsqueeze(0)
        if padding_mask is not None and padding_mask.dim()==1:
            padding_mask = padding_mask.unsqueeze(0)

        B,S,Din=x_tangent_in.shape
        assert Din == self.input_dim
        dev=x_tangent_in.device
        dtype=next(iter(self.parameters())).dtype if len(list(self.parameters())) > 0 else torch.float32

        cur_tan = self.input_tangent_to_H0_tangent(x_tangent_in)
        m0=PoincareBall(c=self.levels[0].get_curvature().to(dev))
        cur_pt = m0.expmap0(cur_tan)
        level_tan_outs = []
        agg_rel_vecs_tan = None
        cur_desc_pt = None
        cur_sigma = None

        for i in range(self.num_levels):
            level = self.levels[i]
            cur_m = PoincareBall(c=level.get_curvature().to(dev))
            cur_pt_proj = cur_m.proju(cur_pt)
            desc_pt_proj = cur_m.proju(cur_desc_pt) if cur_desc_pt is not None else None

            pt_out, tan_out, ld_pt_out, bound_pts, sigma_out = level(point_in=cur_pt_proj, relative_vectors_tangent_in=agg_rel_vecs_tan, descriptor_point_in=desc_pt_proj, sigma_in=cur_sigma)
            level_tan_outs.append(tan_out)

            if i < self.num_levels - 1:
                trans = self.transforms[i]
                m_next = PoincareBall(c=self.levels[i+1].get_curvature().to(dev))
                pt_next, bound_trans, ld_next = trans(point_in=pt_out, boundaries_in=bound_pts, descriptor_in=ld_pt_out, manifold_in_current=cur_m, manifold_out_current=m_next)

                agg_rel_vecs_tan = None
                has_bounds = bound_trans is not None and bound_trans.numel() > 0
                if has_bounds and self.relative_vector_aggregation not in ['none', None]:
                    pt_next_proj = m_next.proju(pt_next)
                    bound_trans_proj = m_next.proju(bound_trans)
                    tan_main_next = m_next.logmap0(pt_next_proj)
                    tan_bounds_next = m_next.logmap0(bound_trans_proj)

                    # Ensure tan_bounds_next can be broadcasted for subtraction
                    if tan_bounds_next.dim() == 2: # Shape (N_bounds, D)
                        tan_bounds_next = tan_bounds_next.unsqueeze(0).unsqueeze(0) # Shape (1, 1, N_bounds, D)
                    elif tan_bounds_next.dim() != 4:
                        logger.error(f"Unexpected boundary tangent dim {tan_bounds_next.dim()}. Cannot calc relative vecs.")
                        has_bounds=False

                    if has_bounds:
                        # Ensure pt_next_proj matches expected B, S, D shape for broadcasting
                        if tan_main_next.dim() == 2: # If B=1, S=1 case or just B, D ?
                            if tan_main_next.shape[0] == B: # Shape (B, D) - needs S dim
                                tan_main_next = tan_main_next.unsqueeze(1) # (B, 1, D)
                            else: # Shape (1, D) or similar
                                tan_main_next = tan_main_next.unsqueeze(0).unsqueeze(0) # (1, 1, D)
                        elif tan_main_next.dim() != 3: # Expecting (B, S, D)
                            logger.error(f"Unexpected pt_next tangent dim {tan_main_next.dim()}. Cannot calc relative vecs.")
                            has_bounds=False

                    if has_bounds:
                        try:
                            # Broadcast tan_main_next (B, S, 1, D) with tan_bounds_next (1, 1, N_bounds, D)
                            # Use simple tangent space subtraction as approximation
                            rel_tan_origin = tan_main_next.unsqueeze(2) - tan_bounds_next # (B, S, 1, D) - (1, 1, Nb, D) -> (B, S, Nb, D)

                            agg_m = self.relative_vector_aggregation
                            if agg_m=="mean":
                                agg_rel_vecs_tan = torch.mean(rel_tan_origin, dim=2)
                            elif agg_m=="sum":
                                agg_rel_vecs_tan = torch.sum(rel_tan_origin, dim=2)
                            else:
                                logger.warning(f"Unsupported relative aggregation '{agg_m}'. Setting to None.")
                                agg_rel_vecs_tan = None
                        except Exception as rel_vec_err:
                            logger.error(f"Error calculating relative vectors L{i+1}: {rel_vec_err}", exc_info=True)
                            agg_rel_vecs_tan = None

                        if agg_rel_vecs_tan is not None and not torch.isfinite(agg_rel_vecs_tan).all():
                            nNaN=torch.isnan(agg_rel_vecs_tan).sum().item()
                            nInf=torch.isinf(agg_rel_vecs_tan).sum().item()
                            logger.warning(f"NaN/Inf ({nNaN}/{nInf}) in L{i+1} aggregated relative vectors. Replacing with 0.")
                            agg_rel_vecs_tan = torch.zeros_like(tan_main_next) # Use tan_main_next for shape

                cur_pt = pt_next
                cur_desc_pt = ld_next
                cur_sigma = sigma_out

        try:
            compat_tans=[]
            for t_idx, t in enumerate(level_tan_outs):
                level_dim = self.hyperbolic_dims[t_idx]
                if t is None:
                    logger.error(f"Tangent output from Level {t_idx} is None. Replacing with zeros.")
                    t = torch.zeros((B, S, level_dim), device=dev, dtype=dtype)
                elif not torch.isfinite(t).all():
                    nNaN=torch.isnan(t).sum().item()
                    nInf=torch.isinf(t).sum().item()
                    logger.warning(f"NaN/Inf ({nNaN}/{nInf}) in Tangent output from Level {t_idx}. Replacing with 0.")
                    t = torch.nan_to_num(t, nan=0., posinf=0., neginf=0.)
                if t.shape[-1] != level_dim:
                    logger.error(f"Tangent output dim mismatch Level {t_idx}: Expected {level_dim}, Got {t.shape[-1]}. Reshaping/Padding - THIS IS LIKELY AN ERROR.")
                    if t.shape[-1] < level_dim:
                        t = F.pad(t, (0, level_dim - t.shape[-1]))
                    else:
                        t = t[..., :level_dim]
                compat_tans.append(t.to(dtype))
            agg_tan = torch.cat(compat_tans, dim=-1)
            expected_agg_dim = sum(self.hyperbolic_dims)
            assert agg_tan.shape[-1] == expected_agg_dim
        except Exception as cat_err:
            logger.error(f"Error during tangent vector aggregation: {cat_err}. Shapes: {[t.shape if t is not None else 'None' for t in level_tan_outs]}", exc_info=True)
            return torch.zeros((B, S, self.output_dim), device=dev, dtype=dtype)

        final_out_tan = self.tangent_to_output(agg_tan)
        if padding_mask is not None:
            mask = padding_mask.to(dtype=torch.bool, device=dev)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            assert mask.shape == (B, S)
            final_out_tan = final_out_tan.masked_fill(mask.unsqueeze(-1), 0.)

        if not torch.isfinite(final_out_tan).all():
            nNaN=torch.isnan(final_out_tan).sum().item()
            nInf=torch.isinf(final_out_tan).sum().item()
            logger.error(f"NaN/Inf ({nNaN}/{nInf}) in final WuBu output tangent vector! Replacing with 0.")
            final_out_tan = torch.nan_to_num(final_out_tan, nan=0., posinf=0., neginf=0.)

        return final_out_tan

# =====================================================================
# Sequence Model Components
# =====================================================================
class WuBuLocalEncoder(nn.Module):
    """Encodes nucleotide sequence into tangent space vectors using Transformers."""
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, manifold: PoincareBall, dropout: float=0.1, vocab_size: int=NUCLEOTIDE_VOCAB_SIZE, max_seq_len: int=1024):
        super().__init__()
        self.hidden_size=hidden_size
        assert isinstance(manifold,PoincareBall)
        self.manifold=manifold
        self.vocab_size=vocab_size
        self.max_seq_len=max_seq_len
        self.nucleotide_embeddings=HyperbolicEmbedding(vocab_size,hidden_size,manifold)
        self.positional_encoding=nn.Embedding(max_seq_len,hidden_size)
        nn.init.normal_(self.positional_encoding.weight,std=0.02)
        encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4, dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True)
        self.transformer=nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm_out=nn.LayerNorm(hidden_size,eps=1e-6)
        self.dropout_embed=nn.Dropout(dropout)
        self.transformer.apply(init_weights)
        self.norm_out.apply(init_weights)
        logger.info(f"WuBuLocalEncoder (Nuc) init: Vocab={vocab_size}, MaxLen={max_seq_len}, Hidden={hidden_size}, Layers={num_layers}, Heads={num_heads}")

    def forward(self,nucleotide_seq:torch.Tensor,padding_mask:Optional[torch.Tensor]=None)->torch.Tensor:
        B,S=nucleotide_seq.shape
        dev=nucleotide_seq.device
        dtype=next(iter(self.parameters())).dtype if len(list(self.parameters())) > 0 else torch.float32

        # --- Refactored Truncation Logic ---
        if S > self.max_seq_len:
            logger.warning(f"Encoder input len {S} > max {self.max_seq_len}. Truncating.")
            nucleotide_seq = nucleotide_seq[:, :self.max_seq_len]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :self.max_seq_len]
            S = self.max_seq_len
        # --- End Refactoring ---

        nuc_clamp=torch.clamp(nucleotide_seq.long(),0,self.vocab_size-1)
        x_hyp=self.nucleotide_embeddings(nuc_clamp).to(dtype)
        x_tan=self.manifold.logmap0(x_hyp)

        pos=torch.arange(0,S,device=dev).unsqueeze(0).expand(B,-1)
        pos_emb=self.positional_encoding(pos)
        x_tan=self.dropout_embed(x_tan+pos_emb.to(dtype))

        proc_tan=self.transformer(x_tan, src_key_padding_mask=padding_mask)
        norm_out_tan=self.norm_out(proc_tan)

        if not torch.isfinite(norm_out_tan).all():
            num_non_finite = (~torch.isfinite(norm_out_tan)).sum().item()
            logger.warning(f"NaN/Inf ({num_non_finite}) in WuBuLocalEncoder output. Replacing with 0.")
            norm_out_tan=torch.nan_to_num(norm_out_tan,nan=0., posinf=0., neginf=0.)

        if padding_mask is not None:
            norm_out_tan=norm_out_tan.masked_fill(padding_mask.unsqueeze(-1),0.)

        return norm_out_tan

class WuBuLocalDecoder(nn.Module):
    """Decodes from tangent space memory and target sequence context."""
    def __init__(self, hidden_size: int, global_tangent_dim: int, num_layers: int, num_heads: int, manifold: PoincareBall, dropout: float=0.1, vocab_size: int=NUCLEOTIDE_VOCAB_SIZE, max_decode_len: int=2048):
        super().__init__()
        self.hidden_size=hidden_size
        self.global_tangent_dim=global_tangent_dim
        assert isinstance(manifold,PoincareBall)
        self.manifold=manifold
        self.vocab_size=vocab_size
        self.max_decode_len=max_decode_len
        self.nucleotide_embeddings=HyperbolicEmbedding(vocab_size,hidden_size,manifold)
        self.positional_encoding=nn.Embedding(max_decode_len,hidden_size)
        nn.init.normal_(self.positional_encoding.weight,std=0.02)
        self.memory_projection=nn.Sequential(nn.Linear(global_tangent_dim,hidden_size*2), nn.GELU(), nn.Linear(hidden_size*2,hidden_size), nn.LayerNorm(hidden_size,eps=1e-6))
        decoder_layer=nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4, dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True)
        self.decoder_norm=nn.LayerNorm(hidden_size,eps=1e-6)
        self.transformer=nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=self.decoder_norm)
        self.nucleotide_pred=nn.Linear(hidden_size,self.vocab_size)
        self.dropout_embed=nn.Dropout(dropout)
        self.memory_projection.apply(init_weights)
        self.transformer.apply(init_weights)
        self.nucleotide_pred.apply(init_weights)
        logger.info(f"WuBuLocalDecoder (Nuc) init: Vocab={self.vocab_size}, MaxLen={self.max_decode_len}, Hidden={hidden_size}, MemDim={global_tangent_dim}, Layers={num_layers}, Heads={num_heads}")

    def _generate_square_subsequent_mask(self,sz:int,device:torch.device)->torch.Tensor:
        return torch.triu(torch.full((sz, sz), True, device=device, dtype=torch.bool), diagonal=1)

    def forward(self,tgt_nucleotide_seq:torch.Tensor,memory_tangent:torch.Tensor,tgt_mask:Optional[torch.Tensor]=None,tgt_key_padding_mask:Optional[torch.Tensor]=None,memory_key_padding_mask:Optional[torch.Tensor]=None)->torch.Tensor:
        B,T = tgt_nucleotide_seq.shape
        dev = tgt_nucleotide_seq.device
        Bm, S_mem, Dmem = memory_tangent.size()
        dtype = next(iter(self.parameters())).dtype if len(list(self.parameters())) > 0 else torch.float32
        assert B==Bm
        assert Dmem==self.global_tangent_dim

        # --- Refactored Truncation Logic ---
        if T > self.max_decode_len:
            logger.warning(f"Decoder target len {T} > max {self.max_decode_len}. Truncating.")
            tgt_nucleotide_seq = tgt_nucleotide_seq[:, :self.max_decode_len]
            if tgt_key_padding_mask is not None:
                tgt_key_padding_mask = tgt_key_padding_mask[:, :self.max_decode_len]
            T = self.max_decode_len
        # --- End Refactoring ---

        tgt_clamp=torch.clamp(tgt_nucleotide_seq.long(),0,self.vocab_size-1)
        tgt_hyp=self.nucleotide_embeddings(tgt_clamp).to(dtype)
        tgt_tan=self.manifold.logmap0(tgt_hyp)

        pos=torch.arange(0,T,device=dev).unsqueeze(0).expand(B,-1)
        pos_clamp=torch.clamp(pos,max=self.positional_encoding.num_embeddings-1)
        pos_emb=self.positional_encoding(pos_clamp).to(dtype)
        tgt_prep_tan=self.dropout_embed(tgt_tan+pos_emb)

        if S_mem == 0:
            proj_mem_tan = torch.zeros(B, 0, self.hidden_size, device=dev, dtype=dtype)
            memory_key_padding_mask = None
            logger.debug("Decoder received empty memory tensor.")
        else:
            proj_mem_tan=self.memory_projection(memory_tangent.to(dtype))

        if memory_key_padding_mask is not None and memory_key_padding_mask.shape != (B, S_mem):
            logger.warning(f"Decoder memory key padding mask shape mismatch. Ignoring.")
            memory_key_padding_mask = None

        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(T, dev)

        tgt_m_bool=tgt_mask.to(dtype=torch.bool) if tgt_mask is not None else None
        tgt_pad_m_bool=tgt_key_padding_mask.to(dtype=torch.bool) if tgt_key_padding_mask is not None else None
        mem_pad_m_bool=memory_key_padding_mask.to(dtype=torch.bool) if memory_key_padding_mask is not None else None

        out_tan = self.transformer(tgt=tgt_prep_tan, memory=proj_mem_tan, tgt_mask=tgt_m_bool, memory_mask=None, tgt_key_padding_mask=tgt_pad_m_bool, memory_key_padding_mask=mem_pad_m_bool)
        nuc_logits = self.nucleotide_pred(out_tan)
        nuc_logits = nuc_logits.float()

        if not torch.isfinite(nuc_logits).all():
            num_non_finite = (~torch.isfinite(nuc_logits)).sum().item()
            logger.warning(f"NaN/Inf ({num_non_finite}) in final decoder logits. Replacing with 0.")
            nuc_logits=torch.nan_to_num(nuc_logits,nan=0.,posinf=0.,neginf=0.)

        return nuc_logits

class WuBuNestingSequenceModel(nn.Module):
    """Combines Encoder, WuBu Core, and Decoder for sequence modeling."""
    def __init__(self, wubu_config: Dict, sequence_config: Dict):
        super().__init__()
        self.wubu_config=wubu_config
        self.sequence_config=sequence_config
        self.local_hidden_size=sequence_config["local_hidden_size"]
        self.decoder_memory_dim=sum(wubu_config["hyperbolic_dims"])
        if sequence_config["decoder_memory_dim"] != self.decoder_memory_dim:
            logger.warning(f"Seq config decoder_memory_dim ({sequence_config['decoder_memory_dim']}) != sum WuBu dims ({self.decoder_memory_dim}). Using {self.decoder_memory_dim}.")
        self.sequence_config["decoder_memory_dim"] = self.decoder_memory_dim
        self.context_window=sequence_config["context_window"]
        self.nucleotide_vocab_size=sequence_config.get("nucleotide_vocab_size",NUCLEOTIDE_VOCAB_SIZE)
        self.encoder_max_seq_len=sequence_config.get("encoder_max_seq_len",1024)
        self.decoder_max_seq_len=sequence_config.get("decoder_max_seq_len",2048)

        try:
            first_lvl_c=wubu_config["initial_curvatures"][0]
        except IndexError:
            first_lvl_c=1.
            logger.warning("WuBu initial_curvatures missing. Defaulting c=1.0 for shared manifold.")

        self.shared_manifold=PoincareBall(c=first_lvl_c)
        logger.info(f"WuBuModel Shared Manifold (Embeddings): {self.shared_manifold.name}")

        self.local_encoder=WuBuLocalEncoder(hidden_size=self.local_hidden_size, num_layers=sequence_config.get("num_encoder_layers",3), num_heads=sequence_config.get("num_encoder_heads",8), manifold=self.shared_manifold, dropout=wubu_config.get("dropout",0.1), vocab_size=self.nucleotide_vocab_size, max_seq_len=self.encoder_max_seq_len)
        self.wubu_model=FullyHyperbolicWuBuNestingModel(input_dim=self.local_hidden_size, output_dim=self.decoder_memory_dim, config=self.wubu_config)
        self.local_decoder=WuBuLocalDecoder(hidden_size=self.local_hidden_size, global_tangent_dim=self.decoder_memory_dim, num_layers=sequence_config.get("num_decoder_layers",6), num_heads=sequence_config.get("num_decoder_heads",8), manifold=self.shared_manifold, dropout=wubu_config.get("dropout",0.1), vocab_size=self.nucleotide_vocab_size, max_decode_len=self.decoder_max_seq_len)
        logger.info("WuBuNestingSequenceModel (Nucleotide) Initialized.")

    def forward(self,nucleotide_seq:torch.Tensor,target_nucleotide_seq:Optional[torch.Tensor]=None,input_padding_mask:Optional[torch.Tensor]=None,target_padding_mask:Optional[torch.Tensor]=None)->torch.Tensor:
        B,S_in = nucleotide_seq.shape
        dev = nucleotide_seq.device
        dtype = next(iter(self.parameters())).dtype if len(list(self.parameters())) > 0 else torch.float32

        try:
            enc_tan=self.local_encoder(nucleotide_seq,padding_mask=input_padding_mask)
            if not torch.isfinite(enc_tan).all():
                logger.error("Encoder output NaN/Inf. Zeroing.")
                enc_tan=torch.nan_to_num(enc_tan,nan=0., posinf=0., neginf=0.)
        except Exception as enc_err:
            logger.error(f"Error in Local Encoder: {enc_err}",exc_info=True)
            T_fall = target_nucleotide_seq.size(1) if target_nucleotide_seq is not None else 0
            return torch.zeros((B,T_fall,self.nucleotide_vocab_size),device=dev,dtype=torch.float32)

        try:
            dec_mem_tan=self.wubu_model(x_tangent_in=enc_tan,padding_mask=input_padding_mask)
            if not torch.isfinite(dec_mem_tan).all():
                logger.error("WuBu Core output NaN/Inf. Zeroing.")
                dec_mem_tan=torch.nan_to_num(dec_mem_tan,nan=0., posinf=0., neginf=0.)
        except Exception as wubu_err:
            logger.error(f"Error in WuBu Core Model: {wubu_err}",exc_info=True)
            T_fall = target_nucleotide_seq.size(1) if target_nucleotide_seq is not None else 0
            return torch.zeros((B,T_fall,self.nucleotide_vocab_size),device=dev,dtype=torch.float32)

        if target_nucleotide_seq is None:
            logger.warning("target_nucleotide_seq is None. Generation not implemented in forward(). Returning zeros.")
            return torch.zeros((B,0,self.nucleotide_vocab_size),device=dev,dtype=torch.float32)

        T_tgt = target_nucleotide_seq.size(1)
        if T_tgt == 0:
            return torch.zeros((B,0,self.nucleotide_vocab_size),device=dev,dtype=torch.float32)

        try:
            final_nuc_logits=self.local_decoder(tgt_nucleotide_seq=target_nucleotide_seq, memory_tangent=dec_mem_tan, tgt_key_padding_mask=target_padding_mask, memory_key_padding_mask=input_padding_mask)
            if not torch.isfinite(final_nuc_logits).all():
                logger.error("Decoder output logits NaN/Inf. Zeroing.")
                final_nuc_logits=torch.nan_to_num(final_nuc_logits,nan=0., posinf=0., neginf=0.)
        except Exception as dec_err:
            logger.error(f"Error in Local Decoder: {dec_err}",exc_info=True)
            return torch.zeros((B,T_tgt,self.nucleotide_vocab_size),device=dev,dtype=torch.float32)

        return final_nuc_logits

    @staticmethod
    def compute_loss(logits:torch.Tensor,targets:torch.Tensor,mask:Optional[torch.Tensor]=None,smoothing:float=0.1,vocab_size:int=NUCLEOTIDE_VOCAB_SIZE)->torch.Tensor:
        B,S,V = logits.shape
        if V != vocab_size:
            logger.warning(f"Logits vocab size ({V}) != expected ({vocab_size}). Check model output.")
        assert targets.shape == (B, S), f"Target shape {targets.shape} mismatch Logits {logits.shape}"
        if S <= 1:
            return torch.tensor(0., device=logits.device, requires_grad=True, dtype=logits.dtype)

        logits_shift = logits[:, :-1, :].contiguous()
        targets_shift = targets[:, 1:].contiguous()
        logits_flat = logits_shift.view(-1, vocab_size)
        targets_flat = targets_shift.view(-1)
        targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)

        if not torch.isfinite(logits_flat).all():
            num_non_finite = (~torch.isfinite(logits_flat)).sum().item()
            logger.error(f"NaN/Inf ({num_non_finite}) in logits to loss. Returning high loss.")
            return torch.tensor(100.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

        loss = F.cross_entropy(logits_flat.float(), targets_flat.long(), label_smoothing=smoothing if 0. < smoothing < 1. else 0., reduction='none')

        if not torch.isfinite(loss).all():
            num_non_finite = (~torch.isfinite(loss)).sum().item()
            logger.error(f"NaN/Inf ({num_non_finite}) during loss calculation. Clamping loss.")
            loss = torch.nan_to_num(loss, nan=100.0, posinf=100.0, neginf=-100.0)

        mean_loss:torch.Tensor
        if mask is not None:
            assert mask.shape == (B, S), f"Mask shape {mask.shape} mismatch target {(B,S)}"
            mask_shift = mask[:, 1:].contiguous()
            mask_flat = mask_shift.view(-1)
            mask_bool = ~mask_flat.bool() # Mask is True for PAD, False for KEEP. Loss wants mask=True for ignore.
            loss = loss.masked_fill(~mask_bool, 0.) # Zero out PAD tokens before sum/mean
            num_active_tokens = mask_bool.sum().clamp(min=1.)
            mean_loss = loss.sum() / num_active_tokens
        else:
            mean_loss = loss.mean()

        if not torch.isfinite(mean_loss):
            logger.error(f"Final mean loss is NaN/Inf ({mean_loss}). Returning high loss.")
            return torch.tensor(100.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

        return mean_loss.to(logits.dtype)

    @torch.no_grad()
    def generate(self,seed_nucleotides:torch.Tensor,max_length:int=100,temperature:float=1.,top_k:Optional[int]=None,top_p:Optional[float]=None)->torch.Tensor:
        self.eval()
        dev = next(iter(self.parameters())).device
        amp_ctx = contextlib.nullcontext()
        assert seed_nucleotides.dim()==2 and seed_nucleotides.size(0)==1, "Generation supports B=1 only."

        seed_nuc = seed_nucleotides.to(dev).long()
        B, S_seed = seed_nuc.size()
        assert S_seed > 0

        generated_seq = seed_nuc.clone()
        vocab_size = self.nucleotide_vocab_size
        gen_iter=tqdm(range(max_length),desc="Generating Nucleotides",total=max_length,unit="nt",leave=False)

        for _ in gen_iter:
            current_context = generated_seq.long()
            input_padding_mask = None
            target_padding_mask = None

            with torch.no_grad(), amp_ctx:
                logits_all = self(nucleotide_seq=current_context, target_nucleotide_seq=current_context, input_padding_mask=input_padding_mask, target_padding_mask=target_padding_mask)

            if logits_all is None or logits_all.numel()==0 or logits_all.shape[1]==0:
                logger.warning("Logits during generation are empty/None. Stopping.")
                break
            if not torch.isfinite(logits_all).all():
                logger.warning("NaN/Inf in logits during generation. Sampling uniformly.")
                logits_all = torch.zeros_like(logits_all)

            next_token_logits = logits_all[0, -1, :].float()
            if temperature <= 0:
                temperature = EPS

            scaled_logits = next_token_logits / temperature
            filtered_logits = scaled_logits

            if top_k is not None and top_k > 0:
                k = min(top_k, filtered_logits.size(-1))
                if k > 0:
                    vals, _ = torch.topk(filtered_logits, k)
                    threshold = vals[-1]
                    remove_mask = filtered_logits < threshold
                    filtered_logits = filtered_logits.masked_fill(remove_mask, float('-inf'))
                else:
                    filtered_logits.fill_(float('-inf'))

            if top_p is not None and 0. < top_p < 1.:
                sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(filtered_logits, dtype=torch.bool).scatter_(0, sorted_indices, sorted_indices_to_remove)
                filtered_logits = filtered_logits.masked_fill(indices_to_remove, float('-inf'))

            probs = F.softmax(filtered_logits, dim=-1)
            if not torch.isfinite(probs).all() or probs.sum() < EPS:
                logger.warning("Invalid probability dist after filtering. Sampling uniformly.")
                probs = torch.ones_like(next_token_logits) / vocab_size

            next_token_idx = torch.multinomial(probs, num_samples=1)
            generated_seq = torch.cat([generated_seq, next_token_idx.unsqueeze(0)], dim=1)
            # Optional EOS check: if next_token_idx.item() == NUCLEOTIDE_MAP.get('N', 4): break

        if hasattr(gen_iter, 'close'):
            gen_iter.close()
        return generated_seq

# =====================================================================
# Riemannian Optimizer
# =====================================================================
class RiemannianEnhancedSGD(torch.optim.Optimizer):
    """Riemannian SGD optimizer adapted for PyTorch, with optional Q-learning rate/momentum control."""
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0.01, max_grad_norm: float = 1.0, q_learning_config: Optional[Dict]=None):
        assert lr>=0 and momentum>=0 and weight_decay>=0
        defaults=dict(lr=lr, base_lr=lr, momentum=momentum, base_momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.q_controller: Optional[HAKMEMQController]=None
        if isinstance(q_learning_config,dict):
            try:
                self.q_controller=HAKMEMQController(**q_learning_config)
                logger.info("RiSGD: Q-Learning Controller enabled.")
            except Exception as e:
                logger.error(f"Q-Learning Controller init failed: {e}. Disabling.",exc_info=True)
        else:
            logger.info("RiSGD: Q-Learning Controller disabled.")
        self.max_grad_norm=max_grad_norm
        self._step_count=0
        self.current_loss:Optional[float]=None
        self.gradient_stats=GradientStats()

    def zero_grad(self, set_to_none: bool=True):
        super().zero_grad(set_to_none=set_to_none)
        self.gradient_stats.reset()

    def set_current_loss(self, loss:Optional[float]):
        if loss is not None and np.isfinite(loss):
            self.current_loss=loss
        else:
            logger.debug(f"Optimizer received invalid loss: {loss}. Ignoring.")
            self.current_loss = None

    @torch.no_grad()
    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()
            self.set_current_loss(loss.item() if isinstance(loss,torch.Tensor) else float(loss) if loss is not None else None)

        if self.q_controller and self.q_controller.prev_action:
            q_action = self.q_controller.prev_action
            for group in self.param_groups:
                base_lr = group.get('base_lr', group['lr'])
                base_mom = group.get('base_momentum', group['momentum'])
                lr_scale = q_action.get('lr_scale', 1.0)
                mom_scale = q_action.get('momentum_scale', 1.0)
                new_lr = base_lr * lr_scale
                new_mom = base_mom * mom_scale
                group['lr'] = float(np.clip(new_lr, 1e-8, 0.1))
                group['momentum'] = float(np.clip(new_mom, 0.5, 0.999))

        for group in self.param_groups:
            lr = group['lr']
            mom = group['momentum']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad
                state = self.state[p]
                if not torch.isfinite(grad).all():
                    num_non_finite = (~torch.isfinite(grad)).sum().item()
                    logger.error(f"Optim Step Error: Non-finite gradient in param {p.shape} ({num_non_finite} values). Skipping update.")
                    self.gradient_stats.non_finite_grads_in_step += 1
                    if 'momentum_buffer' in state:
                        del state['momentum_buffer']
                    continue

                manifold:Optional[Manifold]=getattr(p,'manifold',None)
                is_riemannian = isinstance(manifold, Manifold)
                if is_riemannian and manifold is not None:
                    p_data = p.data
                    try:
                        r_grad = manifold.egrad2rgrad(p_data, grad)
                    except Exception as rgrad_err:
                        logger.error(f"Riemannian Grad Error {p.shape} on {manifold.name}: {rgrad_err}. Skip.",exc_info=False)
                        self.gradient_stats.non_finite_grads_in_step += 1
                        if 'momentum_buffer' in state:
                            del state['momentum_buffer']
                        continue # Skip this parameter update

                    if not torch.isfinite(r_grad).all():
                        logger.error(f"Non-finite RGrad {p.shape} on {manifold.name}. Skip.")
                        self.gradient_stats.non_finite_grads_in_step += 1
                        if 'momentum_buffer' in state:
                            del state['momentum_buffer']
                        continue # Skip this parameter update

                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(r_grad)
                    buf = state['momentum_buffer']
                    buf.mul_(mom).add_(r_grad)
                    if not torch.isfinite(buf).all():
                        logger.warning(f"Non-finite Riem mom buf {p.shape}. Resetting.")
                        buf.zero_()
                        self.gradient_stats.non_finite_grads_in_step += 1

                    update_tangent = buf
                    if wd != 0:
                        try:
                            log_p0 = manifold.logmap0(p_data)
                            update_tangent = update_tangent.add(log_p0, alpha=wd)
                        except Exception as wd_err:
                            logger.warning(f"Riem WD Error {p.shape}: {wd_err}. Skipping WD.")

                    try:
                        final_update_vector = update_tangent.mul(-lr)
                        new_p_data = manifold.expmap(p_data, final_update_vector)
                        p.data = manifold.proju(new_p_data)
                        if not torch.isfinite(p.data).all():
                            raise ValueError("Retraction resulted in NaN/Inf.")
                    except Exception as retr_err:
                        logger.error(f"Retraction Error {p.shape} on {manifold.name}: {retr_err}. Skip.",exc_info=False)
                        self.gradient_stats.non_finite_grads_in_step += 1
                        if 'momentum_buffer' in state:
                            del state['momentum_buffer']
                        continue # Skip this parameter update
                else: # Standard Euclidean update
                    p_data = p.data
                    d_p = grad
                    if wd != 0:
                        d_p = d_p.add(p_data, alpha=wd)
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    buf = state['momentum_buffer']
                    buf.mul_(mom).add_(d_p)
                    if not torch.isfinite(buf).all():
                        logger.warning(f"Non-finite Euc mom buf {p.shape}. Resetting.")
                        buf.zero_()
                        self.gradient_stats.non_finite_grads_in_step += 1
                    update_step = buf * lr
                    p_data.add_(-update_step)
                    if not torch.isfinite(p.data).all():
                        logger.error(f"Non-finite Euc update {p.shape}. Check LR/WD/Mom.")
                        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=0.0, neginf=0.0)
                        self.gradient_stats.non_finite_grads_in_step += 1
                        if 'momentum_buffer' in state:
                            state['momentum_buffer'].zero_()

        self._step_count += 1
        return loss

    def get_q_info(self)->Dict:
        if hasattr(self,'q_controller') and self.q_controller:
            return self.q_controller.get_info()
        return {"Q-Controller":"Disabled"}

# =====================================================================
# Trainer Class
# =====================================================================
class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, train_loader: DataLoader, val_loader: Optional[DataLoader], grad_accum_steps: int = 1, use_amp: bool = True, log_interval: int = 10, save_interval: int = 1000, checkpoint_dir: str = "checkpoints", wandb_enabled: bool = False, max_grad_norm: float = 1.0, rank: int = 0, world_size: int = 1, detect_anomaly: bool = False, nucleotide_vocab_size: int = NUCLEOTIDE_VOCAB_SIZE):
        self.model=model
        self.optimizer=optimizer
        self.device=device
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.grad_accum_steps=max(1,grad_accum_steps)
        self.rank=rank
        self.world_size=world_size
        self.use_amp = use_amp and torch.cuda.is_available() and hasattr(torch, "amp") and device.type == 'cuda'
        self.scaler=amp.GradScaler(enabled=self.use_amp)
        self.log_interval=log_interval
        self.save_interval=max(1,save_interval) if save_interval > 0 else -1
        self.checkpoint_dir=checkpoint_dir
        self.is_main = self.is_main_process()
        self.wandb_enabled = wandb_enabled and WANDB_AVAILABLE and self.is_main
        self.max_grad_norm=max_grad_norm
        self.global_step=0
        self.current_epoch=0
        self.last_val_metrics:Optional[Dict[str,float]]=None
        self.detect_anomaly=detect_anomaly
        self.nucleotide_vocab_size=nucleotide_vocab_size
        self.has_grad_stats = hasattr(self.optimizer, 'gradient_stats') and isinstance(getattr(self.optimizer, 'gradient_stats', None), GradientStats)
        self.has_q_controller = hasattr(self.optimizer, 'q_controller') and isinstance(getattr(self.optimizer, 'q_controller', None), HAKMEMQController)
        self.wandb_run=wandb.run if self.wandb_enabled and wandb is not None else None
        logger.info(f"Trainer(Nuc) Rank {rank}/{world_size}: Device={device}, AMP={self.use_amp}, Accum={self.grad_accum_steps}, MaxNorm={self.max_grad_norm}, Anomaly={self.detect_anomaly}, QCtrl={self.has_q_controller}, Vocab={nucleotide_vocab_size}")
        if self.is_main:
            os.makedirs(self.checkpoint_dir,exist_ok=True)

    def is_main_process(self) -> bool:
        return self.rank == 0

    def _get_hyperbolic_stats(self)->Dict[str,float]:
        stats={}
        m = self.model.module if isinstance(self.model, DDP) else self.model
        if hasattr(m,'wubu_model') and hasattr(m.wubu_model,'levels'):
            for i, lvl in enumerate(m.wubu_model.levels):
                try:
                    if hasattr(lvl,'get_curvature'):
                        stats[f"L{i}_Curv"]=lvl.get_curvature().item()
                    if hasattr(lvl,'get_scale'):
                        stats[f"L{i}_Scale"]=lvl.get_scale().item()
                    if hasattr(lvl,'get_spread') and lvl.use_spread:
                        stats[f"L{i}_Spread"]=lvl.get_spread().item()
                except Exception as e:
                    logger.warning(f"Error retrieving hyperbolic stats for Level {i}: {e}")
        return stats

    def _train_epoch(self):
        self.model.train()
        total_loss_cycle=0.
        optim_steps_epoch=0
        micro_steps_cycle=0
        micro_batches_processed = 0
        approx_optim_steps=None
        approx_micro_batches=None
        try:
            dset_len=0
            sampler=getattr(self.train_loader, 'sampler', None)
            dset=getattr(self.train_loader, 'dataset', None)
            if hasattr(sampler,'__len__'):
                dset_len=len(sampler)
            elif hasattr(dset,'__len__'):
                L = len(dset)
                dset_len = max(1, L // self.world_size) if self.world_size > 1 and not hasattr(sampler, '__len__') else L
            # --- Refactored Estimation Logic ---
            if dset_len > 0:
                bs = getattr(self.train_loader, 'batch_size', 1) or 1
                approx_micro_batches = dset_len
                if approx_micro_batches > 0 and self.grad_accum_steps > 0:
                    approx_optim_steps = approx_micro_batches // self.grad_accum_steps
                    logger.debug(f"Rank {self.rank} Epoch Estimate: {approx_micro_batches} micro-batches, {approx_optim_steps} optim steps.")
            # --- End Refactoring ---
            else:
                logger.info(f"Rank {self.rank}: Cannot accurately estimate epoch length (IterableDataset?).")
        except Exception as e:
            logger.warning(f"Could not estimate epoch length: {e}")

        disable_tqdm = not self.is_main
        batch_iter=tqdm(self.train_loader, desc=f"Ep {self.current_epoch+1}|Opt 0/?", disable=disable_tqdm, total=approx_micro_batches, unit="batch", dynamic_ncols=True, leave=False)
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch_data in enumerate(batch_iter):
            micro_batches_processed += 1
            micro_steps_cycle += 1
            is_last_micro_step = (micro_steps_cycle % self.grad_accum_steps == 0)
            is_last_batch_epoch = (approx_micro_batches is not None and i == (approx_micro_batches - 1))
            should_optim_step = is_last_micro_step or is_last_batch_epoch
            sync_ctx=contextlib.nullcontext()
            if self.world_size > 1 and isinstance(self.model, DDP) and not should_optim_step:
                sync_ctx = self.model.no_sync()
            anomaly_ctx=torch.autograd.detect_anomaly(check_nan=True) if self.detect_anomaly else contextlib.nullcontext()
            loss=None
            cur_step_loss=0.
            try:
                with sync_ctx, anomaly_ctx:
                    if isinstance(batch_data,(list,tuple)) and len(batch_data)==2:
                        ctx,tgts=batch_data
                    else:
                        # --- Refactored Batch Skip Logic ---
                        logger.warning(f"Rank {self.rank}: Skipping unexpected batch type {type(batch_data)}")
                        if should_optim_step:
                            logger.warning(f"Rank {self.rank}: Skipping optim step G{self.global_step} due to bad batch data.")
                            self.optimizer.zero_grad(set_to_none=True) # Discard potentially partial grads
                            total_loss_cycle = 0.0
                            micro_steps_cycle = 0
                        continue # Continue to the next batch
                        # --- End Refactoring ---

                    ctx=ctx.to(self.device,non_blocking=True)
                    tgts=tgts.to(self.device,non_blocking=True)
                    if ctx.numel()==0 or tgts.numel() == 0:
                        logger.warning(f"Rank {self.rank}: Skipping empty tensor batch.")
                        if should_optim_step:
                            logger.warning(f"Rank {self.rank}: Skipping optim step G{self.global_step} due to empty last micro-batch.")
                            self.optimizer.zero_grad(set_to_none=True)
                            total_loss_cycle = 0.0
                            micro_steps_cycle = 0
                        continue # Continue to the next batch

                    in_pad_mask = (ctx == NUCLEOTIDE_MAP.get('-', 4)) if torch.any(ctx == NUCLEOTIDE_MAP.get('-', 4)) else None # Use pad index N
                    tgt_pad_mask = (tgts == NUCLEOTIDE_MAP.get('-', 4)) if torch.any(tgts == NUCLEOTIDE_MAP.get('-', 4)) else None # Use pad index N

                    amp_dtype = torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
                    with amp.autocast(device_type=self.device.type,dtype=amp_dtype,enabled=self.use_amp):
                        logits=self.model(nucleotide_seq=ctx, target_nucleotide_seq=ctx, input_padding_mask=in_pad_mask, target_padding_mask=in_pad_mask) # Use same mask for now
                        assert logits is not None, "Model forward returned None logits"
                        loss=WuBuNestingSequenceModel.compute_loss(logits.float(), tgts, mask=tgt_pad_mask, smoothing=0.1, vocab_size=self.nucleotide_vocab_size)
                    assert loss is not None and torch.isfinite(loss), f"Non-finite or None loss: {loss}"
                    loss_scaled = loss / self.grad_accum_steps
                self.scaler.scale(loss_scaled).backward()
                cur_step_loss = loss.item()
                if not np.isfinite(cur_step_loss):
                    logger.warning(f"Rank {self.rank}: Non-finite loss ({cur_step_loss}) at GStep {self.global_step}, MicroStep {micro_steps_cycle}. Not accumulating.")
                    cur_step_loss = 0.0
            except Exception as batch_ex:
                # --- Refactored Exception Handling ---
                logger.error(f"MicroStep Error G{self.global_step} M{micro_steps_cycle} R{self.rank}: {batch_ex}",exc_info=False)
                total_loss_cycle = 0.
                micro_steps_cycle = 0
                should_optim_step = False # Prevent optim step on error
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                    logger.warning(f"Rank {self.rank}: Zeroed gradients after micro-step error.")
                except Exception as zge:
                    logger.error(f"Error zeroing gradients after micro-step error: {zge}")
                if 'CUDA out of memory' in str(batch_ex) and torch.cuda.is_available():
                    logger.warning("Clearing CUDA cache after OOM.")
                    torch.cuda.empty_cache()
                continue
                # --- End Refactoring ---
            total_loss_cycle += cur_step_loss

            if should_optim_step:
                avg_loss_cycle = total_loss_cycle / micro_steps_cycle if micro_steps_cycle > 0 else 0.
                optim_skipped = False
                unclipped_norm = 0.
                is_clipped = False
                clip_ratio = None
                grad_norm_error = False
                try:
                    self.scaler.unscale_(self.optimizer)
                except Exception as unscale_error:
                    logger.error(f"AMP Unscale Error G{self.global_step} R{self.rank}: {unscale_error}. Skipping optim step.")
                    optim_skipped = True

                if not optim_skipped and self.max_grad_norm > 0:
                    params_with_grad = [p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
                    if params_with_grad:
                        try:
                            total_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=float('inf'), norm_type=2.0)
                            unclipped_norm = total_norm.item()
                            if not np.isfinite(unclipped_norm):
                                logger.warning(f"Rank {self.rank}: Non-finite grad norm ({unclipped_norm}) BEFORE clip GStep {self.global_step}. Skipping optim step.")
                                optim_skipped = True
                                grad_norm_error = True
                            elif unclipped_norm > self.max_grad_norm:
                                is_clipped = True
                                clip_ratio = self.max_grad_norm / (unclipped_norm + EPS)
                                torch.nn.utils.clip_grad_norm_(params_with_grad, self.max_grad_norm, norm_type=2.0)
                        except Exception as norm_error:
                            logger.error(f"Grad Norm/Clip Error G{self.global_step} R{self.rank}: {norm_error}. Skipping optim step.")
                            optim_skipped = True
                            grad_norm_error = True
                            unclipped_norm = float('inf')
                    else:
                        unclipped_norm = 0.0
                elif not optim_skipped: # Calculate norm even if not clipping, for logging
                     params_with_grad = [p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
                     if params_with_grad:
                         try:
                             total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach().float(), 2.0) for p in params_with_grad]), 2.0)
                             unclipped_norm = total_norm.item()
                         except Exception as norm_calc_err:
                             logger.error(f"Error calculating grad norm (no clip): {norm_calc_err}")
                             unclipped_norm = float('inf')
                             grad_norm_error = True
                         if not np.isfinite(unclipped_norm):
                             grad_norm_error = True
                     else:
                         unclipped_norm = 0.0 # No grads, norm is 0


                if self.has_grad_stats:
                    if grad_norm_error:
                        self.optimizer.gradient_stats.non_finite_grads_in_step += 1
                    self.optimizer.gradient_stats.record_gradient(unclipped_norm, is_clipped, clip_ratio)

                if not optim_skipped and self.has_q_controller:
                    try:
                        q_ctrl = self.optimizer.q_controller
                        self.optimizer.set_current_loss(avg_loss_cycle)
                        grp0 = self.optimizer.param_groups[0]
                        cur_lr = grp0['lr']
                        cur_mom = grp0.get('momentum', 0.)
                        q_grad_norm = unclipped_norm if np.isfinite(unclipped_norm) else 100.0
                        q_state = q_ctrl.get_state(lr=cur_lr, momentum=cur_mom, grad_norm=q_grad_norm, loss=avg_loss_cycle)
                        if q_ctrl.prev_state is not None and q_ctrl.prev_action is not None and q_state is not None and q_ctrl.prev_loss is not None:
                           reward = q_ctrl.compute_reward(avg_loss_cycle, q_ctrl.prev_loss, q_grad_norm)
                           if np.isfinite(reward):
                               q_ctrl.update(q_ctrl.prev_state, q_ctrl.prev_action, reward, q_state)
                           else:
                               logger.warning(f"QCtrl non-finite reward ({reward}). Skip Q update.")
                        q_ctrl.prev_loss = avg_loss_cycle if np.isfinite(avg_loss_cycle) else q_ctrl.prev_loss
                        q_ctrl.prev_state = q_state
                        q_ctrl.prev_action = q_ctrl.choose_action(q_state) if q_state is not None else None
                    except Exception as q_controller_error:
                        # --- Refactored Exception Handling ---
                        logger.error(f"Q-Controller Update Error G{self.global_step} R{self.rank}: {q_controller_error}",exc_info=False)
                        if self.has_q_controller:
                            q_ctrl = self.optimizer.q_controller
                            q_ctrl.prev_state = None
                            q_ctrl.prev_action = None
                            q_ctrl.prev_loss = None
                        # --- End Refactoring ---

                if not optim_skipped:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                grad_stats = {}
                if self.has_grad_stats:
                    grad_stats=self.optimizer.gradient_stats.record_step(self.global_step, skipped=optim_skipped)

                if not optim_skipped:
                    optim_steps_epoch += 1
                    self.global_step += 1
                    if self.is_main:
                        opt_step_str=f"{optim_steps_epoch}/{(approx_optim_steps or '?')}"
                        batch_iter.set_description(f"Ep {self.current_epoch+1}|Opt {opt_step_str}")
                        cur_lr = self.optimizer.param_groups[0]['lr']
                        applied_norm = min(unclipped_norm, self.max_grad_norm) if is_clipped else unclipped_norm
                        batch_iter.set_postfix(Loss=f"{avg_loss_cycle:.3f}", LR=f"{cur_lr:.3e}", Grad=f"{applied_norm:.2f}", Scale=f"{self.scaler.get_scale():.0f}", refresh=False)

                        if self.global_step % self.log_interval == 0:
                            cur_mom=self.optimizer.param_groups[0].get('momentum',0.)
                            q_info = self.optimizer.get_q_info() if self.has_q_controller else {}
                            hyp_stats = self._get_hyperbolic_stats()
                            log_data = {"Epoch": self.current_epoch + 1, "Step": self.global_step, "Train Loss (Cycle Avg)": avg_loss_cycle, "LR": cur_lr, "Momentum": cur_mom, "Grad Norm (Applied)": applied_norm, "Grad Norm (Unclipped Max)": grad_stats.get('max_gradient', unclipped_norm if np.isfinite(unclipped_norm) else -1.0), "Clip %": grad_stats.get('clip_percentage', 0.), "NonFinite Grads": grad_stats.get('non_finite_grads', 0), "Optim Skipped": grad_stats.get('step_skipped', False), "AMP Scale": self.scaler.get_scale()}
                            log_data.update({f"Hyp/{k}":v for k,v in hyp_stats.items()})
                            log_data.update({f"QCtrl/{k}":v for k,v in q_info.items() if k!='last_action'})
                            last_q_act=q_info.get('last_action')
                            if last_q_act:
                                log_data["QCtrl/LR_Scale"]=last_q_act.get('lr_scale',1.)
                                log_data["QCtrl/Mom_Scale"]=last_q_act.get('momentum_scale',1.)
                            log_parts=[f"S{self.global_step}",f"Ep{self.current_epoch+1} Opt{optim_steps_epoch}",f"Loss {log_data['Train Loss (Cycle Avg)']:.4f}",f"LR {log_data['LR']:.3e}",f"Grad {log_data['Grad Norm (Applied)']:.2f}",f"Scale {log_data['AMP Scale']:.0f}"]
                            if hyp_stats:
                                log_parts.append(f"Crv[0] {hyp_stats.get('L0_Curv',0):.3g}")
                            if self.has_q_controller:
                                log_parts.append(f"QScale(L/M) {log_data.get('QCtrl/LR_Scale',1.):.2f}/{log_data.get('QCtrl/Mom_Scale',1.):.2f}")
                            if grad_stats.get('clip_percentage',0.) > 1.:
                                log_parts.append(f"Clip% {log_data['Clip %']:.1f}")
                            if grad_stats.get('non_finite_grads',0) > 0:
                                log_parts.append(f"NFGrads {log_data['NonFinite Grads']}")
                            if grad_stats.get('step_skipped', False):
                                log_parts.append("SKIPPED")
                            logger.info(" | ".join(log_parts))

                            if self.wandb_enabled and self.wandb_run:
                                try:
                                    wandb_log_data = {f"train/{k.replace(' ', '_')}": v for k, v in log_data.items()}
                                    wandb_log_data.update({f"train_hyp/{k}": v for k, v in hyp_stats.items()})
                                    wandb_log_data.update({f"train_qctrl/{k.replace('/','_')}": v for k,v in log_data.items() if k.startswith('QCtrl/')})
                                    wandb.log(wandb_log_data, step=self.global_step)
                                except Exception as wbe:
                                    logger.error(f"Wandb train log failed: {wbe}")

                        if self.is_main and self.save_interval > 0 and self.global_step % self.save_interval == 0:
                            self._save_checkpoint(is_intermediate=True, metrics={'train_loss_cycle': avg_loss_cycle})

                total_loss_cycle = 0.
                micro_steps_cycle = 0

        if self.is_main and hasattr(batch_iter, 'close'):
            batch_iter.close()

        if self.world_size > 1:
            logger.debug(f"Rank {self.rank} end-of-train-epoch barrier.")
            torch.distributed.barrier()
            logger.debug(f"Rank {self.rank} passed barrier.")

    @torch.no_grad()
    def _validate(self)->Dict[str,float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        approx_val_batches=None
        try:
            val_dset_len=0
            sampler=getattr(self.val_loader, 'sampler', None)
            dset=getattr(self.val_loader, 'dataset', None)
            if hasattr(sampler,'__len__'):
                val_dset_len=len(sampler)
            elif hasattr(dset,'__len__'):
                L=len(dset)
                val_dset_len = max(1, L // self.world_size) if self.world_size > 1 and not hasattr(sampler, '__len__') else L
            if val_dset_len > 0:
                bs = getattr(self.val_loader, 'batch_size', 1) or 1
                approx_val_batches = val_dset_len
        except Exception as e:
            logger.warning(f"Could not estimate validation length: {e}")

        val_iter=tqdm(self.val_loader, desc=f"Validating Ep {self.current_epoch+1}", disable=not self.is_main, total=approx_val_batches, unit="batch", leave=False)
        batch_losses=[]
        for batch_data in val_iter:
            try:
                if isinstance(batch_data,(list,tuple)) and len(batch_data)==2:
                    ctx,tgts=batch_data
                else:
                    continue
                ctx=ctx.to(self.device,non_blocking=True)
                tgts=tgts.to(self.device,non_blocking=True)
                if ctx.numel()==0 or tgts.numel() == 0:
                    continue

                in_pad_mask = (ctx == NUCLEOTIDE_MAP.get('-', 4)) if torch.any(ctx == NUCLEOTIDE_MAP.get('-', 4)) else None
                tgt_pad_mask = (tgts == NUCLEOTIDE_MAP.get('-', 4)) if torch.any(tgts == NUCLEOTIDE_MAP.get('-', 4)) else None

                amp_dtype=torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
                with torch.no_grad(), amp.autocast(device_type=self.device.type,dtype=amp_dtype,enabled=self.use_amp):
                    m_eval = self.model.module if isinstance(self.model, DDP) else self.model
                    logits = m_eval(nucleotide_seq=ctx, target_nucleotide_seq=ctx, input_padding_mask=in_pad_mask, target_padding_mask=in_pad_mask)
                    loss = WuBuNestingSequenceModel.compute_loss(logits.float(), tgts, mask=tgt_pad_mask, smoothing=0., vocab_size=self.nucleotide_vocab_size)

                if loss is not None and torch.isfinite(loss):
                    batch_losses.append(loss.item())
                else:
                    logger.warning(f"Rank {self.rank}: Non-finite validation loss ({loss}).")
            except Exception as ve:
                logger.error(f"Rank {self.rank} Validation batch error: {ve}",exc_info=False)
                continue

        all_losses = []
        global_loss_sum = 0.0
        global_loss_count = 0.0
        if self.world_size > 1:
            local_loss_sum = torch.tensor(sum(batch_losses), dtype=torch.float64, device=self.device)
            local_loss_count = torch.tensor(len(batch_losses), dtype=torch.float64, device=self.device)
            try:
                torch.distributed.all_reduce(local_loss_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(local_loss_count, op=torch.distributed.ReduceOp.SUM)
                global_loss_sum = local_loss_sum.item()
                global_loss_count = local_loss_count.item()
                logger.debug(f"Rank {self.rank} gathered global loss sum {global_loss_sum}, count {global_loss_count}")
            except Exception as gather_error:
                logger.error(f"Rank {self.rank}: Validation loss aggregation failed: {gather_error}. Reporting local metrics only (if main).")
                global_loss_sum = sum(batch_losses)
                global_loss_count = len(batch_losses)
        else:
            global_loss_sum = sum(batch_losses)
            global_loss_count = len(batch_losses)

        metrics={}
        if self.is_main:
            avg_loss = global_loss_sum / global_loss_count if global_loss_count > 0 else float('inf')
            ppl = float('inf')
            if np.isfinite(avg_loss):
                try:
                    clamped_loss = min(avg_loss, 700)
                    ppl=math.exp(clamped_loss)
                except OverflowError:
                    logger.warning(f"PPL overflow (avg_loss={avg_loss}).")
                    ppl=float('inf')
                except Exception as ppl_err:
                    logger.warning(f"Error calculating perplexity: {ppl_err}")
                    ppl=float('inf')
            metrics={'val_loss':avg_loss,'val_perplexity':ppl}
            self.last_val_metrics=metrics
            logger.info(f"Validation Ep {self.current_epoch+1} | Avg Loss: {metrics['val_loss']:.4f} | Perplexity: {metrics['val_perplexity']:.2f}")

            if self.wandb_enabled and self.wandb_run:
                try:
                    wblog={**{f"val/{k}":v for k,v in metrics.items()},"epoch":self.current_epoch+1}
                    hyp_stats = self._get_hyperbolic_stats()
                    wblog.update({f"val_hyp/{k}":v for k,v in hyp_stats.items()})
                    wandb.log(wblog, step=self.global_step)
                except Exception as wbe:
                    logger.error(f"Wandb validation log failed: {wbe}")

        if hasattr(val_iter,'close'):
            val_iter.close()

        if self.world_size > 1:
            logger.debug(f"Rank {self.rank} end-of-validation barrier.")
            torch.distributed.barrier()
            logger.debug(f"Rank {self.rank} passed barrier.")
        return metrics

    def _save_checkpoint(self, is_intermediate: bool=False, metrics: Optional[Dict]=None):
        if not self.is_main or self.save_interval < 0:
            return
        state_indicator = f"step_{self.global_step}" if is_intermediate else f"epoch_{self.current_epoch+1}_final"
        metric_str = ""
        current_metrics = metrics if metrics is not None else self.last_val_metrics
        if current_metrics:
            vloss = current_metrics.get('val_loss')
            tloss = current_metrics.get('train_loss_cycle')
            metric_key, metric_val = ('vloss', vloss) if vloss is not None and np.isfinite(vloss) else ('tloss', tloss) if tloss is not None and np.isfinite(tloss) else (None, None)
            if metric_key and metric_val is not None:
                metric_format = f"{metric_val:.2e}" if abs(metric_val) < 1e-3 and metric_val != 0 else f"{metric_val:.3f}"
                metric_str = f"_{metric_key}{metric_format}"

        filename = f"checkpoint_{state_indicator}{metric_str}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        clean_state_dict = {k: v for k, v in model_to_save.state_dict().items() if not k.endswith('.manifold')}
        checkpoint_data = {'epoch': self.current_epoch,
                           'global_step': self.global_step,
                           'model_state_dict': clean_state_dict,
                           'optimizer_state_dict': self.optimizer.state_dict(),
                           'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
                           'metrics': current_metrics,
                           'amp_enabled': self.use_amp,
                           'args': getattr(self,'args',None),
                           'wubu_config': getattr(model_to_save,'wubu_config',{}),
                           'sequence_config': getattr(model_to_save,'sequence_config',{}),
                           'nucleotide_vocab_size': self.nucleotide_vocab_size}

        if self.has_q_controller and self.optimizer.q_controller:
            try:
                q_ctrl = self.optimizer.q_controller
                # Convert NumPy arrays in q_table to lists for saving if necessary
                q_table_serializable = {}
                for state, action_dict in q_ctrl.q_table.items():
                    # Convert tuple keys to string for JSON compatibility if needed by torch.save backend
                    state_key_str = str(state)
                    q_table_serializable[state_key_str] = {param: arr.tolist() if isinstance(arr, np.ndarray) else arr for param, arr in action_dict.items()}

                q_state_data={'q_table': q_table_serializable, # Save serializable version
                              'epsilon': q_ctrl.epsilon,
                              'access_count': {str(k):v for k,v in q_ctrl.q_table_access_count.items()}, # Stringify keys
                              'creation_time': {str(k):v for k,v in q_ctrl.q_table_creation_time.items()}, # Stringify keys
                              'loss_window': list(q_ctrl.loss_window),
                              'grad_norm_window': list(q_ctrl.grad_norm_window),
                              'performance_window': list(q_ctrl.performance_window),
                              'stable_steps': q_ctrl.stable_steps,
                              'oscillation_counter': q_ctrl.oscillation_counter,
                              'prev_loss': q_ctrl.prev_loss,
                              'prev_state': str(q_ctrl.prev_state) if q_ctrl.prev_state else None, # Stringify state tuple
                              'prev_action': q_ctrl.prev_action,
                              # Save action_ranges as lists
                              'action_ranges': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in q_ctrl.action_ranges.items()},
                              'num_actions': q_ctrl.num_actions}
                checkpoint_data['q_controller_state'] = q_state_data
            except Exception as q_save_err:
                logger.error(f"Error preparing Q-Controller state for saving: {q_save_err}")

        temp_filepath = filepath + ".tmp." + str(random.randint(1000,9999))
        try:
            torch.save(checkpoint_data, temp_filepath)
            os.replace(temp_filepath, filepath)
            logger.info(f"Checkpoint saved: {filepath}")
        except Exception as save_error:
            logger.error(f"Failed to save checkpoint {filepath}: {save_error}", exc_info=True)
            # --- Refactored Cleanup ---
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError as remove_err:
                    logger.error(f"Could not remove temp ckpt file {temp_filepath}: {remove_err}")
            # --- End Refactoring ---

    def load_checkpoint(self, filepath: str)->int:
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint file not found: {filepath}")
            return 0
        try:
            checkpoint_data = torch.load(filepath, map_location='cpu')
            logger.info(f"Loading checkpoint: {os.path.basename(filepath)}")
            model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
            incompatible = model_to_load.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
            if incompatible.missing_keys:
                logger.warning(f"Ckpt Load - Missing model keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logger.warning(f"Ckpt Load - Unexpected model keys: {incompatible.unexpected_keys}")
            model_to_load.to(self.device)
            logger.info("Model state loaded.")

            if 'optimizer_state_dict' in checkpoint_data:
                try:
                    self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    # Move optimizer state to device
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                try:
                                    state[k] = v.to(self.device)
                                except Exception as move_err:
                                    logger.warning(f"Could not move optim state tensor {k} to {self.device}: {move_err}")
                    logger.info("Optimizer state loaded.")
                except Exception as optim_load_err:
                    # --- Refactored Reset ---
                    logger.warning(f"Failed load optim state: {optim_load_err}. Resetting optim state.")
                    self.optimizer.state = defaultdict(dict)
                    # --- End Refactoring ---
            else:
                logger.warning("Optim state not found. Starting fresh.")

            saved_amp_enabled = checkpoint_data.get('amp_enabled', False)
            if self.use_amp:
                if 'scaler_state_dict' in checkpoint_data and checkpoint_data['scaler_state_dict'] is not None and saved_amp_enabled:
                    try:
                        self.scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
                        logger.info("AMP scaler state loaded.")
                    except Exception as scaler_load_err:
                        # --- Refactored Reset ---
                        logger.warning(f"Failed load AMP scaler state: {scaler_load_err}. Resetting.")
                        self.scaler = amp.GradScaler(enabled=self.use_amp)
                        # --- End Refactoring ---
                elif not saved_amp_enabled:
                    logger.warning("Ckpt saved w/o AMP, but AMP enabled. Fresh scaler.")
            elif saved_amp_enabled:
                logger.warning("Ckpt saved w/ AMP, but AMP disabled. Ignoring scaler.")

            start_epoch = checkpoint_data.get('epoch', -1) + 1
            self.global_step = checkpoint_data.get('global_step', 0)
            self.current_epoch = start_epoch - 1 if start_epoch > 0 else 0
            self.last_val_metrics = checkpoint_data.get('metrics')
            if self.last_val_metrics:
                logger.info(f"Restored last val metrics: {self.last_val_metrics}")

            loaded_vocab_size = checkpoint_data.get('nucleotide_vocab_size', None)
            if loaded_vocab_size is not None and loaded_vocab_size != self.nucleotide_vocab_size:
                logger.warning(f"Ckpt vocab size ({loaded_vocab_size}) differs from current ({self.nucleotide_vocab_size}).")

            if self.has_q_controller and self.optimizer.q_controller and 'q_controller_state' in checkpoint_data:
                q_state = checkpoint_data['q_controller_state']
                logger.info("Loading Q-Controller state...")
                try:
                    q_ctrl = self.optimizer.q_controller
                    # Load action ranges and convert back to NumPy arrays
                    loaded_action_ranges = q_state.get('action_ranges', {})
                    q_ctrl.action_ranges = {k: np.array(v, dtype=np.float32) for k, v in loaded_action_ranges.items()}

                    # Load q_table and convert lists back to NumPy arrays
                    loaded_q_table = q_state.get('q_table', {})
                    q_ctrl.q_table = {}
                    for state_str, action_dict in loaded_q_table.items():
                       # Convert the stringified state tuple elements back
                       try:
                           state_key = tuple(eval(state_str))
                       except Exception:
                           state_key = state_str # Fallback if eval fails
                       q_ctrl.q_table[state_key] = {param: np.array(arr, dtype=np.float32) for param, arr in action_dict.items()}

                    # Check if loaded action ranges match current - if not, Q-table is likely invalid
                    if q_ctrl.action_ranges != {k: np.array(v, dtype=np.float32) for k, v in loaded_action_ranges.items()}:
                         logger.warning("QCtrl action ranges differ. Q-table might be incompatible. Resetting Q-state.")
                         q_ctrl.q_table = {}
                         q_ctrl.q_table_access_count = defaultdict(int)
                         q_ctrl.q_table_creation_time = {}
                    # else: q_ctrl.q_table = q_table_deserialized # Use deserialized table - this line seems redundant now


                    q_ctrl.epsilon = q_state.get('epsilon', q_ctrl.epsilon)
                    # Convert keys in access_count and creation_time back to tuples if needed
                    q_ctrl.q_table_access_count = defaultdict(int)
                    for k_str, v in q_state.get('access_count', {}).items():
                        try:
                            state_key = tuple(eval(k_str))
                        except Exception:
                            state_key = k_str
                        q_ctrl.q_table_access_count[state_key] = v

                    q_ctrl.q_table_creation_time = {}
                    for k_str, v in q_state.get('creation_time', {}).items():
                        try:
                            state_key = tuple(eval(k_str))
                        except Exception:
                            state_key = k_str
                        q_ctrl.q_table_creation_time[state_key] = v

                    maxlen_loss = q_ctrl.loss_window.maxlen
                    q_ctrl.loss_window = deque(q_state.get('loss_window', []), maxlen=maxlen_loss)
                    maxlen_grad = q_ctrl.grad_norm_window.maxlen
                    q_ctrl.grad_norm_window = deque(q_state.get('grad_norm_window', []), maxlen=maxlen_grad)
                    maxlen_perf = q_ctrl.performance_window.maxlen
                    q_ctrl.performance_window = deque(q_state.get('performance_window', []), maxlen=maxlen_perf)
                    q_ctrl.stable_steps=q_state.get('stable_steps',0)
                    q_ctrl.oscillation_counter=q_state.get('oscillation_counter',0)
                    q_ctrl.prev_loss=q_state.get('prev_loss')

                    # Convert prev_state back to tuple if needed
                    prev_state_loaded = q_state.get('prev_state')
                    try:
                        q_ctrl.prev_state = tuple(eval(prev_state_loaded)) if isinstance(prev_state_loaded, str) else tuple(prev_state_loaded) if prev_state_loaded else None
                    except Exception:
                        q_ctrl.prev_state = prev_state_loaded if prev_state_loaded else None # Fallback

                    q_ctrl.prev_action=q_state.get('prev_action')
                    q_ctrl.num_actions = q_state.get('num_actions', q_ctrl.num_actions) # Load num_actions
                    logger.info("Q-Controller state loaded.")
                except Exception as q_load_err:
                    # --- Refactored Reset ---
                    logger.warning(f"Failed load Q-Controller state: {q_load_err}. Resetting.",exc_info=False)
                    if self.has_q_controller and self.optimizer.q_controller:
                        self.optimizer.q_controller.q_table = {}
                        self.optimizer.q_controller.q_table_access_count = defaultdict(int)
                        self.optimizer.q_controller.q_table_creation_time = {}
                        self.optimizer.q_controller.loss_window.clear()
                        self.optimizer.q_controller.grad_norm_window.clear()
                        self.optimizer.q_controller.performance_window.clear()
                        self.optimizer.q_controller.prev_loss = None
                        self.optimizer.q_controller.prev_state = None
                        self.optimizer.q_controller.prev_action = None
                    # --- End Refactoring ---
            elif self.has_q_controller:
                logger.warning("QCtrl enabled, but no state found in ckpt. Starting fresh.")

            logger.info(f"Ckpt '{os.path.basename(filepath)}' loaded. Resuming Epoch {start_epoch} (GStep {self.global_step}).")
            return start_epoch
        except Exception as load_error:
            # --- Refactored Reset ---
            logger.error(f"Failed load ckpt '{filepath}': {load_error}", exc_info=True)
            self.global_step = 0
            self.current_epoch = 0
            self.optimizer.state = defaultdict(dict)
            if self.use_amp:
                self.scaler = amp.GradScaler(enabled=self.use_amp)
            if self.has_q_controller and self.optimizer.q_controller:
                 self.optimizer.q_controller.q_table = {}
                 self.optimizer.q_controller.q_table_access_count = defaultdict(int)
                 self.optimizer.q_controller.q_table_creation_time = {}
                 self.optimizer.q_controller.prev_loss = None
                 self.optimizer.q_controller.prev_state = None
                 self.optimizer.q_controller.prev_action = None
            return 0
            # --- End Refactoring ---

    def train(self, epochs: int, start_epoch: int=0):
        self.current_epoch = start_epoch
        if self.is_main:
            logger.info(f"=== Starting Training ===")
            logger.info(f" - Epochs: {epochs}")
            logger.info(f" - Start Epoch: {start_epoch + 1}")
            logger.info(f" - Initial Global Step: {self.global_step}")
            logger.info(f" - Device: {self.device}")
            logger.info(f" - World Size: {self.world_size}")
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            epoch_start_time = time.monotonic()
            if self.is_main:
                logger.info(f"--- Starting Epoch {epoch+1}/{epochs} ---")
            if isinstance(getattr(self.train_loader, 'sampler', None), DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            if self.val_loader and isinstance(getattr(self.val_loader, 'sampler', None), DistributedSampler):
                self.val_loader.sampler.set_epoch(epoch)
            if hasattr(self.train_loader.dataset,'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
            if self.val_loader and hasattr(self.val_loader.dataset,'set_epoch'):
                self.val_loader.dataset.set_epoch(epoch)

            self._train_epoch()
            train_duration = time.monotonic() - epoch_start_time
            if self.is_main:
                logger.info(f"Epoch {epoch+1} Training finished in {timedelta(seconds=train_duration)}.")

            val_start_time = time.monotonic()
            val_metrics = self._validate()
            val_duration = time.monotonic() - val_start_time
            if self.is_main and val_metrics:
                logger.info(f"Epoch {epoch+1} Validation finished in {timedelta(seconds=val_duration)}.")

            if self.is_main and self.save_interval >= 0:
                self._save_checkpoint(is_intermediate=False, metrics=val_metrics if val_metrics else None)

            if self.world_size > 1:
                logger.debug(f"Rank {self.rank} end-of-full-epoch barrier.")
                torch.distributed.barrier()
                logger.debug(f"Rank {self.rank} passed barrier.")

        if self.is_main:
            logger.info(f"=== Training finished after {epochs} epochs ===")

# =====================================================================
# Default Configuration
# =====================================================================
DEFAULT_CONFIG_WUBU = {
    "num_levels": 3,
    "hyperbolic_dims": [128, 96, 64],
    "initial_curvatures": [1.0, 0.7, 0.4],
    "initial_scales": [1.0, 1.2, 1.5],
    "initial_spread_values": [0.1, 0.2, 0.3],
    "boundary_points_per_level": [16, 12, 8],
    "learnable_curvature": True,
    "learnable_scales": True,
    "learnable_spread": True,
    "curvature_min_value": 0.01,
    "scale_min_value": 0.1,
    "spread_min_value": 0.01,
    "use_level_descriptors": True,
    "use_level_spread": True,
    "level_descriptor_init_scale": 1e-5,
    "transform_types": ["mlp", "mlp"], # L0->L1, L1->L2
    "transform_hidden_dims": [96, 64], # Hidden dims for MLPs
    "relative_vector_aggregation": "mean", # 'mean', 'sum', 'none'
    "aggregation_method": "concat_tangent", # Only 'concat_tangent' supported
    "dropout": 0.1,
    "use_tangent_flow": True,
    "tangent_flow_type": "mlp", # 'mlp', 'linear', 'none'
    "tangent_flow_hidden_dim_ratio": 0.5,
    "tangent_flow_scale": 0.1
}

DEFAULT_CONFIG_SEQUENCE = {
    "local_hidden_size": 256,
    "decoder_memory_dim": sum(DEFAULT_CONFIG_WUBU["hyperbolic_dims"]), # Overridden later
    "context_window": 256,
    "num_encoder_layers": 4,
    "num_encoder_heads": 8,
    "num_decoder_layers": 6,
    "num_decoder_heads": 8,
    "encoder_max_seq_len": 1024,
    "decoder_max_seq_len": 2048,
    "nucleotide_vocab_size": NUCLEOTIDE_VOCAB_SIZE,
}

DEFAULT_CONFIG_QLEARN = {
    "learning_rate": 0.015,
    "discount": 0.90,
    "epsilon": 0.15,
    "epsilon_decay": 0.99985,
    "min_epsilon": 0.005,
    "lr_scale_options": [0.95, 0.98, 1.0, 1.02, 1.05],
    "momentum_scale_options": [0.98, 0.99, 1.0, 1.005, 1.01],
    "max_q_table_size": 15000
}

# =====================================================================
# Argument Parsing
# =====================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBu Nesting Model Trainer (Nucleotide Adaptation)")
    # --- Dataset and Paths ---
    parser.add_argument("--dataset_name", type=str, default="rnacentral_homo_sapiens_ncrna", help="Name of dataset if not using combined.")
    parser.add_argument("--use_combined_dataset", action="store_true", help="Use a combined dataset from multiple sources.")
    parser.add_argument("--max_combined_datasets", type=int, default=None, help="Limit number of datasets in combined dataset.")
    parser.add_argument("--balanced_sampling", action=argparse.BooleanOptionalAction, default=True, help="When using combined dataset, sample evenly.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Directory for data.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_nucleotide", help="Directory for checkpoints.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to resume.")
    parser.add_argument("--numpy_allow_pickle", action="store_true", help="Allow NumPy to use pickle for loading arrays")
    # --- Model Configuration ---
    parser.add_argument("--context_size", type=int, default=DEFAULT_CONFIG_SEQUENCE["context_window"], help="Sequence length for model input.")
    parser.add_argument("--local_hidden_size", type=int, default=DEFAULT_CONFIG_SEQUENCE["local_hidden_size"], help="Hidden size for encoder/decoder.")
    parser.add_argument("--num_encoder_layers", type=int, default=DEFAULT_CONFIG_SEQUENCE["num_encoder_layers"], help="Encoder layers.")
    parser.add_argument("--num_decoder_layers", type=int, default=DEFAULT_CONFIG_SEQUENCE["num_decoder_layers"], help="Decoder layers.")
    parser.add_argument("--num_encoder_heads", type=int, default=DEFAULT_CONFIG_SEQUENCE["num_encoder_heads"], help="Encoder heads.")
    parser.add_argument("--num_decoder_heads", type=int, default=DEFAULT_CONFIG_SEQUENCE["num_decoder_heads"], help="Decoder heads.")
    parser.add_argument("--encoder_max_len", type=int, default=DEFAULT_CONFIG_SEQUENCE["encoder_max_seq_len"], help="Max pos encoding length encoder.")
    parser.add_argument("--decoder_max_len", type=int, default=DEFAULT_CONFIG_SEQUENCE["decoder_max_seq_len"], help="Max pos encoding length decoder.")
    # --- WuBu Core Configuration ---
    parser.add_argument("--wubu_levels", type=int, default=DEFAULT_CONFIG_WUBU["num_levels"], help="WuBu nesting levels.")
    parser.add_argument("--wubu_dims", nargs='+', type=int, default=DEFAULT_CONFIG_WUBU["hyperbolic_dims"], help="Hidden dims per WuBu level.")
    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU (before creative batching).")
    parser.add_argument("--target_effective_batch_size", type=int, default=None, help="Target effective batch size (used by creative batching).")
    parser.add_argument("--creative_batching", action="store_true", help="Enable dynamic batch_size/accum adjustment for VRAM.")
    parser.add_argument("--creative_batching_vram_gb", type=float, default=96.0, help="VRAM GB for creative batching calculation.")
    parser.add_argument("--creative_batching_safety_factor", type=float, default=1.5, help="Safety factor for VRAM usage estimate.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Optimizer momentum.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Optimizer weight decay.")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps (if creative batching off).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True, help="Use Automatic Mixed Precision (AMP).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--disable_q_learning", action="store_true", help="Disable Q-Learning optimizer control.")
    # --- Dataloader and Hardware ---
    default_num_workers = 0
    cpu_count = os.cpu_count()
    if cpu_count:
        default_num_workers = min(cpu_count // 2 if cpu_count > 1 else 1, 8)
    parser.add_argument("--num_workers", type=int, default=default_num_workers, help="Dataloader workers per GPU.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Dataloader prefetch factor.")
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True, help="Use pinned memory for dataloaders.")
    parser.add_argument("--distributed_backend", type=str, default="nccl", choices=["nccl", "gloo"], help="Distributed backend.")
    # --- Logging and Saving ---
    parser.add_argument("--log_interval", type=int, default=50, help="Log training status every N optim steps.")
    parser.add_argument("--save_interval", type=int, default=5000, help="Save checkpoint every N optim steps (-1 to disable).")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    parser.add_argument("--wandb_project", type=str, default="WuBuNestingNucleotide", help="WandB project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity (optional).")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable WandB logging.")
    parser.add_argument("--detect_anomaly", action="store_true", help="Enable torch.autograd.detect_anomaly.")
    args = parser.parse_args()

    # --- Post-processing and Validation ---
    if len(args.wubu_dims) != args.wubu_levels:
        logger.warning(f"Mismatch: --wubu_levels={args.wubu_levels} vs --wubu_dims ({len(args.wubu_dims)}). Adjusting...")
        default_dims = DEFAULT_CONFIG_WUBU["hyperbolic_dims"]
        adjusted_dims = args.wubu_dims[:args.wubu_levels]
        while len(adjusted_dims) < args.wubu_levels:
            fallback_dim = default_dims[len(adjusted_dims)] if len(adjusted_dims) < len(default_dims) else (adjusted_dims[-1] if adjusted_dims else 64)
            adjusted_dims.append(fallback_dim)
        args.wubu_dims = adjusted_dims
        logger.warning(f"Using adjusted wubu_dims: {args.wubu_dims}")
    if args.local_hidden_size % args.num_encoder_heads != 0:
        raise ValueError(f"Encoder hidden size ({args.local_hidden_size}) must be divisible by encoder heads ({args.num_encoder_heads})")
    if args.local_hidden_size % args.num_decoder_heads != 0:
        raise ValueError(f"Decoder hidden size ({args.local_hidden_size}) must be divisible by decoder heads ({args.num_decoder_heads})")
    if args.num_workers < 0:
        args.num_workers = 0
    if args.num_workers == 0 and args.prefetch_factor is not None:
        args.prefetch_factor = None
    if not torch.cuda.is_available() and args.pin_memory:
        args.pin_memory = False
    args.nucleotide_vocab_size = NUCLEOTIDE_VOCAB_SIZE
    if args.use_combined_dataset and args.balanced_sampling and args.max_combined_datasets == 1:
        logger.warning("Balanced sampling enabled, but only max 1 dataset requested. No effect.")
    if args.creative_batching and args.target_effective_batch_size is None:
        args.target_effective_batch_size = args.batch_size * args.grad_accum_steps
        logger.info(f"Creative batching: Default target_effective_batch_size={args.target_effective_batch_size}")
    elif args.creative_batching and args.target_effective_batch_size is not None:
        logger.info(f"Creative batching enabled with target_effective_batch_size={args.target_effective_batch_size}. Base batch/accum may be overridden.")
    return args

# =====================================================================
# Distributed Setup Utilities
# =====================================================================
def setup_distributed(backend='nccl'):
    if is_initialized():
        logger.warning("DDP already initialized.")
        rank = get_rank()
        world_size = get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count() if torch.cuda.is_available() else 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        return device, rank, local_rank, world_size
    try:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '12355')
        if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            if backend.lower() != 'nccl':
                logger.warning(f"CUDA available, but backend is '{backend}'. NCCL recommended.")
        else:
            device = torch.device("cpu")
            if backend.lower() == 'nccl':
                logger.warning("CUDA unavailable/insufficient. NCCL needs CUDA. Switching to 'gloo'.")
                backend = 'gloo'
            else:
                logger.info(f"Using CPU with '{backend}' backend.")
        print(f"Rank {rank} Initializing DDP: Backend={backend}, Addr={master_addr}:{master_port}, WorldSize={world_size}")
        init_process_group(backend=backend, init_method=f'tcp://{master_addr}:{master_port}', world_size=world_size, rank=rank, timeout=timedelta(seconds=1800))
        logger.info(f"DDP Rank {rank}/{world_size} initialized. Backend: {backend}. Device: {device}.")
        logger.debug(f"Rank {rank} entering barrier after init.")
        torch.distributed.barrier()
        logger.debug(f"Rank {rank} passed barrier.")
        return device, rank, local_rank, world_size
    except KeyError as e:
        logger.error(f"DDP env var missing: {e}. Use torchrun.")
        raise RuntimeError(f"DDP env var missing: {e}") from e
    except Exception as e:
        logger.error(f"DDP init failed: {e}", exc_info=True)
        raise RuntimeError("DDP init failed") from e

def is_main_process() -> bool:
    return not is_initialized() or get_rank() == 0

def cleanup_distributed():
    if is_initialized():
        rank = get_rank()
        logger.debug(f"Rank {rank} cleaning up DDP group.")
        destroy_process_group()
        logger.debug(f"Rank {rank} DDP cleanup finished.")

# =====================================================================
# Main Execution Logic
# =====================================================================
def run():
    args = parse_arguments()
    is_distributed = 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1
    device, rank, local_rank, world_size = torch.device("cpu"), 0, 0, 1
    if is_distributed:
        try:
            device, rank, local_rank, world_size = setup_distributed(backend=args.distributed_backend)
        except Exception as e:
            print(f"FATAL: DDP setup failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        logger.info("CUDA not available, running on CPU.")
        if args.distributed_backend.lower() == 'nccl':
            args.distributed_backend = 'gloo'

    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = f'%(asctime)s - R{rank} - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level_int, format=log_format, force=True)
    logger.setLevel(log_level_int)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    if rank == 0:
         logger.info("=====================================================================")
         logger.info(" WuBu Nesting Model Trainer (Nucleotide Adaptation)")
         logger.info("=====================================================================")
         logger.info(f"Run Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
         logger.info(f"Platform: {platform.system()} ({platform.release()})")
         logger.info(f"Hostname: {socket.gethostname()}")
         logger.info(f"Python Version: {sys.version.split()[0]}")
         logger.info(f"PyTorch Version: {torch.__version__}")
         logger.info(f"BioPython Available: {BIOPYTHON_AVAILABLE}")
         logger.info(f"WandB Available: {WANDB_AVAILABLE}")
         logger.info(f"Distributed Run: {is_distributed} (World Size: {world_size})")
         logger.info(f"Device: {device}")
         if torch.cuda.is_available():
             logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(local_rank if is_distributed else 0)}")
         logger.info(f"Using Combined Dataset: {args.use_combined_dataset}")
         if args.use_combined_dataset:
             logger.info(f" - Balanced Sampling: {args.balanced_sampling}")
         logger.info(f"Creative Batching: {args.creative_batching} (VRAM Target: {args.creative_batching_vram_gb} GB, Safety: {args.creative_batching_safety_factor})")
         logger.info(f"Arguments: {vars(args)}")
         logger.info("=====================================================================")

    seed_offset = args.seed + rank
    torch.manual_seed(seed_offset)
    np.random.seed(seed_offset)
    random.seed(seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_offset)
        logger.debug(f"Rank {rank}: CUDA seeded with {seed_offset}")
    else:
        logger.debug(f"Rank {rank}: Torch/NumPy/Random seeded with {seed_offset} (CPU)")

    processed_npy_path = None
    dataset_class = None
    dataset_args = {}
    if rank == 0:
        if args.use_combined_dataset:
            logger.info(f"Rank 0: Prep COMBINED dataset (Max: {args.max_combined_datasets})...")
            processed_npy_path = prepare_combined_dataset(args.data_dir, max_datasets=args.max_combined_datasets)
        else:
            logger.info(f"Rank 0: Prep SINGLE dataset '{args.dataset_name}'...")
            processed_npy_path = prepare_dataset(args.dataset_name, args.data_dir)

        if processed_npy_path is None:
            logger.error(f"Rank 0: Dataset prep failed. Exiting.")
            if world_size > 1:
                cleanup_distributed()
            sys.exit(1)
        else:
            logger.info(f"Rank 0: Dataset ready at '{processed_npy_path}'")

    if world_size > 1:
        logger.debug(f"Rank {rank} waiting at dataset path barrier...")
        path_list = [processed_npy_path] if rank == 0 else [None]
        torch.distributed.broadcast_object_list(path_list, src=0)
        processed_npy_path = path_list[0]
        torch.distributed.barrier()
        logger.debug(f"Rank {rank} passed path barrier. Path: {processed_npy_path}")
        if processed_npy_path is None:
            logger.error(f"Rank {rank}: No valid dataset path from Rank 0. Exiting.")
            cleanup_distributed()
            sys.exit(1)
    elif processed_npy_path is None:
        logger.error("Dataset prep failed. Exiting.")
        sys.exit(1)

    if args.use_combined_dataset:
        dataset_class = BalancedWuBuNestingDataset
        dataset_args = {"npy_file_path": processed_npy_path, "context_size": args.context_size, "balanced_sampling": args.balanced_sampling, "metadata_path": os.path.join(args.data_dir, COMBINED_DATA_INFO_FILE)}
    else:
        dataset_class = WuBuNestingDataset
        dataset_args = {"npy_file_path": processed_npy_path, "context_size": args.context_size}

    effective_batch_size = args.batch_size * args.grad_accum_steps
    adjusted_batch_size = args.batch_size
    adjusted_accum_steps = args.grad_accum_steps
    if args.creative_batching and torch.cuda.is_available():
        try:
            vram_gb = args.creative_batching_vram_gb
            safety_factor = max(1.1, args.creative_batching_safety_factor)
            fixed_mem_gb_estimate = vram_gb * 0.08
            logger.debug(f"CB Est fixed mem: {fixed_mem_gb_estimate:.2f} GB")
            bytes_per_element = 2 if args.use_amp else 4
            overhead_factor = 20
            max_len = max(args.encoder_max_len, args.decoder_max_len, args.context_size)
            sample_mem_mb = (max_len * args.local_hidden_size * overhead_factor * bytes_per_element) / (1024 * 1024)
            logger.debug(f"CB Est mem/sample: {sample_mem_mb:.2f} MB")
            if sample_mem_mb > 0:
                available_gb_for_batches = (vram_gb - fixed_mem_gb_estimate) * 0.75
                available_mb_for_batches = available_gb_for_batches * 1024
                logger.debug(f"CB Avail mem/batch: {available_mb_for_batches:.2f} MB")
                optimal_gpu_batch_size = int(available_mb_for_batches / (sample_mem_mb * safety_factor))
                optimal_gpu_batch_size = max(8, min(1024, optimal_gpu_batch_size))
                logger.info(f"CB Optimal GPU batch size calc: {optimal_gpu_batch_size}")
                target_eff_batch = args.target_effective_batch_size if args.target_effective_batch_size is not None else effective_batch_size
                logger.info(f"CB Target effective batch: {target_eff_batch}")
                adjusted_accum_steps = max(1, round(target_eff_batch / (optimal_gpu_batch_size * world_size)))
                adjusted_batch_size = max(1, target_eff_batch // (adjusted_accum_steps * world_size))
                if adjusted_batch_size != args.batch_size or adjusted_accum_steps != args.grad_accum_steps:
                    logger.warning(f"CB Applied: Adjusted BS per GPU: {args.batch_size} -> {adjusted_batch_size}")
                    logger.warning(f"CB Applied: Adjusted Accum Steps: {args.grad_accum_steps} -> {adjusted_accum_steps}")
                    logger.warning(f" -> New Effective BS: {adjusted_batch_size * adjusted_accum_steps * world_size}")
                else:
                    logger.info("CB: No adjustment needed.")
            else:
                logger.warning("CB: Could not estimate mem/sample. Using original settings.")
                adjusted_batch_size = args.batch_size
                adjusted_accum_steps = args.grad_accum_steps
        except Exception as cb_err:
            logger.error(f"CB calc failed: {cb_err}. Using original settings.", exc_info=True)
            adjusted_batch_size = args.batch_size
            adjusted_accum_steps = args.grad_accum_steps
    else:
        adjusted_batch_size = args.batch_size
        adjusted_accum_steps = args.grad_accum_steps
        if rank == 0:
            logger.info("Creative batching disabled or not applicable.")

    try:
        train_dataset = dataset_class(**dataset_args)
        val_dataset = dataset_class(**dataset_args)
        train_dataset.set_seed(args.seed + rank)
        val_dataset.set_seed(args.seed + world_size + rank)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed + rank, drop_last=True) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_distributed else None
        rank_worker_offset = rank * args.num_workers
        g = torch.Generator()
        g.manual_seed(args.seed + rank)
        worker_init_fn = functools.partial(seed_worker, base_seed=args.seed, rank_offset=rank_worker_offset)
        train_loader = DataLoader(train_dataset, batch_size=adjusted_batch_size, sampler=train_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None, worker_init_fn=worker_init_fn if args.num_workers > 0 else None, generator=g if args.num_workers > 0 else None, persistent_workers=args.num_workers > 0, drop_last=is_distributed)
        val_loader = DataLoader(val_dataset, batch_size=adjusted_batch_size, sampler=val_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None, worker_init_fn=worker_init_fn if args.num_workers > 0 else None, generator=g if args.num_workers > 0 else None, persistent_workers=args.num_workers > 0, drop_last=False)
        train_len_est = len(train_loader) if hasattr(train_loader, '__len__') else 'N/A'
        val_len_est = len(val_loader) if hasattr(val_loader, '__len__') else 'N/A'
        logger.info(f"Rank {rank}: Dataloaders created (Train Est Steps: {train_len_est}, Val Est Steps: {val_len_est}, GPU Batch: {adjusted_batch_size})")
    except Exception as e:
        logger.error(f"Rank {rank}: Failed create datasets/loaders: {e}", exc_info=True)
        if is_distributed:
            cleanup_distributed()
        sys.exit(1)

    wubu_config = DEFAULT_CONFIG_WUBU.copy()
    wubu_config["num_levels"] = args.wubu_levels
    wubu_config["hyperbolic_dims"] = args.wubu_dims
    num_levels = wubu_config["num_levels"]
    num_transforms = max(0, num_levels - 1)

    def _resize_config_list(cd, key, tlen, dlist):
        clist = cd.get(key, [])
        if len(clist) == tlen:
            return
        logger.debug(f"Adjusting WuBu cfg '{key}' to len {tlen}.")
        rlist = dlist[:tlen]
        while len(rlist) < tlen:
            fbval = rlist[-1] if rlist else (dlist[0] if dlist else None)
            rlist.append(fbval)
        cd[key] = rlist

    _resize_config_list(wubu_config, "initial_curvatures", num_levels, DEFAULT_CONFIG_WUBU["initial_curvatures"])
    _resize_config_list(wubu_config, "initial_scales", num_levels, DEFAULT_CONFIG_WUBU["initial_scales"])
    _resize_config_list(wubu_config, "initial_spread_values", num_levels, DEFAULT_CONFIG_WUBU["initial_spread_values"])
    _resize_config_list(wubu_config, "boundary_points_per_level", num_levels, DEFAULT_CONFIG_WUBU["boundary_points_per_level"])
    _resize_config_list(wubu_config, "transform_types", num_transforms, DEFAULT_CONFIG_WUBU["transform_types"])
    _resize_config_list(wubu_config, "transform_hidden_dims", num_transforms, DEFAULT_CONFIG_WUBU["transform_hidden_dims"])

    sequence_config = DEFAULT_CONFIG_SEQUENCE.copy()
    sequence_config["local_hidden_size"] = args.local_hidden_size
    sequence_config["decoder_memory_dim"] = sum(wubu_config["hyperbolic_dims"])
    sequence_config["context_window"] = args.context_size
    sequence_config["num_encoder_layers"] = args.num_encoder_layers
    sequence_config["num_decoder_layers"] = args.num_decoder_layers
    sequence_config["num_encoder_heads"] = args.num_encoder_heads
    sequence_config["num_decoder_heads"] = args.num_decoder_heads
    sequence_config["encoder_max_seq_len"] = args.encoder_max_len
    sequence_config["decoder_max_seq_len"] = args.decoder_max_len
    sequence_config["nucleotide_vocab_size"] = args.nucleotide_vocab_size

    try:
        model = WuBuNestingSequenceModel(wubu_config=wubu_config, sequence_config=sequence_config)
        model.to(device)
    except Exception as e:
        logger.error(f"Rank {rank}: Failed model init: {e}", exc_info=True)
        if is_distributed:
            cleanup_distributed()
        sys.exit(1)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {type(model).__name__}, Total Params: {num_params:,}, Trainable: {num_trainable:,}")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None, output_device=local_rank if device.type == 'cuda' else None, find_unused_parameters=False, broadcast_buffers=True)
        logger.info(f"Rank {rank}: Model wrapped with DDP.")

    q_config = DEFAULT_CONFIG_QLEARN if not args.disable_q_learning else None
    try:
        optimizer = RiemannianEnhancedSGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, q_learning_config=q_config)
        logger.info(f"Rank {rank}: Optimizer: {type(optimizer).__name__} (QCtrl: {q_config is not None})")
    except Exception as e:
        logger.error(f"Rank {rank}: Failed optim init: {e}", exc_info=True)
        if is_distributed:
            cleanup_distributed()
        sys.exit(1)

    if rank == 0 and not args.disable_wandb and WANDB_AVAILABLE:
        try:
            run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            ds_flag = "Combined" if args.use_combined_dataset else args.dataset_name[:10]
            cb_flag = "_CB" if args.creative_batching else ""
            q_flag = "_NoQ" if args.disable_q_learning else ""
            eff_bs = adjusted_batch_size*adjusted_accum_steps*world_size
            wandb_run_name = f"wubu_nuc_{ds_flag}_L{args.wubu_levels}_H{args.local_hidden_size}_EBS{eff_bs}{cb_flag}{q_flag}_{run_ts}"
            run_config = vars(args).copy()
            run_config['adjusted_batch_size_per_gpu'] = adjusted_batch_size
            run_config['adjusted_grad_accum_steps'] = adjusted_accum_steps
            run_config['effective_batch_size'] = eff_bs
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=run_config, name=wandb_run_name, job_type="train", resume="allow")
            logger.info("WandB initialized.")
            # wandb.watch(model, log='gradients', log_freq=max(100, args.log_interval))
        except Exception as e:
            logger.error(f"WandB init failed: {e}. Disabling.", exc_info=True)
            args.disable_wandb = True
    elif rank == 0 and not args.disable_wandb and not WANDB_AVAILABLE:
        logger.warning("WandB requested but not found. Disabling.")
        args.disable_wandb = True

    try:
        trainer = Trainer(model=model, optimizer=optimizer, device=device, train_loader=train_loader, val_loader=val_loader, grad_accum_steps=adjusted_accum_steps, use_amp=args.use_amp, log_interval=args.log_interval, save_interval=args.save_interval, checkpoint_dir=args.checkpoint_dir, wandb_enabled=(rank == 0 and not args.disable_wandb), max_grad_norm=args.max_grad_norm, rank=rank, world_size=world_size, detect_anomaly=args.detect_anomaly, nucleotide_vocab_size=args.nucleotide_vocab_size)
        trainer.args = args
        logger.info(f"Rank {rank}: Trainer initialized.")
    except Exception as e:
        logger.error(f"Rank {rank}: Failed Trainer init: {e}", exc_info=True)
        if rank == 0 and not args.disable_wandb and wandb is not None and wandb.run:
            wandb.finish(exit_code=1)
        if is_distributed:
            cleanup_distributed()
        sys.exit(1)


    start_epoch = 0
    if args.load_checkpoint:
        try:
            if rank == 0:
                logger.info(f"Attempting load ckpt: {args.load_checkpoint}")
                start_epoch = trainer.load_checkpoint(args.load_checkpoint)
                logger.info(f"Rank {rank}: Ckpt loaded. Resume epoch {start_epoch+1}.")
            if world_size > 1:
                start_epoch_tensor = torch.tensor(start_epoch, dtype=torch.int, device=device)
                torch.distributed.broadcast(start_epoch_tensor, src=0)
                start_epoch = start_epoch_tensor.item()
                # Ensure non-rank-0 processes load state after rank 0 validates path etc.
                if rank > 0:
                     logger.info(f"Rank {rank}: Loading checkpoint {args.load_checkpoint} based on rank 0.")
                     _ = trainer.load_checkpoint(args.load_checkpoint) # Load on other ranks, ignore return value
                torch.distributed.barrier() # Ensure all ranks finished loading before proceeding
        except FileNotFoundError:
            logger.error(f"Ckpt file not found: {args.load_checkpoint}. Start fresh.")
        except Exception as e:
            logger.error(f"Failed load ckpt: {e}. Start fresh.", exc_info=True)
            start_epoch = 0
            trainer.optimizer.state = defaultdict(dict)
            trainer.global_step = 0
            trainer.current_epoch = 0
            if trainer.use_amp:
                trainer.scaler = amp.GradScaler(enabled=trainer.use_amp)
            if trainer.has_q_controller and trainer.optimizer.q_controller:
                trainer.optimizer.q_controller.q_table = {}
                trainer.optimizer.q_controller.q_table_access_count = defaultdict(int)
                trainer.optimizer.q_controller.q_table_creation_time = {}
                trainer.optimizer.q_controller.prev_loss = None
                trainer.optimizer.q_controller.prev_state = None
                trainer.optimizer.q_controller.prev_action = None

    try:
        if rank == 0:
            logger.info("Starting main training loop...")
        trainer.train(epochs=args.epochs, start_epoch=start_epoch)
        if rank == 0:
            logger.info("Training finished successfully.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted.")
        if rank == 0 and args.save_interval >= 0:
            logger.info("Saving interrupt ckpt...")
            trainer._save_checkpoint(is_intermediate=True, metrics=trainer.last_val_metrics)
    except Exception as e:
        logger.error(f"Unhandled training exception Rank {rank}: {e}", exc_info=True)
        if rank == 0 and not args.disable_wandb and WANDB_AVAILABLE and wandb.run:
            logger.error("Finishing WandB run with error.")
            wandb.finish(exit_code=1)
        if is_distributed:
            cleanup_distributed()
        sys.exit(1)


    if is_distributed:
        cleanup_distributed()
    if rank == 0 and not args.disable_wandb and WANDB_AVAILABLE and wandb.run:
        logger.info("Finishing WandB run normally.")
        wandb.finish()
    logger.info(f"Rank {rank}: Script execution finished.")

# =====================================================================
# Entry Point
# =====================================================================
if __name__ == "__main__":
    # Example launch commands:
    #   Combined Dataset, 4 GPUs, Creative Batching (Target Effective BS=2048):
    #   torchrun --standalone --nproc_per_node=4 wubu_nucleotide_trainer.py --use_combined_dataset --balanced_sampling --creative_batching --target_effective_batch_size 2048 --use_amp --wandb_project MyWuBuRuns --num_workers 4 --context_size 512
    #
    #   Single Dataset (Rfam Seed), Single GPU, Standard Batching:
    #   python wubu_nucleotide_trainer.py --dataset_name rfam_seed --batch_size 64 --grad_accum_steps 1 --context_size 256
    run()