# =====================================================================
# WuBu Nesting Model Trainer (Nucleotide Adaptation - Hybrid Spatial/Hyperbolic V3 - PyTorch Fallback)
# =====================================================================
# Description:
# This script trains a sequence model based on the WuBu Nesting
# architecture, adapted for nucleotide sequences (e.g., mRNA).
# It uses a hyperbolic geometry core with tangent space bridges.
# CRITICAL CHANGE: Encoder/Decoder components use hybrid spatial
# layers with 3D coordinate information derived from predicted
# RNA secondary structures. Triton kernels are REPLACED with PyTorch.
# Data is stored and read using HDF5 format.
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
# --- Keep original np.load for potential compatibility, but primary data uses HDF5 ---
orig_np_load = np.load
# Define a custom load function that handles the allow_pickle parameter
def custom_np_load(*args, **kwargs):
    allow_pickle = kwargs.pop('allow_pickle', False)
    mmap_mode = kwargs.pop('mmap_mode', None)
    if mmap_mode is not None:
        kwargs['mode'] = mmap_mode
        return np.lib.format.open_memmap(*args, **kwargs)
    else:
        if allow_pickle:
             return orig_np_load(*args, allow_pickle=allow_pickle, **kwargs)
        else:
             return orig_np_load(*args, **kwargs)
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank, get_world_size
from torch import amp
from dataclasses import dataclass, field
import itertools
from tqdm import tqdm
import inspect
import functools
import requests
import gzip
import shutil
import json

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    h5py = None
    H5PY_AVAILABLE = False
    print("WARNING: h5py not found (`pip install h5py`). HDF5 data handling will fail.")

try:
    from Bio import SeqIO, Seq, SeqRecord
    BIOPYTHON_AVAILABLE = True
except ImportError:
    SeqIO, Seq, SeqRecord = None, None, None
    BIOPYTHON_AVAILABLE = False
    print("WARNING: BioPython not found (`pip install biopython`). FASTA/Stockholm parsing will fail.")

try:
    import ViennaRNA
    VIENNARNA_AVAILABLE = True
except ImportError:
    ViennaRNA = None
    VIENNARNA_AVAILABLE = False
    print("WARNING: ViennaRNA Python bindings not found (`pip install ViennaRNA`). Secondary structure prediction will fail.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False
    print("WARNING: NetworkX not found (`pip install networkx`). 3D coordinate calculation will fail.")

try:
    import gffutils
    GFFUTILS_AVAILABLE = True
except ImportError:
    gffutils = None
    GFFUTILS_AVAILABLE = False
    print("INFO: gffutils not found (`pip install gffutils`). Falling back to basic GTF/GFF parsing.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

logger = logging.getLogger("WuBuNestingHybridTrainer")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True)

EPS = 1e-7
DATA_DIR = "data_h5"
HDF5_COMPRESSION = "gzip"
HDF5_COMPRESSION_LEVEL = 4
CHUNK_SIZE_HDF5 = 16384

NUCLEOTIDE_MAP = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4, '-': 4, 'T': 1}
NUCLEOTIDE_VOCAB_SIZE = 5
STRUCTURE_MAP = {'.': 0, '(': 1, ')': 2, '[': 3, ']': 4, '{': 5, '}': 6, '<': 7, '>': 8, '?': 9}
STRUCTURE_VOCAB_SIZE = len(STRUCTURE_MAP)
REGION_MAP = {'UNKNOWN': 0, '5UTR': 1, 'CDS': 2, '3UTR': 3, 'Intron': 4, 'ncRNA': 5, 'StartCodon': 6, 'StopCodon': 7, 'Intergenic': 8}
REGION_VOCAB_SIZE = len(REGION_MAP)
STRUCTURE_SOURCE_MAP = {'Consensus': 0, 'PredictedMFE': 1, 'Unknown': 2}
STRUCTURE_SOURCE_VOCAB_SIZE = len(STRUCTURE_SOURCE_MAP)
CODONS_TRIPLET = sorted([a+b+c for a in 'AUCG' for b in 'AUCG' for c in 'AUCG'])
CODON_MAP = {codon: i for i, codon in enumerate(CODONS_TRIPLET)}
CODON_MAP['START'] = len(CODON_MAP)
CODON_MAP['STOP_UAA'] = len(CODON_MAP)
CODON_MAP['STOP_UAG'] = len(CODON_MAP)
CODON_MAP['STOP_UGA'] = len(CODON_MAP)
CODON_MAP['NonCoding'] = len(CODON_MAP)
CODON_MAP['Unknown'] = len(CODON_MAP)
CODON_VOCAB_SIZE = len(CODON_MAP)
DATASET_SOURCE_MAP = {'rfam_seed': 0, 'gencode_human_cdna': 1, 'refseq_human_rna': 2, 'Unknown': 3}
DATASET_SOURCE_VOCAB_SIZE = len(DATASET_SOURCE_MAP)
COMBINED_DATA_INFO_FILE = "combined_rna_dataset_info_h5.json"

# =====================================================================
# Data Preparation Utilities (HDF5 & Feature Enrichment)
# =====================================================================
def download_file(url: str, dest_path: str, chunk_size=8192):
    temp_dest_path = dest_path + ".part"
    try:
        logger.info(f"Downloading {url} to {dest_path}...")
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(temp_dest_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path), total=total_size, unit='iB', unit_scale=True, unit_divisor=1024, leave=False
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    size = f.write(chunk)
                    bar.update(size)
        shutil.move(temp_dest_path, dest_path)
        logger.info(f"Download complete: {dest_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed for {url}: {e}")
        if os.path.exists(temp_dest_path): _try_remove_file(temp_dest_path)
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during download of {url}: {e}")
        if os.path.exists(temp_dest_path): _try_remove_file(temp_dest_path)
        return False

def _try_remove_file(filepath: str, max_retries: int = 5, initial_delay: float = 0.2):
    if not os.path.exists(filepath): return True
    retry_delay = initial_delay
    for attempt in range(max_retries):
        try:
            os.remove(filepath)
            logger.info(f"Successfully removed file '{filepath}' on attempt {attempt + 1}.")
            return True
        except PermissionError as pe:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to remove '{filepath}': {pe}. Retrying...")
            time.sleep(retry_delay); retry_delay *= 1.5
        except FileNotFoundError: return True
        except Exception as remove_err:
            logger.error(f"Unexpected error removing '{filepath}' on attempt {attempt + 1}: {remove_err}", exc_info=True)
            return False
    logger.error(f"Failed to remove '{filepath}' after {max_retries} attempts.")
    return False

def _predict_structure(sequence: str) -> Tuple[Optional[str], Optional[float]]:
    if not VIENNARNA_AVAILABLE: return None, None
    if not sequence or not isinstance(sequence, str): return None, None
    valid_chars = "ACGU"
    cleaned_seq = "".join(c for c in sequence.upper().replace("T", "U") if c in valid_chars)
    if not cleaned_seq: return None, None
    try:
        struct, mfe = ViennaRNA.fold(cleaned_seq)
        return struct, mfe
    except Exception as e:
        logger.warning(f"ViennaRNA fold failed for seq (len {len(cleaned_seq)}): {e}", exc_info=False)
        return None, None

def _calculate_3d_coords(structure: str) -> Optional[np.ndarray]:
    if not structure or not NETWORKX_AVAILABLE: return None
    seq_len = len(structure)
    if seq_len == 0: return None
    try:
        bp_graph = nx.Graph()
        stack = []
        for i, char in enumerate(structure):
            bp_graph.add_node(i)
            if char == '(': stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    bp_graph.add_edge(i, j, type='basepair')
            if i > 0: bp_graph.add_edge(i, i - 1, type='backbone')

        pos_2d = nx.spring_layout(bp_graph, dim=2, seed=hash(structure) % (2**32), k=1.0/math.sqrt(seq_len) if seq_len > 1 else 1.0, iterations=50)
        coords_3d = np.zeros((seq_len, 3), dtype=np.float32)
        nesting_level = 0
        for i in range(seq_len):
            if structure[i] == '(': nesting_level += 1
            coords_3d[i, 0] = pos_2d[i][0]
            coords_3d[i, 1] = pos_2d[i][1]
            coords_3d[i, 2] = -nesting_level * 0.5
            if structure[i] == ')': nesting_level = max(0, nesting_level - 1)

        center = np.mean(coords_3d, axis=0)
        coords_3d -= center
        scale = np.max(np.abs(coords_3d))
        if scale > EPS: coords_3d /= scale
        return coords_3d
    except Exception as e:
        logger.warning(f"3D coordinate calculation failed for struct (len {seq_len}): {e}", exc_info=False)
        return None

def _get_region_annotations_from_gff(gff_db_path: str, record_id: str, seq_len: int) -> Optional[List[int]]:
    if not GFFUTILS_AVAILABLE or gff_db_path is None or not os.path.exists(gff_db_path): return None
    try:
        db = gffutils.FeatureDB(gff_db_path)
        region_indices = [REGION_MAP['UNKNOWN']] * seq_len
        for feature in db.features_of_type(('CDS', 'five_prime_UTR', 'three_prime_UTR', 'ncRNA', 'exon', 'intron'), seqid=record_id):
            region_type_str = feature.featuretype
            region_idx = REGION_MAP.get(region_type_str.upper() if region_type_str == 'CDS' else region_type_str)
            if region_idx is None:
                if region_type_str == 'five_prime_UTR': region_idx = REGION_MAP['5UTR']
                elif region_type_str == 'three_prime_UTR': region_idx = REGION_MAP['3UTR']
                elif region_type_str == 'ncRNA': region_idx = REGION_MAP['ncRNA']
                elif region_type_str == 'intron': region_idx = REGION_MAP['Intron']
                else: region_idx = REGION_MAP['UNKNOWN']

            start = max(0, feature.start - 1)
            end = min(seq_len, feature.end)
            for i in range(start, end):
                current_region = region_indices[i]
                if region_idx == REGION_MAP['CDS']: region_indices[i] = region_idx
                elif region_idx in [REGION_MAP['5UTR'], REGION_MAP['3UTR']] and current_region != REGION_MAP['CDS']: region_indices[i] = region_idx
                elif region_idx == REGION_MAP['ncRNA'] and current_region not in [REGION_MAP['CDS'], REGION_MAP['5UTR'], REGION_MAP['3UTR']]: region_indices[i] = region_idx
                elif region_idx == REGION_MAP['Intron'] and current_region not in [REGION_MAP['CDS'], REGION_MAP['5UTR'], REGION_MAP['3UTR'], REGION_MAP['ncRNA']]: region_indices[i] = region_idx
                elif current_region == REGION_MAP['UNKNOWN']: region_indices[i] = region_idx
        return region_indices
    except Exception as e:
        logger.error(f"Error reading GFF DB '{gff_db_path}' for ID '{record_id}': {e}", exc_info=True)
        return None

def _get_codon_indices(nucleotide_indices: List[int], region_indices: List[int]) -> List[int]:
    seq_len = len(nucleotide_indices)
    codon_indices = [CODON_MAP['Unknown']] * seq_len
    nt_map_rev = {v: k for k, v in NUCLEOTIDE_MAP.items() if k != '-' and k != 'N'}
    for i in range(seq_len):
        if region_indices[i] == REGION_MAP['CDS']:
            cds_start = i
            while cds_start > 0 and region_indices[cds_start - 1] == REGION_MAP['CDS']: cds_start -= 1
            frame_pos = (i - cds_start) % 3
            if i >= frame_pos and (i - frame_pos + 2) < seq_len:
                is_full_codon_in_cds = all(region_indices[j] == REGION_MAP['CDS'] for j in range(i - frame_pos, i - frame_pos + 3))
                if is_full_codon_in_cds:
                    nuc1_idx, nuc2_idx, nuc3_idx = nucleotide_indices[i - frame_pos], nucleotide_indices[i - frame_pos + 1], nucleotide_indices[i - frame_pos + 2]
                    nuc1, nuc2, nuc3 = nt_map_rev.get(nuc1_idx, 'N'), nt_map_rev.get(nuc2_idx, 'N'), nt_map_rev.get(nuc3_idx, 'N')
                    codon_str = nuc1 + nuc2 + nuc3
                    if 'N' in codon_str: codon_indices[i] = CODON_MAP['Unknown']
                    else:
                        if codon_str in ["UAA", "UAG", "UGA"]: codon_indices[i] = CODON_MAP.get(f"STOP_{codon_str}", CODON_MAP['Unknown'])
                        else: codon_indices[i] = CODON_MAP.get(codon_str, CODON_MAP['Unknown'])
                else: codon_indices[i] = CODON_MAP['NonCoding']
            else: codon_indices[i] = CODON_MAP['NonCoding']
        else: codon_indices[i] = CODON_MAP['NonCoding']
    return codon_indices

def parse_fasta_to_streams(fasta_path: str, dataset_source_index: int, gff_db_path: Optional[str] = None) -> Dict[str, List[Any]]:
    if not BIOPYTHON_AVAILABLE: raise ImportError("BioPython is required for parsing FASTA files.")
    streams = defaultdict(list)
    open_func = gzip.open if fasta_path.endswith(".gz") else open
    unknown_nt_idx = NUCLEOTIDE_MAP.get('N', 4)
    logger.info(f"Parsing FASTA {fasta_path} to streams...")
    try:
        with open_func(fasta_path, 'rt', errors='ignore') as f:
            for record in SeqIO.parse(f, "fasta"):
                seq_str = str(record.seq).upper().replace("T", "U")
                seq_len = len(seq_str)
                if seq_len == 0: continue
                nucleotide_indices = [NUCLEOTIDE_MAP.get(nt, unknown_nt_idx) for nt in seq_str]
                streams['nucleotides'].extend(nucleotide_indices)
                structure, _ = _predict_structure(seq_str)
                structure_indices = [STRUCTURE_MAP.get(s, STRUCTURE_MAP['?']) for s in structure] if structure else [STRUCTURE_MAP['?']] * seq_len
                coords = _calculate_3d_coords(structure) if structure else np.zeros((seq_len, 3), dtype=np.float32)
                streams['structure_symbols'].extend(structure_indices)
                streams['coords'].extend(coords.tolist())
                streams['structure_sources'].extend([STRUCTURE_SOURCE_MAP['PredictedMFE'] if structure else STRUCTURE_SOURCE_MAP['Unknown']] * seq_len)
                region_indices = _get_region_annotations_from_gff(gff_db_path, record.id, seq_len) if gff_db_path else [REGION_MAP['UNKNOWN']] * seq_len
                if region_indices is None: region_indices = [REGION_MAP['UNKNOWN']] * seq_len
                streams['region_types'].extend(region_indices)
                streams['codons'].extend(_get_codon_indices(nucleotide_indices, region_indices))
                streams['dataset_sources'].extend([dataset_source_index] * seq_len)
        total_len = len(streams['nucleotides'])
        logger.info(f"Parsed {total_len:,} total entries from FASTA into streams.")
        for name, stream_list in streams.items():
            if len(stream_list) != total_len:
                logger.error(f"Stream length mismatch! Nucleotides: {total_len}, {name}: {len(stream_list)}. Padding/truncating!")
                padding_val = [0.0,0.0,0.0] if name == 'coords' else (CODON_MAP['Unknown'] if name == 'codons' else 0)
                if len(stream_list) < total_len: stream_list.extend([padding_val] * (total_len - len(stream_list)))
                else: streams[name] = stream_list[:total_len]
        return dict(streams)
    except FileNotFoundError: logger.error(f"FASTA file not found: {fasta_path}"); raise
    except Exception as e: logger.error(f"Error parsing FASTA {fasta_path}: {e}", exc_info=True); raise

def parse_stockholm_to_streams(stockholm_path: str, dataset_source_index: int) -> Dict[str, List[Any]]:
    if not BIOPYTHON_AVAILABLE: raise ImportError("BioPython is required for parsing Stockholm files.")
    streams = defaultdict(list)
    open_func = gzip.open if stockholm_path.endswith(".gz") else open
    unknown_nt_idx = NUCLEOTIDE_MAP.get('N', 4)
    logger.info(f"Parsing Stockholm {stockholm_path} to streams...")
    try:
        with open_func(stockholm_path, 'rt', errors='ignore') as handle:
            for record in SeqIO.parse(handle, "stockholm"):
                seq_str = str(record.seq).upper().replace("T", "U").replace("-", "")
                seq_len = len(seq_str)
                if seq_len == 0: continue
                nucleotide_indices = [NUCLEOTIDE_MAP.get(nt, unknown_nt_idx) for nt in seq_str]
                streams['nucleotides'].extend(nucleotide_indices)
                structure = None; structure_source_val = STRUCTURE_SOURCE_MAP['Unknown']
                if "SS_cons" in record.letter_annotations:
                    structure = record.letter_annotations["SS_cons"].replace('-', '.')
                    structure_source_val = STRUCTURE_SOURCE_MAP['Consensus']
                else:
                    structure_pred_res, _ = _predict_structure(seq_str)
                    if structure_pred_res: structure, structure_source_val = structure_pred_res, STRUCTURE_SOURCE_MAP['PredictedMFE']
                structure_indices = [STRUCTURE_MAP.get(s, STRUCTURE_MAP['?']) for s in structure] if structure else [STRUCTURE_MAP['?']] * seq_len
                coords = _calculate_3d_coords(structure) if structure else np.zeros((seq_len, 3), dtype=np.float32)
                streams['structure_symbols'].extend(structure_indices)
                streams['coords'].extend(coords.tolist())
                streams['structure_sources'].extend([structure_source_val] * seq_len)
                region_indices = [REGION_MAP['UNKNOWN']] * seq_len
                streams['region_types'].extend(region_indices)
                streams['codons'].extend(_get_codon_indices(nucleotide_indices, region_indices))
                streams['dataset_sources'].extend([dataset_source_index] * seq_len)
        total_len = len(streams['nucleotides'])
        logger.info(f"Parsed {total_len:,} total entries from Stockholm into streams.")
        for name, stream_list in streams.items():
            if len(stream_list) != total_len:
                logger.error(f"Stockholm stream length mismatch! Nucleotides: {total_len}, {name}: {len(stream_list)}. Fixing...")
                padding_val = [0.0,0.0,0.0] if name == 'coords' else (CODON_MAP['Unknown'] if name == 'codons' else 0)
                if len(stream_list) < total_len: stream_list.extend([padding_val] * (total_len - len(stream_list)))
                else: streams[name] = stream_list[:total_len]
        return dict(streams)
    except FileNotFoundError: logger.error(f"Stockholm file not found: {stockholm_path}"); raise
    except Exception as e: logger.error(f"Error parsing Stockholm {stockholm_path}: {e}", exc_info=True); raise

def prepare_dataset_h5(dataset_name: str, data_dir: str, gff_db_paths: Optional[Dict[str, str]] = None) -> Optional[Tuple[str, int]]:
    if not H5PY_AVAILABLE: logger.error("h5py package not found."); return None
    os.makedirs(data_dir, exist_ok=True)
    processed_h5_path = os.path.join(data_dir, f"{dataset_name}_streams.h5")
    if os.path.exists(processed_h5_path):
        try:
            with h5py.File(processed_h5_path, 'r') as f:
                if 'nucleotides' in f and f['nucleotides'].shape[0] > 0:
                    logger.info(f"Found: {processed_h5_path} (Entries: {f['nucleotides'].shape[0]:,})")
                    return processed_h5_path, f['nucleotides'].shape[0]
                _try_remove_file(processed_h5_path)
        except Exception as e: logger.warning(f"Error checking {processed_h5_path}: {e}. Re-processing."); _try_remove_file(processed_h5_path)
    logger.info(f"HDF5 '{processed_h5_path}' not found/invalid. Processing...")
    ENSEMBL_RELEASE = "113"
    datasets_info = {
        "rfam_seed": {"url": "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz", "raw_filename": "Rfam.seed.gz", "format": "stockholm", "parser": parse_stockholm_to_streams, "requires_biopython": True, "source_index": DATASET_SOURCE_MAP.get("rfam_seed", DATASET_SOURCE_MAP['Unknown'])},
        "gencode_human_cdna": {"url": f"https://ftp.ensembl.org/pub/release-{ENSEMBL_RELEASE}/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz", "raw_filename": f"Homo_sapiens.GRCh38.cdna.all.release{ENSEMBL_RELEASE}.fa.gz", "format": "fasta", "warning": "Large cDNA.", "parser": parse_fasta_to_streams, "requires_biopython": True, "annotation_key": "gencode", "source_index": DATASET_SOURCE_MAP.get("gencode_human_cdna", DATASET_SOURCE_MAP['Unknown'])},
        "refseq_human_rna": {"url": "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_rna.fna.gz", "raw_filename": "GRCh38_latest_rna.fna.gz", "format": "fasta", "warning": "RefSeq RNA.", "parser": parse_fasta_to_streams, "requires_biopython": True, "annotation_key": "refseq", "source_index": DATASET_SOURCE_MAP.get("refseq_human_rna", DATASET_SOURCE_MAP['Unknown'])},
    }
    if dataset_name not in datasets_info: logger.error(f"Unknown dataset: '{dataset_name}'."); return None
    info = datasets_info[dataset_name]
    raw_file_path = os.path.join(data_dir, info["raw_filename"])
    if info.get("requires_biopython", False) and not BIOPYTHON_AVAILABLE: logger.error(f"'{dataset_name}' needs BioPython."); return None
    if info["format"] == "stockholm" and not VIENNARNA_AVAILABLE: logger.warning(f"ViennaRNA missing for {dataset_name}.")
    if info["format"] == "fasta" and not (VIENNARNA_AVAILABLE and NETWORKX_AVAILABLE): logger.warning(f"Vienna/NetworkX missing for {dataset_name}.")
    if "warning" in info: logger.warning(info["warning"])
    if not os.path.exists(raw_file_path) and not download_file(info["url"], raw_file_path): logger.error(f"Failed download for {dataset_name}"); return None
    gff_path = None
    if info["format"] == "fasta" and "annotation_key" in info and gff_db_paths:
        gff_path = gff_db_paths.get(info["annotation_key"])
        if gff_path and (not os.path.exists(gff_path) or not GFFUTILS_AVAILABLE):
            logger.warning(f"GFF DB '{gff_path}' for {dataset_name} not found or gffutils missing. No region annotation."); gff_path = None
        elif gff_path: logger.info(f"Using GFF DB '{gff_path}' for {dataset_name}.")
    try:
        parser_func = info.get("parser")
        streams_dict = parser_func(raw_file_path, info["source_index"], gff_db_path=gff_path) if info["format"] == "fasta" else parser_func(raw_file_path, info["source_index"])
        if not streams_dict or 'nucleotides' not in streams_dict or not streams_dict['nucleotides']: logger.error(f"Parsing empty for {dataset_name}."); return None
        num_entries = len(streams_dict['nucleotides'])
        logger.info(f"Parsing complete. Got {num_entries:,} entries across {len(streams_dict)} streams.")
        with h5py.File(processed_h5_path, 'w') as f:
            f.attrs['dataset_name'], f.attrs['creation_time'], f.attrs['total_entries'] = dataset_name, datetime.now().isoformat(), num_entries
            for stream_name, data_list in streams_dict.items():
                try:
                    dtype = np.float32 if stream_name == 'coords' else (np.uint8 if stream_name in ['nucleotides', 'structure_symbols', 'region_types', 'structure_sources', 'dataset_sources'] else (np.uint16 if stream_name == 'codons' else object))
                    data_array = np.array(data_list, dtype=dtype)
                    chunk_shape = (CHUNK_SIZE_HDF5,) if data_array.ndim == 1 else ((max(1, CHUNK_SIZE_HDF5 // (data_array.shape[1] or 1)), data_array.shape[1]) if data_array.ndim == 2 else None)
                    logger.debug(f"HDF5 ds '{stream_name}' Shape: {data_array.shape}, DType: {data_array.dtype}, Chunks: {chunk_shape}")
                    f.create_dataset(stream_name, data=data_array, chunks=chunk_shape, compression=HDF5_COMPRESSION, compression_opts=HDF5_COMPRESSION_LEVEL, shuffle=True)
                except Exception as h5_err: logger.error(f"Error writing stream '{stream_name}': {h5_err}", exc_info=True); _try_remove_file(processed_h5_path); return None
                del data_list, data_array; gc.collect()
        logger.info(f"HDF5 dataset saved: {processed_h5_path}")
        return processed_h5_path, num_entries
    except Exception as e: logger.error(f"HDF5 prep error for {dataset_name}: {e}", exc_info=True); _try_remove_file(processed_h5_path); return None

def prepare_combined_dataset_h5(data_dir: str, max_datasets: Optional[int] = None, gff_db_paths: Optional[Dict[str, str]] = None) -> Optional[str]:
    if not H5PY_AVAILABLE: logger.error("h5py not found."); return None
    os.makedirs(data_dir, exist_ok=True)
    combined_h5_path = os.path.join(data_dir, "combined_rna_streams.h5")
    metadata_path = os.path.join(data_dir, COMBINED_DATA_INFO_FILE)
    temp_combined_path = combined_h5_path + ".building"
    if os.path.exists(temp_combined_path) and not _try_remove_file(temp_combined_path): return None
    if os.path.exists(combined_h5_path) and os.path.exists(metadata_path):
        logger.info(f"Combined HDF5 dataset found: {combined_h5_path}")
        try:
            with open(metadata_path, 'r') as f: meta = json.load(f)
            with h5py.File(combined_h5_path, 'r') as hf:
                if 'nucleotides' in hf and meta.get('total_entries') == hf['nucleotides'].shape[0]:
                    logger.info("Metadata matches HDF5. Using existing."); return combined_h5_path
            logger.warning("Metadata/HDF5 mismatch. Rebuilding."); _try_remove_file(combined_h5_path); _try_remove_file(metadata_path)
        except Exception as check_err: logger.warning(f"Error checking existing combined HDF5: {check_err}. Rebuilding."); _try_remove_file(combined_h5_path); _try_remove_file(metadata_path)
    logger.info(f"Combined HDF5 '{combined_h5_path}' or metadata invalid. Creating...")
    dataset_names_to_process = ["rfam_seed", "gencode_human_cdna", "refseq_human_rna"]
    if max_datasets is not None and max_datasets > 0: dataset_names_to_process = dataset_names_to_process[:max_datasets]; logger.warning(f"Limiting to first {max_datasets} sources: {dataset_names_to_process}")
    prepared_datasets_info, total_expected_entries, all_stream_names, stream_dtypes, stream_shapes = [], 0, set(), {}, {}
    logger.info("--- Stage 1: Preparing individual HDF5 datasets ---")
    all_datasets_prepared = True
    for dataset_name in dataset_names_to_process:
        logger.info(f"Preparing '{dataset_name}'...")
        prep_result = prepare_dataset_h5(dataset_name, data_dir, gff_db_paths)
        if prep_result:
            path, size = prep_result
            if size > 0:
                current_streams = {}
                with h5py.File(path, 'r') as hf:
                    for stream_name in hf.keys():
                        all_stream_names.add(stream_name); dset = hf[stream_name]
                        current_streams[stream_name] = {'dtype': str(dset.dtype), 'shape_suffix': dset.shape[1:]}
                        if stream_name not in stream_dtypes: stream_dtypes[stream_name], stream_shapes[stream_name] = dset.dtype, dset.shape[1:]
                        elif stream_dtypes[stream_name] != dset.dtype or stream_shapes[stream_name] != dset.shape[1:]: logger.error(f"Inconsistent stream '{stream_name}' props. Aborting combine."); return None
                prepared_datasets_info.append((dataset_name, path, size, current_streams))
                total_expected_entries += size
                logger.info(f"'{dataset_name}' ready (Size: {size:,}). Cumulative: {total_expected_entries:,}")
            else: logger.warning(f"Dataset '{dataset_name}' prepared but empty. Skipping.")
        else: logger.warning(f"Failed to prepare '{dataset_name}'. Skipping."); all_datasets_prepared = False
    if not prepared_datasets_info or total_expected_entries == 0: logger.error("No valid datasets prepared. Cannot combine."); return None
    if not all_datasets_prepared: logger.warning("One or more datasets failed. Combined dataset incomplete.")
    logger.info(f"--- Stage 1 Complete: Total expected entries: {total_expected_entries:,} across {len(all_stream_names)} streams: {sorted(list(all_stream_names))} ---")
    logger.info(f"--- Stage 2: Creating combined file '{temp_combined_path}' and appending data ---")
    dataset_boundaries, current_offset, chunk_buffer_size = [], 0, 100_000
    try:
        with h5py.File(temp_combined_path, 'w') as combined_f:
            combined_f.attrs['creation_time'], combined_f.attrs['total_entries'], combined_f.attrs['stream_names'] = datetime.now().isoformat(), total_expected_entries, sorted(list(all_stream_names))
            h5_datasets = {}
            for stream_name in all_stream_names:
                dtype, shape_suffix = stream_dtypes[stream_name], stream_shapes[stream_name]
                full_shape = (total_expected_entries,) + tuple(shape_suffix)
                chunk_shape = (CHUNK_SIZE_HDF5,) if not shape_suffix else ((max(1, CHUNK_SIZE_HDF5 // (shape_suffix[0] or 1)), shape_suffix[0]) if len(shape_suffix) == 1 else None)
                logger.debug(f"Creating combined dataset '{stream_name}' Shape: {full_shape}, DType: {dtype}, Chunks: {chunk_shape}")
                h5_datasets[stream_name] = combined_f.create_dataset(stream_name, shape=full_shape, dtype=dtype, chunks=chunk_shape, compression=HDF5_COMPRESSION, compression_opts=HDF5_COMPRESSION_LEVEL, shuffle=True)
            for name, path, size, streams_info in prepared_datasets_info:
                logger.info(f"Appending '{name}' ({size:,} entries) from {path}...")
                append_start_index, append_end_index = current_offset, current_offset + size
                with h5py.File(path, 'r') as source_f:
                    for stream_name in all_stream_names:
                        if stream_name in source_f:
                            source_dset, target_dset = source_f[stream_name], h5_datasets[stream_name]
                            logger.debug(f"  Appending stream '{stream_name}'...")
                            for i in tqdm(range(0, size, chunk_buffer_size), desc=f"    {stream_name}", leave=False, unit="chunk"):
                                start, end = i, min(i + chunk_buffer_size, size)
                                target_dset[append_start_index + start : append_start_index + end] = source_dset[start:end]
                        else:
                            logger.warning(f"  Stream '{stream_name}' missing in '{path}'. Filling with zeros.")
                            fill_val = 0.0 if stream_name == 'coords' else (CODON_MAP['Unknown'] if stream_name == 'codons' else (REGION_MAP['UNKNOWN'] if stream_name == 'region_types' else 0))
                            fill_chunk = np.full((size,) + tuple(stream_shapes[stream_name]), fill_val, dtype=stream_dtypes[stream_name])
                            h5_datasets[stream_name][append_start_index : append_end_index] = fill_chunk
                dataset_boundaries.append({"name": name, "start_index": current_offset, "end_index": append_end_index})
                current_offset = append_end_index
                logger.info(f"Finished appending '{name}'. Current offset: {current_offset:,}")
        if current_offset != total_expected_entries: logger.error(f"Final write offset ({current_offset:,}) != expected size ({total_expected_entries:,}). Aborting."); _try_remove_file(temp_combined_path); return None
        logger.info(f"Renaming temporary file {temp_combined_path} to {combined_h5_path}")
        os.replace(temp_combined_path, combined_h5_path)
        metadata = {"combined_h5_path": os.path.basename(combined_h5_path), "dataset_boundaries": dataset_boundaries, "total_entries": total_expected_entries, "stream_info": {s: {'dtype': str(stream_dtypes[s]), 'shape_suffix': stream_shapes[s]} for s in all_stream_names}, "creation_time": datetime.now().isoformat(), "included_datasets": [info["name"] for info in dataset_boundaries]}
        logger.debug(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'w') as f: json.dump(metadata, f, indent=2)
        logger.info(f"Combined HDF5 dataset saved: {combined_h5_path}"); logger.info(f"Metadata saved: {metadata_path}")
        return combined_h5_path
    except Exception as build_err:
        logger.error(f"Error during Stage 2 (building combined HDF5): {build_err}", exc_info=True)
        _try_remove_file(temp_combined_path); _try_remove_file(combined_h5_path); _try_remove_file(metadata_path)
        return None

# =====================================================================
# Dataset Classes (HDF5 Adaptation)
# =====================================================================
class BaseWuBuDatasetH5(IterableDataset):
    def __init__(self, h5_file_path: str, context_size: int, stream_names: Optional[List[str]] = None):
        super().__init__()
        if not H5PY_AVAILABLE: raise RuntimeError("h5py package is required.")
        if not os.path.exists(h5_file_path): raise FileNotFoundError(f"Dataset HDF5 file not found: {h5_file_path}")
        if context_size <= 0: raise ValueError("context_size must be positive")
        self.h5_file_path, self.context_size, self.data_size, self.num_possible_samples, self.seed, self.epoch = h5_file_path, context_size, 0, 0, None, 0
        self.stream_names, self.stream_dtypes, self.stream_shapes = [], {}, {}
        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                if 'nucleotides' not in f: raise ValueError(f"HDF5 file {h5_file_path} missing 'nucleotides' dataset.")
                self.data_size = f['nucleotides'].shape[0]
                if self.data_size == 0: raise ValueError("HDF5 file contains no data.")
                available_streams = list(f.keys())
                self.stream_names = [s for s in (stream_names or available_streams) if s in available_streams]
                if stream_names and set(stream_names) - set(available_streams): logger.warning(f"Requested streams not in HDF5: {set(stream_names) - set(available_streams)}")
                for s_name in self.stream_names:
                    dset = f[s_name]
                    self.stream_dtypes[s_name], self.stream_shapes[s_name] = dset.dtype, dset.shape[1:]
                    if dset.shape[0] != self.data_size: raise ValueError(f"Inconsistent data size for stream '{s_name}' ({dset.shape[0]}) vs nucleotides ({self.data_size})")
            self.num_possible_samples = max(0, self.data_size - self.context_size)
            if self.num_possible_samples == 0: raise ValueError(f"No samples possible: Ctx={self.context_size:,} DataSize={self.data_size:,}.")
            logger.info(f"{self.__class__.__name__} '{os.path.basename(h5_file_path)}': Size={self.data_size:,}, Samples={self.num_possible_samples:,}, Streams={self.stream_names}")
        except Exception as e: logger.error(f"Error initializing HDF5 dataset {h5_file_path}: {e}", exc_info=True); raise

    def _get_worker_info(self) -> Tuple[int, int, int, int]:
        worker_info = torch.utils.data.get_worker_info()
        num_workers, worker_id = (worker_info.num_workers, worker_info.id) if worker_info else (1, 0)
        rank, world_size = (get_rank(), get_world_size()) if is_initialized() else (0, 1)
        return rank, world_size, worker_id, num_workers

    def __len__(self):
        if self.num_possible_samples == 0: return 0
        rank, world_size, worker_id, num_workers = self._get_worker_info()
        total_effective_workers = max(1, num_workers * world_size)
        base_samples, remainder = divmod(self.num_possible_samples, total_effective_workers)
        global_worker_id = rank * num_workers + worker_id
        num_samples = base_samples + (1 if global_worker_id < remainder else 0)
        return max(1, num_samples) if self.num_possible_samples > 0 else 0

    def _iterate_indices(self, indices: np.ndarray):
        rank, world_size, worker_id, num_workers = self._get_worker_info()
        h5_file = None
        try:
            h5_file = h5py.File(self.h5_file_path, 'r')
            h5_datasets = {name: h5_file[name] for name in self.stream_names if name in h5_file}
            if not h5_datasets: logger.error(f"W:{worker_id} R:{rank}: No streams loaded."); return
            for idx in indices:
                start_ctx, end_ctx, end_tgt = int(idx), int(idx) + self.context_size, int(idx) + self.context_size + 1
                if end_tgt > self.data_size: continue
                batch_dict, valid_slice = {}, True
                try:
                    for stream_name, dset in h5_datasets.items():
                        ctx_slice, tgt_slice = dset[start_ctx : end_ctx], dset[start_ctx + 1 : end_tgt]
                        if ctx_slice.shape[0] != self.context_size or tgt_slice.shape[0] != self.context_size:
                            logger.warning(f"W:{worker_id} R:{rank}: Slice length mismatch stream '{stream_name}', index {idx}. Skip."); valid_slice = False; break
                        batch_dict[f'context_{stream_name}'] = torch.from_numpy(ctx_slice)
                        batch_dict[f'target_{stream_name}'] = torch.from_numpy(tgt_slice)
                    if valid_slice:
                        for key_base in ['nucleotides', 'structure_symbols', 'region_types', 'structure_sources', 'dataset_sources', 'codons']:
                            for prefix in ['context_', 'target_']:
                                key = f"{prefix}{key_base}"
                                if key in batch_dict: batch_dict[key] = batch_dict[key].long()
                        for prefix in ['context_', 'target_']:
                            key = f"{prefix}coords"
                            if key in batch_dict: batch_dict[key] = batch_dict[key].float()
                        yield batch_dict
                except IndexError: logger.warning(f"W:{worker_id} R:{rank}: HDF5 IndexError for index {idx}. Skip."); continue
                except Exception as e: logger.error(f"W:{worker_id} R:{rank}: Error processing index {idx}: {e}"); continue
        except FileNotFoundError: logger.error(f"W:{worker_id} R:{rank}: HDF5 file not found: {self.h5_file_path}")
        except Exception as e: logger.error(f"W:{worker_id} R:{rank}: HDF5 Iterator failed: {e}", exc_info=True)
        finally:
            if h5_file:
                try:
                    h5_file.close()
                    logger.debug(f"W:{worker_id} R:{rank}: Closed HDF5.")
                except Exception as close_err:
                    logger.warning(f"W:{worker_id} R:{rank}: Error closing HDF5: {close_err}")
            gc.collect()

    def set_seed(self, seed: int): self.seed = seed
    def set_epoch(self, epoch: int): self.epoch = epoch

class WuBuNestingDatasetH5(BaseWuBuDatasetH5):
    def __init__(self, h5_file_path: str, context_size: int, stream_names: Optional[List[str]] = None):
        super().__init__(h5_file_path, context_size, stream_names)
    def __iter__(self):
        rank, world_size, worker_id, num_workers = self._get_worker_info()
        if self.num_possible_samples == 0: return iter([])
        total_effective_workers = max(1, num_workers * world_size)
        global_worker_id = rank * num_workers + worker_id
        base_samples, remainder = divmod(self.num_possible_samples, total_effective_workers)
        start_sample_idx = global_worker_id * base_samples + min(global_worker_id, remainder)
        num_samples_this_worker = base_samples + (1 if global_worker_id < remainder else 0)
        end_sample_idx = min(start_sample_idx + num_samples_this_worker, self.num_possible_samples)
        if start_sample_idx >= end_sample_idx: logger.debug(f"W:{worker_id} R:{rank}: No samples assigned."); return iter([])
        worker_indices = np.arange(start_sample_idx, end_sample_idx, dtype=np.int64)
        base_seed = self.seed if self.seed is not None else int(time.time())
        seed_for_worker = (base_seed + global_worker_id + self.epoch * total_effective_workers) % (2**32)
        rng = np.random.default_rng(seed=seed_for_worker)
        rng.shuffle(worker_indices)
        logger.debug(f"W:{worker_id} R:{rank}: Processing {len(worker_indices)} indices from {start_sample_idx} to {end_sample_idx-1} (Seed {seed_for_worker}, Epoch {self.epoch})")
        return self._iterate_indices(worker_indices)

class BalancedWuBuNestingDatasetH5(BaseWuBuDatasetH5):
    def __init__(self, h5_file_path: str, context_size: int, balanced_sampling: bool = True, metadata_path: Optional[str] = None, stream_names: Optional[List[str]] = None):
        super().__init__(h5_file_path, context_size, stream_names)
        self.balanced_sampling = balanced_sampling
        self.dataset_boundaries_info = []
        if metadata_path is None: metadata_path = os.path.join(os.path.dirname(self.h5_file_path), COMBINED_DATA_INFO_FILE)
        if self.balanced_sampling:
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f: meta = json.load(f)
                    self.dataset_boundaries_info = meta.get("dataset_boundaries", [])
                    if self.dataset_boundaries_info and self.dataset_boundaries_info[-1].get("end_index") != self.data_size: logger.warning(f"Metadata boundary end doesn't match data size. Balanced sampling might be inaccurate.")
                    if not self.dataset_boundaries_info: logger.warning("Metadata file found but no boundaries. Disabling balanced sampling."); self.balanced_sampling = False
                    else: logger.info(f"Loaded boundaries for {len(self.dataset_boundaries_info)} datasets for balanced sampling.")
                except Exception as e: logger.warning(f"Failed to load/parse boundaries from {metadata_path}: {e}. Disabling balanced sampling."); self.balanced_sampling = False
            else: logger.warning(f"Metadata file '{metadata_path}' not found. Disabling balanced sampling."); self.balanced_sampling = False
        logger.info(f"BalancedWuBuNestingDatasetH5 Init: Balanced={self.balanced_sampling}")

    def __iter__(self):
        rank, world_size, worker_id, num_workers = self._get_worker_info()
        if self.num_possible_samples == 0: return iter([])
        total_effective_workers = max(1, num_workers * world_size)
        global_worker_id = rank * num_workers + worker_id
        target_samples_this_worker = self.__len__()
        base_seed = self.seed if self.seed is not None else int(time.time())
        seed_for_worker = (base_seed + global_worker_id + self.epoch * total_effective_workers) % (2**32)
        rng = np.random.default_rng(seed=seed_for_worker)
        worker_indices = []
        if self.balanced_sampling and self.dataset_boundaries_info:
            num_source_datasets = len(self.dataset_boundaries_info)
            if num_source_datasets == 0:
                logger.warning(f"W:{worker_id} R:{rank}: No boundaries for balanced sampling. Falling back.")
                worker_indices = rng.choice(self.num_possible_samples, size=target_samples_this_worker, replace=True).tolist()
            else:
                samples_per_ds_target = max(1, target_samples_this_worker // num_source_datasets)
                logger.debug(f"W:{worker_id} R:{rank}: Target {target_samples_this_worker} samples, ~{samples_per_ds_target} per dataset.")
                temp_indices_list = []
                for i, boundary_info in enumerate(self.dataset_boundaries_info):
                    start_idx, end_idx = boundary_info['start_index'], boundary_info['end_index']
                    possible_starts_in_ds = max(0, (end_idx - start_idx) - self.context_size)
                    if possible_starts_in_ds == 0: continue
                    num_to_sample = min(samples_per_ds_target, possible_starts_in_ds)
                    if i < (target_samples_this_worker % num_source_datasets): num_to_sample = min(num_to_sample + 1, possible_starts_in_ds)
                    if num_to_sample > 0:
                        temp_indices_list.extend((start_idx + rng.choice(possible_starts_in_ds, size=num_to_sample, replace=True)).tolist())
                rng.shuffle(temp_indices_list); worker_indices = temp_indices_list
                while len(worker_indices) < target_samples_this_worker:
                    logger.debug(f"W:{worker_id} R:{rank}: Resampling needed ({len(worker_indices)} < {target_samples_this_worker})")
                    worker_indices.extend(rng.choice(self.num_possible_samples, size=(target_samples_this_worker - len(worker_indices)), replace=True).tolist())
                worker_indices = worker_indices[:target_samples_this_worker]
                logger.debug(f"W:{worker_id} R:{rank}: Generated {len(worker_indices)} balanced indices.")
        else:
            worker_indices = rng.choice(self.num_possible_samples, size=target_samples_this_worker, replace=True).tolist()
            logger.debug(f"W:{worker_id} R:{rank}: Generated {len(worker_indices)} simple random indices.")
        return self._iterate_indices(np.array(worker_indices, dtype=np.int64))

def seed_worker(worker_id: int, base_seed: int, rank_offset: int):
    worker_seed = (base_seed + rank_offset + worker_id) % (2**32)
    np.random.seed(worker_seed); random.seed(worker_seed); torch.manual_seed(worker_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(worker_seed)

# =====================================================================
# HAKMEM Components
# =====================================================================
@dataclass
class SamplerConfig: low_entropy_threshold: float = 0.3; medium_entropy_threshold: float = 1.2; high_entropy_threshold: float = 2.5
class GradientStats:
    def __init__(self): self.reset()
    def reset(self): self.total_gradients, self.clipped_gradients, self.max_gradient_norm, self.sum_clip_ratios, self.non_finite_grads_in_step, self.step_stats = 0,0,0.0,0.0,0,{}
    def record_gradient(self, original_norm: float, clipped: bool, clip_ratio: Optional[float] = None):
        if np.isfinite(original_norm): self.total_gradients += 1; self.max_gradient_norm = max(self.max_gradient_norm, original_norm);
        if clipped: self.clipped_gradients += 1; self.sum_clip_ratios += (clip_ratio if clip_ratio is not None else 0.0)
        else: self.non_finite_grads_in_step += 1
    def get_step_stats(self) -> dict:
        if self.total_gradients == 0 and self.non_finite_grads_in_step == 0: return {"gradients_clipped":0,"total_gradients":0,"clip_ratio_avg":0.0,"max_gradient":0.0,"clip_percentage":0.0,"non_finite_grads":0}
        clip_percentage = (self.clipped_gradients / self.total_gradients) * 100 if self.total_gradients > 0 else 0.0
        avg_clip_ratio = self.sum_clip_ratios / self.clipped_gradients if self.clipped_gradients > 0 else 0.0
        return {"gradients_clipped":self.clipped_gradients, "total_gradients":self.total_gradients, "non_finite_grads":self.non_finite_grads_in_step, "clip_ratio_avg":avg_clip_ratio, "max_gradient":self.max_gradient_norm, "clip_percentage":clip_percentage}
    def record_step(self, step: int, skipped: bool = False) -> dict: stats = self.get_step_stats(); stats['step_skipped'] = skipped; self.step_stats[step] = stats; self.reset(); return stats
class HAKMEMEntropyHelper:
    def __init__(self, max_cache_size: int = 50000): self.entropy_cache, self.max_cache_size = {}, max_cache_size
    def _clean_cache(self):
        if len(self.entropy_cache) > self.max_cache_size:
            keys_to_remove = list(itertools.islice(self.entropy_cache.keys(), len(self.entropy_cache) - (self.max_cache_size*4//5)))
            for k in keys_to_remove:
                if k in self.entropy_cache: del self.entropy_cache[k]
    def compute_entropy(self, data_window: Union[np.ndarray, Tuple[int, ...], List[int], bytes, torch.Tensor], vocab_size: int = NUCLEOTIDE_VOCAB_SIZE) -> float:
        item_list = list(data_window) if isinstance(data_window, (tuple, list, bytes)) else (data_window.tolist() if isinstance(data_window, np.ndarray) else (data_window.cpu().long().tolist() if isinstance(data_window, torch.Tensor) else []))
        if not item_list: return 0.0
        item_list_int = [int(b) for b in item_list if isinstance(b, (int, float)) and b >= 0]
        if not item_list_int: return 0.0
        cache_key = f"v{vocab_size}_" + str(tuple(item_list_int))
        if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
        try:
            counts = np.bincount(np.array(item_list_int, dtype=np.int64), minlength=vocab_size)
            if counts.sum() == 0: return 0.0
            probs = counts[counts > 0] / counts.sum()
            result = max(0.0, float(-np.sum(probs * np.log2(probs + EPS))))
            self.entropy_cache[cache_key] = result; self._clean_cache()
            return result
        except Exception as e: logger.warning(f"Entropy calculation failed for key {cache_key}: {e}"); return 0.0
class HAKMEMQController:
    def __init__(self, learning_rate: float=0.01, discount: float=0.95, epsilon: float=0.2, epsilon_decay: float=0.9998, min_epsilon: float=0.01, lr_scale_options: Optional[List[float]]=None, momentum_scale_options: Optional[List[float]]=None, max_q_table_size: int=10000):
        self.q_table: Dict[Tuple, Dict[str, np.ndarray]]={}; self.alpha,self.gamma,self.epsilon,self.min_epsilon,self.epsilon_decay=learning_rate,discount,epsilon,min_epsilon,epsilon_decay
        self.prev_loss, self.prev_state, self.prev_action = None, None, None
        self.action_ranges={'lr_scale':np.array(lr_scale_options or [0.9,0.95,1.0,1.05,1.1],dtype=np.float32),'momentum_scale':np.array(momentum_scale_options or [0.95,0.98,1.0,1.01,1.02],dtype=np.float32)}
        self.num_actions={p:len(s) for p,s in self.action_ranges.items()}
        self.loss_window,self.grad_norm_window,self.lr_window,self.momentum_window,self.performance_window,self.prev_actions_log = deque(maxlen=20),deque(maxlen=20),deque(maxlen=10),deque(maxlen=10),deque(maxlen=50),deque(maxlen=10)
        self.stable_steps, self.oscillation_counter, self.max_q_table_size = 0,0,max(100,max_q_table_size)
        self.q_table_access_count, self.q_table_creation_time = defaultdict(int), {}
        self.flow_coefficient,self.oscillation_penalty,self.stability_reward_bonus,self.large_grad_penalty_factor=0.05,0.15,0.05,0.1
        logger.info(f"QController init: a={self.alpha:.3f},g={self.gamma:.3f},e={self.epsilon:.3f},decay={self.epsilon_decay:.5f},min_e={self.min_epsilon:.3f}, LR actions={self.action_ranges['lr_scale']}, Mom actions={self.action_ranges['momentum_scale']}")
    def get_state(self,lr: float, momentum: float, grad_norm: Optional[float], loss: Optional[float])->Optional[Tuple]:
        if loss is None or grad_norm is None or not np.isfinite(loss) or not np.isfinite(grad_norm): logger.debug("QState skip: Invalid input"); return None
        self.loss_window.append(loss); self.grad_norm_window.append(grad_norm); self.lr_window.append(lr); self.momentum_window.append(momentum)
        if len(self.loss_window)<5 or len(self.grad_norm_window)<5: logger.debug("QState skip: Insufficient history."); return None
        try:
            y_loss=np.array(list(self.loss_window)[-10:],dtype=np.float32); slope_loss=np.polyfit(np.arange(len(y_loss)),y_loss,1)[0] if len(y_loss)>=3 and len(np.unique(y_loss))>1 else 0.
            loss_trend_bin=np.digitize(slope_loss/(abs(np.median(y_loss))+EPS),bins=[-0.05,-0.005,0.005,0.05]).item() if len(y_loss)>=3 else 2
            grad_norm_level_bin=np.digitize(np.median(list(self.grad_norm_window)),bins=[0.1,0.5,1.5,5.0]).item()
            lr_level_bin=np.digitize(lr/1e-4,bins=[0.5,2.0,10.0,50.0]).item(); momentum_level_bin=np.digitize(momentum,bins=[0.85,0.92,0.97]).item()
            if len(self.performance_window)>=5:
                sign_changes=np.sum(np.abs(np.diff(np.sign([r for r in list(self.performance_window)[-5:] if r!=0]))))/2.0
                self.oscillation_counter=min(self.oscillation_counter+1,5) if sign_changes>=2 else max(0,self.oscillation_counter-1)
            oscillation_bin=1 if self.oscillation_counter>=3 else 0
        except Exception as e: logger.error(f"QState calc error: {e}",exc_info=True); return None
        state=(loss_trend_bin,grad_norm_level_bin,oscillation_bin,lr_level_bin,momentum_level_bin); self.q_table_access_count[state]+=1; return state
    def compute_reward(self,current_loss: Optional[float],prev_loss: Optional[float],grad_norm: Optional[float])->float:
        if current_loss is None or prev_loss is None or grad_norm is None or not np.isfinite(current_loss) or not np.isfinite(prev_loss) or not np.isfinite(grad_norm): logger.debug("Reward skip: Invalid input"); return 0.0
        base_reward=np.tanh((prev_loss-current_loss)/(abs(np.median(list(self.loss_window)[:-1]) if len(self.loss_window)>1 else prev_loss)+EPS)*10.0)
        grad_penalty=-self.large_grad_penalty_factor*min(1.0,max(0.0,(grad_norm-5.0)/10.0)) if grad_norm>5.0 else 0.0
        osc_penalty=-self.oscillation_penalty if self.oscillation_counter>=3 else 0.0
        self.performance_window.append(base_reward+grad_penalty+osc_penalty)
        
        current_performance = base_reward + grad_penalty + osc_penalty
        if current_performance > 0.01:
            self.stable_steps += 1
            stab_bonus = min(0.15, self.stability_reward_bonus * math.log1p(self.stable_steps / 5.0))
        else:
            self.stable_steps = 0
            stab_bonus = 0.0
            
        return float(np.clip(base_reward+grad_penalty+osc_penalty+stab_bonus,-1.0,1.0))
    
    def choose_action(self,state: Optional[Tuple])->Dict[str, float]:
        if state is None: return {'lr_scale':1.0,'momentum_scale':1.0}
        if state not in self.q_table: self.q_table[state]={p:np.zeros(self.num_actions[p],dtype=np.float32) for p in self.action_ranges.keys()}; self.q_table_creation_time[state]=time.time(); self.q_table_access_count[state]=1; self._manage_q_table_size()
        self.epsilon=max(self.min_epsilon,self.epsilon*self.epsilon_decay); chosen_actions={}
        for param,q_values in self.q_table[state].items():
            action_space=self.action_ranges[param]
            if random.random()<self.epsilon or not np.any(np.isfinite(q_values)): chosen_idx=random.randrange(len(action_space))
            else: finite_q_values=q_values[np.isfinite(q_values)]; best_indices = np.where(np.isfinite(q_values))[0][np.isclose(finite_q_values, np.max(finite_q_values))]; chosen_idx=np.random.choice(best_indices) if len(best_indices)>0 else random.randrange(len(action_space))
            chosen_actions[param]=float(action_space[chosen_idx])
        self.prev_actions_log.append(chosen_actions.copy()); return chosen_actions
    def update(self,state: Optional[Tuple],action: Optional[Dict[str, float]],reward: float,next_state: Optional[Tuple]):
        if state is None or next_state is None or action is None: logger.debug("QUpdate skip: Invalid state/action."); return
        if state not in self.q_table: logger.warning(f"QCtrl Warning: State {state} not found in update."); return
        if next_state not in self.q_table: self.q_table[next_state]={p:np.zeros(self.num_actions[p],dtype=np.float32) for p in self.action_ranges.keys()}; self.q_table_creation_time[next_state]=time.time(); self.q_table_access_count[next_state]=0; self._manage_q_table_size()
        for param,chosen_value in action.items():
            if param not in self.q_table[state]: logger.warning(f"QCtrl Warning: Param {param} not in Q-table {state}. Skip update."); continue
            action_indices=np.where(np.isclose(self.action_ranges[param],chosen_value))[0]
            if len(action_indices)==0: logger.warning(f"QCtrl Warning: Cannot find action index {param}={chosen_value}. Skip update."); continue
            action_idx=action_indices[0]; current_q=self.q_table[state][param][action_idx]
            finite_next_q=self.q_table[next_state][param][np.isfinite(self.q_table[next_state][param])]; max_future_q=np.max(finite_next_q) if len(finite_next_q)>0 else 0.0
            if not np.isfinite(max_future_q): max_future_q=0.0
            td_target=reward+self.gamma*max_future_q; td_error=td_target-current_q
            new_q=current_q+min(0.5,max(0.001,self.alpha*(1.0+self.flow_coefficient*np.tanh(abs(td_error)*0.5))))*td_error
            self.q_table[state][param][action_idx]=np.clip(new_q,-1e4,1e4) if np.isfinite(new_q) else 0.0
    def _manage_q_table_size(self):
        if len(self.q_table)>self.max_q_table_size:
            num_to_remove=len(self.q_table)-self.max_q_table_size; logger.info(f"QTable prune: size {len(self.q_table)} > max {self.max_q_table_size}. Removing {num_to_remove}.")
            try:
                if not self.q_table_access_count or not self.q_table_creation_time or len(self.q_table_access_count)<len(self.q_table)//2 or len(self.q_table_creation_time)<len(self.q_table)//2:
                    states_to_remove = random.sample(list(self.q_table.keys()), min(num_to_remove, len(self.q_table.keys())))
                else: states_to_remove=sorted(self.q_table.keys(),key=lambda s:(self.q_table_access_count.get(s,0),self.q_table_creation_time.get(s,float('inf'))))[:num_to_remove]
                for state_to_remove in states_to_remove: self.q_table.pop(state_to_remove, None); self.q_table_access_count.pop(state_to_remove, None); self.q_table_creation_time.pop(state_to_remove, None)
                logger.info(f"QTable pruned {len(states_to_remove)} states. New size: {len(self.q_table)}")
            except Exception as e:
                logger.warning(f"QTable prune error: {e}. Fallback random removal.")
                current_keys=list(self.q_table.keys()); num_to_remove_fallback=max(0,len(current_keys)-self.max_q_table_size)
                if num_to_remove_fallback>0:
                    for state_to_remove_fb in random.sample(current_keys,min(num_to_remove_fallback,len(current_keys))): self.q_table.pop(state_to_remove_fb,None); self.q_table_access_count.pop(state_to_remove_fb,None); self.q_table_creation_time.pop(state_to_remove_fb,None)
                    logger.info(f"QTable fallback pruned. New size: {len(self.q_table)}")
    def get_info(self) -> Dict:
        q_mem = sum(sys.getsizeof(k) + sys.getsizeof(v_dict) + sum(sys.getsizeof(pk) + (arr.nbytes if isinstance(arr, np.ndarray) else sys.getsizeof(arr)) for pk, arr in v_dict.items()) for k, v_dict in self.q_table.items()) / (1024**2) if self.q_table else 0
        return {"epsilon":self.epsilon,"stable_steps":self.stable_steps,"oscillation_counter":self.oscillation_counter,"q_table_size":len(self.q_table),"q_table_mem_mb_approx":round(q_mem,2),"last_action":self.prev_actions_log[-1] if self.prev_actions_log else None,f"avg_reward_last_{self.performance_window.maxlen}":np.mean(list(self.performance_window)) if self.performance_window else 0.0}

# =====================================================================
# Hyperbolic Geometry Implementation
# =====================================================================
class Manifold:
    def __init__(self): pass
    def dist(self, x, y, keepdim=False): raise NotImplementedError
    def sqdist(self, x, y, keepdim=False): raise NotImplementedError
    def egrad2rgrad(self, p, dp): raise NotImplementedError
    def proj(self, p, dp): raise NotImplementedError
    def proju(self, p): raise NotImplementedError
    def expmap(self, p, dp): raise NotImplementedError
    def logmap(self, p, y): raise NotImplementedError
    def expmap0(self, dp): raise NotImplementedError
    def logmap0(self, p): raise NotImplementedError
    def mobius_add(self, x, y): raise NotImplementedError
    def mobius_matvec(self, m, x): raise NotImplementedError
    def init_weights(self, w, irange=1e-5): raise NotImplementedError
    def zero_grad(self, p): p.grad.data.zero_()
    def normalize(self, p): return self.proju(p)
    def check_point_on_manifold(self, p, atol=1e-5): raise NotImplementedError
    def check_vector_on_tangent(self, p, dp, atol=1e-5): raise NotImplementedError
    @property
    def name(self): return self.__class__.__name__

class PoincareBall(Manifold):
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c.item() if isinstance(c, torch.Tensor) else float(c)
        
        if self.c <= 0:
            logger.warning(f"PoincareBall init c={self.c:.3g} <= 0. Ops may act Euclidean.")
            self.k = 0.0
            self.sqrt_c = 0.0  # Ensure sqrt_c is always defined
            self.radius = float('inf')
        else:
            self.k = -self.c
            self.sqrt_c = math.sqrt(self.c)
            self.radius = 1.0 / self.sqrt_c
            
        self.max_norm = (self.radius * (1.0 - EPS)) if self.c > 0 else float('inf')
        self.min_norm = EPS
        self._name = f'PoincareBall(c={self.c:.3g})'
    @property
    def name(self): return self._name
    def _check_c(self, require_positive=True): assert not require_positive or self.c > 0, f"{self.name}: Op requires c>0 (is {self.c})."
    def lambda_x(self, x, keepdim=False):
        if self.c <= 0: return torch.ones_like(x[..., 0:1]) * 2.0
        return 2. / torch.clamp(1. - self.c * torch.sum(x.float().pow(2), dim=-1, keepdim=True), min=EPS).to(x.dtype)
    def sqdist(self, x, y, keepdim=False):
        if self.c <= 0: return torch.clamp(torch.sum((x - y).pow(2), dim=-1, keepdim=keepdim), min=0.0)
        compute_dtype = torch.float32 if x.dtype != torch.float64 else torch.float64
        with torch.enable_grad() if x.requires_grad or y.requires_grad else torch.no_grad():
            x_p, y_p = self.proju(x).to(compute_dtype), self.proju(y).to(compute_dtype)
            diff_norm_sq = torch.sum((x_p - y_p).pow(2), dim=-1, keepdim=True)
            denom_x = torch.clamp(1. - self.c * torch.sum(x_p.pow(2), dim=-1, keepdim=True).clamp(min=0), min=EPS)
            denom_y = torch.clamp(1. - self.c * torch.sum(y_p.pow(2), dim=-1, keepdim=True).clamp(min=0), min=EPS)
            acosh_arg = torch.clamp(1. + (2. * self.c * diff_norm_sq) / (denom_x * denom_y + EPS), min=1.0 + EPS)
            sq_dist_val = (1.0 / self.c) * torch.acosh(acosh_arg).pow(2)
        return sq_dist_val.squeeze(-1) if not keepdim else sq_dist_val.to(x.dtype)
    def dist(self, x, y, keepdim=False):
        if self.c <= 0: return torch.clamp(torch.norm(x - y, p=2, dim=-1, keepdim=keepdim), min=0.0)
        sq_dist = self.sqdist(x, y, keepdim=True); dist_val = torch.sqrt(sq_dist + EPS).clamp(min=0.0)
        return dist_val.squeeze(-1) if not keepdim else dist_val
    def proju(self, x):
        if self.c <= 0: return x
        if not torch.is_tensor(x): logger.warning(f"proju received non-tensor type {type(x)}."); return x
        safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) if not torch.isfinite(x).all() else x
        scale = torch.clamp_max(self.max_norm / (torch.norm(safe_x, p=2, dim=-1, keepdim=True) + EPS), 1.0)
        projected_x = safe_x * scale
        return torch.nan_to_num(projected_x, nan=0.0) if not torch.isfinite(projected_x).all() else projected_x
    def proj(self, p, dp): return dp
    def expmap(self, p, dp):
        if self.c <= 0: return p + dp
        p_proj, dp_f = self.proju(p), dp.to(torch.float32 if p.dtype != torch.float64 else torch.float64)
        dp_norm_f = torch.norm(dp_f, p=2, dim=-1, keepdim=True).clamp(min=EPS)
        if torch.all(dp_norm_f < EPS): return p_proj
        lambda_p_f = self.lambda_x(p_proj.to(dp_f.dtype), keepdim=True)
        factor = torch.tanh(torch.clamp(self.sqrt_c * lambda_p_f * dp_norm_f / 2., -15., 15.)) / (self.sqrt_c * dp_norm_f + EPS)
        return self.proju(self.mobius_add(p_proj, factor * dp).to(p.dtype))
    def logmap(self, p, y):
        if self.c <= 0: return y - p
        p_proj, y_proj = self.proju(p), self.proju(y)
        if torch.allclose(p_proj, y_proj, atol=EPS): return torch.zeros_like(p_proj)
        sub_f = self.mobius_add(-p_proj, y_proj).to(torch.float32 if p.dtype != torch.float64 else torch.float64)
        sub_norm_f = torch.norm(sub_f, p=2, dim=-1, keepdim=True).clamp(min=EPS)
        lambda_p_f = self.lambda_x(p_proj.to(sub_f.dtype), keepdim=True)
        factor = (2. / (self.sqrt_c * lambda_p_f + EPS)) * torch.atanh(torch.clamp(self.sqrt_c * sub_norm_f, -1.+EPS, 1.-EPS)) / (sub_norm_f + EPS)
        return (factor * sub_f).to(p.dtype)
    def expmap0(self, dp):
        if self.c <= 0: return dp
        dp_f = dp.to(torch.float32 if dp.dtype != torch.float64 else torch.float64)
        dp_norm_f = torch.norm(dp_f, p=2, dim=-1, keepdim=True).clamp(min=EPS)
        if torch.all(dp_norm_f < EPS): return torch.zeros_like(dp)
        factor = torch.tanh(torch.clamp(self.sqrt_c * dp_norm_f, -15., 15.)) / (self.sqrt_c * dp_norm_f + EPS)
        return self.proju((factor * dp_f).to(dp.dtype))
    def logmap0(self, p):
        if self.c <= 0: return p
        p_proj = self.proju(p)
        p_f = p_proj.to(torch.float32 if p.dtype != torch.float64 else torch.float64)
        p_norm_f = torch.norm(p_f, p=2, dim=-1, keepdim=True).clamp(min=EPS)
        if torch.all(p_norm_f < EPS): return torch.zeros_like(p_proj)
        factor = torch.atanh(torch.clamp(self.sqrt_c * p_norm_f, -1.+EPS, 1.-EPS)) / (self.sqrt_c * p_norm_f + EPS)
        return (factor * p_f).to(p.dtype)
    def mobius_add(self, x, y):
        if self.c <= 0: return x + y
        with torch.enable_grad() if x.requires_grad or y.requires_grad else torch.no_grad():
            x_p, y_p = self.proju(x).to(torch.float32 if x.dtype != torch.float64 else torch.float64), self.proju(y).to(torch.float32 if y.dtype != torch.float64 else torch.float64)
            x_norm_sq, y_norm_sq = torch.sum(x_p.pow(2), dim=-1, keepdim=True).clamp(min=0), torch.sum(y_p.pow(2), dim=-1, keepdim=True).clamp(min=0)
            xy_dot = torch.sum(x_p * y_p, dim=-1, keepdim=True)
            num = (1. + 2. * self.c * xy_dot + self.c * y_norm_sq) * x_p + torch.clamp(1. - self.c * x_norm_sq, min=EPS) * y_p
            den = torch.clamp(1. + 2. * self.c * xy_dot + self.c**2 * x_norm_sq * y_norm_sq, min=EPS)
            return self.proju((num / den).to(x.dtype))
    def mobius_scalar_mul(self, r: Union[float, torch.Tensor], x):
        if self.c <= 0: return r * x
        x_p = self.proju(x).to(torch.float32 if x.dtype != torch.float64 else torch.float64)
        x_norm_f = torch.norm(x_p, p=2, dim=-1, keepdim=True).clamp(min=EPS)
        if torch.all(x_norm_f < EPS): return torch.zeros_like(x)
        r_f = r.float() if isinstance(r, torch.Tensor) else float(r)
        tanh_term = torch.tanh(torch.clamp(r_f * torch.atanh(torch.clamp(self.sqrt_c * x_norm_f, -1.+EPS, 1.-EPS)), -15., 15.))
        return self.proju(((tanh_term / (self.sqrt_c + EPS)) * (x_p / (x_norm_f + EPS))).to(x.dtype))
    def mobius_matvec(self, M: nn.Parameter, x):
        if self.c <= 0: return F.linear(x, M)
        x_p = self.proju(x)
        if torch.allclose(x_p, torch.zeros_like(x_p), atol=EPS): return torch.zeros_like(x_p)
        return self.proju(self.expmap0(F.linear(self.logmap0(x_p), M)))
    def egrad2rgrad(self, p, dp):
        if self.c <= 0: return dp
        p_proj = self.proju(p)
        lambda_p_sq = self.lambda_x(p_proj.to(torch.float32 if p.dtype != torch.float64 else torch.float64), keepdim=True).pow(2)
        rgrad = (dp / torch.clamp(lambda_p_sq, min=EPS)).to(p.dtype)
        return torch.nan_to_num(rgrad, nan=0.0, posinf=0.0, neginf=0.0) if not torch.isfinite(rgrad).all() else rgrad
    def init_weights(self, w: nn.Parameter, irange=1e-5):
        if not hasattr(w,'manifold') or w.manifold!=self: logger.warning("init_weights on param not assigned to this manifold.")
        with torch.no_grad(): w.data.uniform_(-irange, irange); w.data = self.proju(w.data); w.manifold = self
    def check_point_on_manifold(self, p, atol=1e-5):
        if self.c <= 0: return torch.ones_like(p[...,0],dtype=torch.bool)
        on_manifold = torch.sum(p.pow(2), dim=-1) <= (self.radius**2 + atol)
        if not torch.all(on_manifold): logger.debug(f"Point check fail: Max norm {torch.sqrt(torch.sum(p.pow(2),dim=-1).max()).item():.4f} vs radius {self.radius:.4f}")
        return on_manifold
    def check_vector_on_tangent(self, p, dp, atol=1e-5): return p.shape == dp.shape
def get_manifold(name="poincare", curvature=1.0) -> Manifold: return PoincareBall(c=curvature) if name.lower() == "poincare" else (_ for _ in ()).throw(ValueError(f"Unknown manifold: {name}"))

# =====================================================================
# PyTorch Spatial Layers (Reimplementation)
# =====================================================================
class SpatialFeedForwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, activation="gelu", dropout=0.1):
        super().__init__()
        self.output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "gelu":
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.dropout_act = nn.Dropout(dropout) # Dropout after activation
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        if self.fc1.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.fc1.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        if self.fc2.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.fc2.bias, -bound, bound)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout_act(x)
        x = self.fc2(x)
        return x

class SpatialAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, geom_scale=1.0, feature_scale=1.0, causal=False, learn_scales=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if learn_scales:
            self.geom_scale = nn.Parameter(torch.tensor(float(geom_scale)))
            self.feature_scale = nn.Parameter(torch.tensor(float(feature_scale)))
        else:
            self.register_buffer('geom_scale', torch.tensor(float(geom_scale)))
            self.register_buffer('feature_scale', torch.tensor(float(feature_scale)))
        self._reset_parameters()

    def _reset_parameters(self):
        gain = 1.0
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None: nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None: nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None: nn.init.zeros_(self.v_proj.bias)
        if self.out_proj.bias is not None: nn.init.zeros_(self.out_proj.bias)

    def forward(self, coords: torch.Tensor, features: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        B, S, D = features.shape
        S_coords = coords.shape[1]
        assert S == S_coords, f"Feature sequence length {S} and coordinate sequence length {S_coords} must match."
        assert coords.shape[2] == 3, "Coordinates must be 3D."

        q = self.q_proj(features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q_scaled = q * (self.head_dim ** -0.5 * self.feature_scale)
        scores_feat = torch.matmul(q_scaled, k.transpose(-2, -1))

        coords_q_expanded = coords.unsqueeze(2)
        coords_k_expanded = coords.unsqueeze(1)
        delta_coords = coords_q_expanded - coords_k_expanded
        dist_sq = torch.sum(delta_coords.pow(2), dim=-1)
        scores_geom = -self.geom_scale * dist_sq
        scores_geom_expanded = scores_geom.unsqueeze(1)

        scores = scores_feat + scores_geom_expanded

        if self.causal:
            causal_mask_val = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(causal_mask_val.unsqueeze(0).unsqueeze(0), float('-inf'))

        if key_padding_mask is not None:
            scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float('-inf'))

        attn_probs = F.softmax(scores, dim=-1)
        attn_output_raw = torch.matmul(attn_probs, v)
        attn_output_reshaped = attn_output_raw.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        output = self.out_proj(attn_output_reshaped)
        output = self.dropout(output)
        return output

class WuBuSpatialEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_hidden_mult=4, dropout=0.1, activation="gelu", norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = SpatialAttentionLayer(hidden_size, num_heads, dropout=dropout, causal=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout_attn_res = nn.Dropout(dropout)
        ffn_output_dim = hidden_size
        self.ffn = SpatialFeedForwardLayer(hidden_size, hidden_size * ffn_hidden_mult, ffn_output_dim, activation=activation, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout_ffn_res = nn.Dropout(dropout)

    def forward(self, coords, features, src_key_padding_mask=None):
        x = features
        if self.norm_first:
            attn_output = self.self_attn(coords, self.norm1(x), key_padding_mask=src_key_padding_mask)
            x = x + self.dropout_attn_res(attn_output)
            ffn_output = self.ffn(self.norm2(x))
            x = x + self.dropout_ffn_res(ffn_output)
        else:
            attn_output = self.self_attn(coords, x, key_padding_mask=src_key_padding_mask)
            x = self.norm1(x + self.dropout_attn_res(attn_output))
            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout_ffn_res(ffn_output))
        return x

class WuBuSpatialDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_hidden_mult=4, dropout=0.1, activation="gelu", norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = SpatialAttentionLayer(hidden_size, num_heads, dropout=dropout, causal=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout_sa_res = nn.Dropout(dropout)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout_ca_res = nn.Dropout(dropout)
        ffn_output_dim = hidden_size
        self.ffn = SpatialFeedForwardLayer(hidden_size, hidden_size * ffn_hidden_mult, ffn_output_dim, activation=activation, dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout_ffn_res = nn.Dropout(dropout)

    def forward(self, tgt_coords, tgt_features, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt_features
        if self.norm_first:
            sa_out = self.self_attn(tgt_coords, self.norm1(x), key_padding_mask=tgt_key_padding_mask)
            x = x + self.dropout_sa_res(sa_out)
            ca_out, _ = self.cross_attn(query=self.norm2(x), key=memory, value=memory, key_padding_mask=memory_key_padding_mask, need_weights=False)
            x = x + self.dropout_ca_res(ca_out)
            ffn_out = self.ffn(self.norm3(x))
            x = x + self.dropout_ffn_res(ffn_out)
        else:
            sa_out = self.self_attn(tgt_coords, x, key_padding_mask=tgt_key_padding_mask)
            x = self.norm1(x + self.dropout_sa_res(sa_out))
            ca_out, _ = self.cross_attn(query=x, key=memory, value=memory, key_padding_mask=memory_key_padding_mask, need_weights=False)
            x = self.norm2(x + self.dropout_ca_res(ca_out))
            ffn_out = self.ffn(x)
            x = self.norm3(x + self.dropout_ffn_res(ffn_out))
        return x

# =====================================================================
# Helper Function for Weight Initialization
# =====================================================================
def init_weights(module):
    if isinstance(module, (SpatialAttentionLayer, SpatialFeedForwardLayer)):
        return # Let them handle their own reset_parameters

    if isinstance(module, nn.Linear):
        is_gyro_linear = 'GyroLinear' in str(type(module))
        is_gyro_bias = hasattr(module, 'bias') and module.bias is not None and hasattr(module.bias, 'manifold') and isinstance(getattr(module.bias, 'manifold', None), Manifold)
        if not is_gyro_linear:
            torch.nn.init.xavier_uniform_(module.weight, gain=math.sqrt(2.0))
            if module.bias is not None and not is_gyro_bias:
                torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
         is_hyperbolic_embedding = 'HyperbolicEmbedding' in str(type(module)) or (hasattr(module.weight, 'manifold') and isinstance(getattr(module.weight, 'manifold', None), Manifold))
         if not is_hyperbolic_embedding:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        is_riemannian_ln = 'RiemannianLayerNorm' in str(type(module)) or (hasattr(module, 'manifold') and isinstance(getattr(module, 'manifold', None), Manifold))
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
        # logger.debug(f"HypEmb init: {num_embeddings}x{embedding_dim} on {manifold.name}")

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
        # logger.debug(f"GyroLinear init: {in_features}->{out_features}, bias={self.use_bias}, manifold={manifold.name}")

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
        # logger.debug(f"RiemLayerNorm init: shape={self.normalized_shape}, affine={elementwise_affine}, manifold={manifold.name}")

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

# =====================================================================
# WuBu Nesting Core
# =====================================================================
class BoundaryManifoldHyperbolic(nn.Module):
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
            # logger.info(f"BoundaryManifoldHyp L{level_idx}: {num_points} pts {point_dim}D ({initial_manifold.name}).")
        else:
            self.register_parameter('hyperbolic_points',None)
            # logger.info(f"BoundaryManifoldHyp L{level_idx}: No boundary points.")

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
            # logger.info(f"HypInterLevelTr ({in_dim}->{out_dim}, {manifold_in.name}->{manifold_out.name}): MLP({mlp_hidden_dim}) T_0")
        elif self.transform_type=='linear':
            self.tangent_transform=nn.Linear(in_dim,out_dim)
            # logger.info(f"HypInterLevelTr ({in_dim}->{out_dim}, {manifold_in.name}->{manifold_out.name}): Linear T_0")
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
                unconstrained_val = math.log(math.expm1(y)) if y >= 1e-6 else math.log(y + EPS)
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
        # logger.info(f"Level {level_idx} TgtCombiner: Linear {comb_in_dim}->{self.dim}")
        self.use_flow=config.get("use_tangent_flow",True)
        self.tangent_flow=None
        self.flow_scale=0.
        if self.use_flow:
            flow_h_dim=max(16,int(dim*config.get("tangent_flow_hidden_dim_ratio",0.5)))
            flow_type=config.get("tangent_flow_type","mlp").lower()
            if flow_type=='mlp':
                self.tangent_flow=nn.Sequential(nn.Linear(dim,flow_h_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(flow_h_dim,dim))
                # logger.info(f"Level {level_idx} TangentFlow: MLP({flow_h_dim})")
            elif flow_type=='linear':
                self.tangent_flow=nn.Linear(dim,dim)
                # logger.info(f"Level {level_idx} TangentFlow: Linear")
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
        if point_in.dim() == 3: B, S, Din = point_in.shape
        elif point_in.dim() == 2: B, Din = point_in.shape; S = 1
        else: raise ValueError(f"Level {self.level_idx}: Invalid input point dimensions: {point_in.shape}")
        dev = point_in.device
        dtype = next(iter(self.parameters())).dtype if len(list(self.parameters())) > 0 else torch.float32
        assert Din==self.dim, f"Level {self.level_idx}: Input dimension {Din} != level dimension {self.dim}"
        cur_c,cur_s,cur_spread=self.get_curvature().to(dev),self.get_scale().to(dev),self.get_spread().to(dev)
        cur_m=PoincareBall(c=cur_c)
        if self.level_descriptor_param is not None: self.level_descriptor_param.manifold=cur_m
        self.boundary_manifold.set_current_manifold(cur_m)
        point_in_proj=cur_m.proju(point_in); tan_main=cur_m.logmap0(point_in_proj)
        tan_rel=relative_vectors_tangent_in if relative_vectors_tangent_in is not None else torch.zeros_like(tan_main)
        tan_ld_comb = torch.zeros_like(tan_main); ld_point_self = None
        if self.use_ld and self.level_descriptor_param is not None:
            ld_point_self = cur_m.proju(self.level_descriptor_param); tan_ld_self = cur_m.logmap0(ld_point_self)
            if tan_ld_self.dim() == 1 and tan_main.dim() == 3: tan_ld_comb = tan_ld_self.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
            elif tan_ld_self.dim() == 1 and tan_main.dim() == 2: tan_ld_comb = tan_ld_self.unsqueeze(0).expand(tan_main.shape[0], -1)
            else: tan_ld_comb = tan_ld_self
        if descriptor_point_in is not None: tan_ld_comb=tan_ld_comb+cur_m.logmap0(cur_m.proju(descriptor_point_in))
        inputs_comb=[tan_main.to(dtype)];
        if self.relative_vector_aggregation not in ['none', None]: inputs_comb.append(tan_rel.to(dtype))
        if self.use_ld: inputs_comb.append(tan_ld_comb.to(dtype))
        assert inputs_comb, f"Level {self.level_idx}: No inputs to tangent combiner!"
        comb_input=torch.cat(inputs_comb,dim=-1)
        exp_comb_dim=self.tangent_combiner[0].in_features
        assert comb_input.shape[-1]==exp_comb_dim, f"Level {self.level_idx} Combiner input dim mismatch: expected {exp_comb_dim}, got {comb_input.shape[-1]}"
        v_comb_tan=self.tangent_combiner(comb_input)
        if self.use_flow and self.tangent_flow is not None: v_comb_tan=v_comb_tan+self.tangent_flow(v_comb_tan)*self.flow_scale
        scaled_tan_out = v_comb_tan * cur_s; point_out = cur_m.expmap0(scaled_tan_out); tan_out = cur_m.logmap0(point_out)
        bound_pts_out = self.boundary_manifold.get_points(); ld_out = None
        if ld_point_self is not None:
            if ld_point_self.dim() == 1 and point_out.dim() == 3: ld_out = ld_point_self.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
            elif ld_point_self.dim() == 1 and point_out.dim() == 2: ld_out = ld_point_self.unsqueeze(0).expand(point_out.shape[0], -1)
            else: ld_out = ld_point_self
        sigma_out = cur_spread; final_outputs=[]
        out_vars=[point_out,tan_out,ld_out,bound_pts_out,sigma_out]; names=["point_out","tan_out","ld_point","boundaries","sigma"]
        for name,out_tensor in zip(names,out_vars):
            if out_tensor is not None:
                if not torch.isfinite(out_tensor).all(): logger.warning(f"NaN/Inf in Level {self.level_idx} output '{name}'. Replacing."); out_tensor=torch.nan_to_num(out_tensor,nan=0., posinf=0., neginf=0.)
                final_outputs.append(out_tensor.to(dtype))
            else: final_outputs.append(None)
        while len(final_outputs) < 5: final_outputs.append(None)
        pt_ret = final_outputs[0] if final_outputs[0] is not None else torch.zeros((B,S,self.dim), device=dev, dtype=dtype)
        tan_ret = final_outputs[1] if final_outputs[1] is not None else torch.zeros((B,S,self.dim), device=dev, dtype=dtype)
        ld_ret = final_outputs[2]; bnd_ret = final_outputs[3]
        sig_ret = final_outputs[4] if final_outputs[4] is not None else torch.tensor(self.min_spread, device=dev, dtype=dtype)
        return pt_ret, tan_ret, ld_ret, bnd_ret, sig_ret

class FullyHyperbolicWuBuNestingModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: Dict):
        super().__init__()
        self.input_dim,self.output_dim,self.config=input_dim,output_dim,config
        self.num_levels=config.get("num_levels",3); assert self.num_levels>0
        self.hyperbolic_dims=config.get("hyperbolic_dims",[128]*self.num_levels); self.initial_curvatures=config.get("initial_curvatures",[1.0]*self.num_levels)
        self.dropout,self.relative_vector_aggregation,self.aggregation_method=config.get("dropout",0.1),config.get("relative_vector_aggregation","mean"),config.get("aggregation_method","concat_tangent")
        list_args={'hyperbolic_dims':self.num_levels,'initial_curvatures':self.num_levels,'initial_scales':self.num_levels,'boundary_points_per_level':self.num_levels,'initial_spread_values':self.num_levels}
        num_trans=max(0,self.num_levels-1); trans_list_args={'transform_types':num_trans,'transform_hidden_dims':num_trans}
        for k,L in list_args.items():
            if k not in config or len(config[k])!=L: raise ValueError(f"Config '{k}' missing or needs length {L}. Got: {config.get(k)}")
        for k,L in trans_list_args.items():
            if k not in config or len(config[k])!=L: raise ValueError(f"Config '{k}' missing or needs length {L}. Got: {config.get(k)}")
        self.input_tangent_to_H0_tangent=nn.Linear(input_dim,self.hyperbolic_dims[0]); self.levels=nn.ModuleList()
        for i in range(self.num_levels): self.levels.append(HyperbolicWuBuNestingLevel(level_idx=i, dim=self.hyperbolic_dims[i], config=self.config, initial_curvature=self.initial_curvatures[i]))
        self.transforms=nn.ModuleList(); trans_types,trans_hdims=config.get("transform_types",[]),config.get("transform_hidden_dims",[])
        for i in range(num_trans):
            m_in,m_out=PoincareBall(c=self.initial_curvatures[i]),PoincareBall(c=self.initial_curvatures[i+1])
            self.transforms.append(HyperbolicInterLevelTransform(in_dim=self.hyperbolic_dims[i], out_dim=self.hyperbolic_dims[i+1], manifold_in=m_in, manifold_out=m_out, transform_type=trans_types[i], hidden_dim=trans_hdims[i], dropout=self.dropout))
        assert self.aggregation_method=="concat_tangent", f"Aggregation method '{self.aggregation_method}' not supported."
        self.tangent_to_output=nn.Linear(sum(self.hyperbolic_dims),output_dim); self.input_tangent_to_H0_tangent.apply(init_weights); self.tangent_to_output.apply(init_weights)
        total_p,train_p=sum(p.numel() for p in self.parameters()),sum(p.numel() for p in self.parameters() if p.requires_grad)
        # logger.info(f"FullyHypWuBuModel init: {self.num_levels} levels. Arch: InTgt({input_dim}) -> H0Tgt({self.hyperbolic_dims[0]}) | Levels Dims:{self.hyperbolic_dims} | Agg:{self.aggregation_method} -> OutTgt({output_dim}). Params:{total_p:,} (Trainable:{train_p:,})")

    def forward(self, x_tangent_in: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_tangent_in.dim() == 2: x_tangent_in = x_tangent_in.unsqueeze(0)
        if padding_mask is not None and padding_mask.dim()==1: padding_mask = padding_mask.unsqueeze(0)
        B,S,Din=x_tangent_in.shape; assert Din == self.input_dim
        dev,dtype=x_tangent_in.device,next(iter(self.parameters())).dtype if len(list(self.parameters())) > 0 else torch.float32
        cur_tan = self.input_tangent_to_H0_tangent(x_tangent_in); m0=PoincareBall(c=self.levels[0].get_curvature().to(dev)); cur_pt = m0.expmap0(cur_tan)
        level_tan_outs, agg_rel_vecs_tan, cur_desc_pt, cur_sigma = [], None, None, None
        for i in range(self.num_levels):
            level = self.levels[i]; cur_m = PoincareBall(c=level.get_curvature().to(dev))
            cur_pt_proj = cur_m.proju(cur_pt); desc_pt_proj = cur_m.proju(cur_desc_pt) if cur_desc_pt is not None else None
            pt_out, tan_out, ld_pt_out, bound_pts, sigma_out = level(point_in=cur_pt_proj, relative_vectors_tangent_in=agg_rel_vecs_tan, descriptor_point_in=desc_pt_proj, sigma_in=cur_sigma)
            level_tan_outs.append(tan_out)
            if i < self.num_levels - 1:
                trans = self.transforms[i]; m_next = PoincareBall(c=self.levels[i+1].get_curvature().to(dev))
                pt_next, bound_trans, ld_next = trans(point_in=pt_out, boundaries_in=bound_pts, descriptor_in=ld_pt_out, manifold_in_current=cur_m, manifold_out_current=m_next)
                agg_rel_vecs_tan = None; has_bounds = bound_trans is not None and bound_trans.numel() > 0
                if has_bounds and self.relative_vector_aggregation not in ['none', None]:
                    pt_next_proj,bound_trans_proj = m_next.proju(pt_next),m_next.proju(bound_trans)
                    tan_main_next,tan_bounds_next = m_next.logmap0(pt_next_proj),m_next.logmap0(bound_trans_proj)
                    if tan_bounds_next.dim() == 2: tan_bounds_next = tan_bounds_next.unsqueeze(0).unsqueeze(0)
                    elif tan_bounds_next.dim() != 4: logger.error(f"Unexpected boundary tangent dim {tan_bounds_next.dim()}."); has_bounds=False
                    if has_bounds:
                        if tan_main_next.dim() == 2: tan_main_next = tan_main_next.unsqueeze(1) if tan_main_next.shape[0] == B else tan_main_next.unsqueeze(0).unsqueeze(0)
                        elif tan_main_next.dim() != 3: logger.error(f"Unexpected pt_next tangent dim {tan_main_next.dim()}."); has_bounds=False
                    if has_bounds:
                        try:
                            rel_tan_origin = tan_main_next.unsqueeze(2) - tan_bounds_next
                            agg_m = self.relative_vector_aggregation
                            if agg_m=="mean": agg_rel_vecs_tan = torch.mean(rel_tan_origin, dim=2)
                            elif agg_m=="sum": agg_rel_vecs_tan = torch.sum(rel_tan_origin, dim=2)
                            else: logger.warning(f"Unsupported relative aggregation '{agg_m}'."); agg_rel_vecs_tan = None
                        except Exception as rel_vec_err: logger.error(f"Error calculating relative vectors L{i+1}: {rel_vec_err}", exc_info=True); agg_rel_vecs_tan = None
                        if agg_rel_vecs_tan is not None and not torch.isfinite(agg_rel_vecs_tan).all():
                            logger.warning(f"NaN/Inf in L{i+1} aggregated relative vectors. Replacing."); agg_rel_vecs_tan = torch.zeros_like(tan_main_next)
                cur_pt, cur_desc_pt, cur_sigma = pt_next, ld_next, sigma_out
        try:
            compat_tans=[]; expected_agg_dim = sum(self.hyperbolic_dims)
            for t_idx, t in enumerate(level_tan_outs):
                level_dim = self.hyperbolic_dims[t_idx]
                if t is None: logger.error(f"Tangent output L{t_idx} is None. Zeroing."); t = torch.zeros((B, S, level_dim), device=dev, dtype=dtype)
                elif not torch.isfinite(t).all(): logger.warning(f"NaN/Inf in Tangent L{t_idx}. Zeroing."); t = torch.nan_to_num(t, nan=0., posinf=0., neginf=0.)
                if t.shape[-1] != level_dim: logger.error(f"Tangent L{t_idx} dim mismatch: Exp {level_dim}, Got {t.shape[-1]}. Reshaping."); t = F.pad(t, (0, level_dim - t.shape[-1])) if t.shape[-1] < level_dim else t[..., :level_dim]
                compat_tans.append(t.to(dtype))
            agg_tan = torch.cat(compat_tans, dim=-1); assert agg_tan.shape[-1] == expected_agg_dim
        except Exception as cat_err: logger.error(f"Error in tangent aggregation: {cat_err}. Shapes: {[t.shape if t is not None else 'None' for t in level_tan_outs]}", exc_info=True); return torch.zeros((B, S, self.output_dim), device=dev, dtype=dtype)
        final_out_tan = self.tangent_to_output(agg_tan)
        if padding_mask is not None:
            mask = padding_mask.to(dtype=torch.bool, device=dev).unsqueeze(0) if padding_mask.dim() == 1 else padding_mask.to(dtype=torch.bool, device=dev)
            assert mask.shape == (B, S); final_out_tan = final_out_tan.masked_fill(mask.unsqueeze(-1), 0.)
        if not torch.isfinite(final_out_tan).all(): logger.error(f"NaN/Inf in final WuBu output. Zeroing."); final_out_tan = torch.nan_to_num(final_out_tan, nan=0., posinf=0., neginf=0.)
        return final_out_tan

# =====================================================================
# Sequence Model Components (Hybrid Spatial Adaptation)
# =====================================================================
class WuBuLocalEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, manifold: PoincareBall, dropout: float=0.1, max_seq_len: int=1024, feature_config: Dict = None):
        super().__init__()
        self.hidden_size = hidden_size
        assert isinstance(manifold, PoincareBall)
        self.manifold = manifold
        self.max_seq_len = max_seq_len
        self.feature_config = feature_config or {}
        self.nucleotide_embedding = HyperbolicEmbedding(self.feature_config.get('nucleotide_vocab_size', NUCLEOTIDE_VOCAB_SIZE), hidden_size, manifold)
        self.structure_embedding = nn.Embedding(self.feature_config.get('structure_vocab_size', STRUCTURE_VOCAB_SIZE), hidden_size)
        self.region_embedding = nn.Embedding(self.feature_config.get('region_vocab_size', REGION_VOCAB_SIZE), hidden_size)
        self.structure_source_embedding = nn.Embedding(self.feature_config.get('structure_source_vocab_size', STRUCTURE_SOURCE_VOCAB_SIZE), hidden_size)
        self.dataset_source_embedding = nn.Embedding(self.feature_config.get('dataset_source_vocab_size', DATASET_SOURCE_VOCAB_SIZE), hidden_size)
        self.codon_embedding = nn.Embedding(self.feature_config.get('codon_vocab_size', CODON_VOCAB_SIZE), hidden_size)
        self.positional_encoding = nn.Embedding(max_seq_len, hidden_size)
        self.feature_combiner_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layers = nn.ModuleList([WuBuSpatialEncoderLayer(hidden_size, num_heads, dropout=dropout, norm_first=True) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout_embed = nn.Dropout(dropout)
        self.apply(init_weights)
        logger.info(f"WuBuLocalEncoder (PyTorch Spatial) init: Hidden={hidden_size}, Layers={num_layers}, Heads={num_heads}, MaxLen={max_seq_len}")

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        nucleotides, coords, structure_sym, region_types = batch_data['context_nucleotides'], batch_data['context_coords'], batch_data['context_structure_symbols'], batch_data['context_region_types']
        struct_source, dataset_source, codons = batch_data.get('context_structure_sources'), batch_data.get('context_dataset_sources'), batch_data.get('context_codons')
        B, S = nucleotides.shape; dev, dtype = nucleotides.device, self.positional_encoding.weight.dtype
        if S > self.max_seq_len:
            logger.warning(f"Encoder input len {S} > max {self.max_seq_len}. Truncating.")
            nucleotides, coords, structure_sym, region_types = nucleotides[:,:S], coords[:,:S,:], structure_sym[:,:S], region_types[:,:S]
            if struct_source is not None: struct_source = struct_source[:,:S]
            if dataset_source is not None: dataset_source = dataset_source[:,:S]
            if codons is not None: codons = codons[:,:S]
            S = self.max_seq_len
        padding_mask = (nucleotides == NUCLEOTIDE_MAP.get('N', 4)) if torch.any(nucleotides == NUCLEOTIDE_MAP.get('N', 4)) else None
        nuc_tan = self.manifold.logmap0(self.nucleotide_embedding(nucleotides).to(dtype))
        struct_emb, region_emb = self.structure_embedding(structure_sym).to(dtype), self.region_embedding(region_types).to(dtype)
        struct_source_emb = self.structure_source_embedding(struct_source).to(dtype) if struct_source is not None else 0
        dataset_source_emb = self.dataset_source_embedding(dataset_source).to(dtype) if dataset_source is not None else 0
        codon_emb = self.codon_embedding(codons).to(dtype) if codons is not None else 0
        pos_emb = self.positional_encoding(torch.arange(0, S, device=dev).unsqueeze(0).expand(B, -1)).to(dtype)
        combined_features = self.feature_combiner_norm(nuc_tan + pos_emb + struct_emb + region_emb + struct_source_emb + dataset_source_emb + codon_emb)
        combined_features = self.dropout_embed(combined_features)
        features_out = combined_features
        for layer in self.layers: features_out = layer(coords=coords, features=features_out, src_key_padding_mask=padding_mask)
        final_tangent_features = self.final_norm(features_out)
        if not torch.isfinite(final_tangent_features).all(): final_tangent_features = torch.nan_to_num(final_tangent_features, nan=0., posinf=0., neginf=0.); logger.warning("NaN/Inf in Spatial Encoder output. Replaced.")
        if padding_mask is not None: final_tangent_features = final_tangent_features.masked_fill(padding_mask.unsqueeze(-1), 0.)
        return final_tangent_features

class WuBuLocalDecoder(nn.Module):
    def __init__(self, hidden_size: int, global_tangent_dim: int, num_layers: int, num_heads: int, manifold: PoincareBall, dropout: float=0.1, max_decode_len: int=2048, feature_config: Dict = None):
        super().__init__()
        self.hidden_size, self.global_tangent_dim, self.manifold, self.max_decode_len, self.feature_config = hidden_size, global_tangent_dim, manifold, max_decode_len, feature_config or {}
        assert isinstance(manifold, PoincareBall)
        self.nucleotide_embedding = HyperbolicEmbedding(self.feature_config.get('nucleotide_vocab_size', NUCLEOTIDE_VOCAB_SIZE), hidden_size, manifold)
        self.structure_embedding = nn.Embedding(self.feature_config.get('structure_vocab_size', STRUCTURE_VOCAB_SIZE), hidden_size)
        self.region_embedding = nn.Embedding(self.feature_config.get('region_vocab_size', REGION_VOCAB_SIZE), hidden_size)
        self.structure_source_embedding = nn.Embedding(self.feature_config.get('structure_source_vocab_size', STRUCTURE_SOURCE_VOCAB_SIZE), hidden_size)
        self.dataset_source_embedding = nn.Embedding(self.feature_config.get('dataset_source_vocab_size', DATASET_SOURCE_VOCAB_SIZE), hidden_size)
        self.codon_embedding = nn.Embedding(self.feature_config.get('codon_vocab_size', CODON_VOCAB_SIZE), hidden_size)
        self.positional_encoding = nn.Embedding(max_decode_len, hidden_size)
        self.feature_combiner_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.memory_projection = nn.Sequential(nn.Linear(global_tangent_dim, hidden_size * 2), nn.GELU(), nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size, eps=1e-6))
        self.layers = nn.ModuleList([WuBuSpatialDecoderLayer(hidden_size, num_heads, dropout=dropout, norm_first=True) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.nucleotide_pred = nn.Linear(hidden_size, self.feature_config.get('nucleotide_vocab_size', NUCLEOTIDE_VOCAB_SIZE))
        self.structure_pred = nn.Linear(hidden_size, self.feature_config.get('structure_vocab_size', STRUCTURE_VOCAB_SIZE))
        self.region_pred = nn.Linear(hidden_size, self.feature_config.get('region_vocab_size', REGION_VOCAB_SIZE))
        self.dropout_embed = nn.Dropout(dropout)
        self.apply(init_weights)
        logger.info(f"WuBuLocalDecoder (PyTorch Spatial) init: Hidden={hidden_size}, MemDim={global_tangent_dim}, Layers={num_layers}, Heads={num_heads}, MaxLen={max_decode_len}")

    def forward(self, batch_data_target: Dict[str, torch.Tensor], memory_tangent: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        tgt_nucleotides, tgt_coords, tgt_structure_sym, tgt_region_types = batch_data_target['target_nucleotides'], batch_data_target['target_coords'], batch_data_target['target_structure_symbols'], batch_data_target['target_region_types']
        tgt_struct_source, tgt_dataset_source, tgt_codons = batch_data_target.get('target_structure_sources'), batch_data_target.get('target_dataset_sources'), batch_data_target.get('target_codons')
        B, T = tgt_nucleotides.shape; dev, dtype = tgt_nucleotides.device, self.positional_encoding.weight.dtype
        if T > self.max_decode_len:
            logger.warning(f"Decoder target len {T} > max {self.max_decode_len}. Truncating.")
            tgt_nucleotides,tgt_coords,tgt_structure_sym,tgt_region_types=tgt_nucleotides[:,:T],tgt_coords[:,:T,:],tgt_structure_sym[:,:T],tgt_region_types[:,:T]
            if tgt_struct_source is not None: tgt_struct_source = tgt_struct_source[:,:T]
            if tgt_dataset_source is not None: tgt_dataset_source = tgt_dataset_source[:,:T]
            if tgt_codons is not None: tgt_codons = tgt_codons[:,:T]
            T = self.max_decode_len
        tgt_key_padding_mask = (tgt_nucleotides == NUCLEOTIDE_MAP.get('N', 4)) if torch.any(tgt_nucleotides == NUCLEOTIDE_MAP.get('N', 4)) else None
        tgt_nuc_tan = self.manifold.logmap0(self.nucleotide_embedding(tgt_nucleotides).to(dtype))
        tgt_struct_emb, tgt_region_emb = self.structure_embedding(tgt_structure_sym).to(dtype), self.region_embedding(tgt_region_types).to(dtype)
        tgt_struct_source_emb = self.structure_source_embedding(tgt_struct_source).to(dtype) if tgt_struct_source is not None else 0
        tgt_dataset_source_emb = self.dataset_source_embedding(tgt_dataset_source).to(dtype) if tgt_dataset_source is not None else 0
        tgt_codon_emb = self.codon_embedding(tgt_codons).to(dtype) if tgt_codons is not None else 0
        tgt_pos_emb = self.positional_encoding(torch.arange(0, T, device=dev).unsqueeze(0).expand(B, -1)).to(dtype)
        tgt_combined = self.feature_combiner_norm(tgt_nuc_tan + tgt_pos_emb + tgt_struct_emb + tgt_region_emb + tgt_struct_source_emb + tgt_dataset_source_emb + tgt_codon_emb)
        tgt_combined = self.dropout_embed(tgt_combined)
        proj_mem_tan = self.memory_projection(memory_tangent.to(dtype))
        if memory_key_padding_mask is not None and proj_mem_tan.shape[1] > 0 and memory_key_padding_mask.shape[1] != proj_mem_tan.shape[1]: memory_key_padding_mask=None; logger.warning("Memory padding mask size mismatch. Ignoring.")
        decoder_output = tgt_combined
        for layer in self.layers: decoder_output = layer(tgt_coords=tgt_coords, tgt_features=decoder_output, memory=proj_mem_tan, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        final_features = self.final_norm(decoder_output)
        output_dict = {"nucleotide_logits": self.nucleotide_pred(final_features).float(), "structure_logits": self.structure_pred(final_features).float(), "region_logits": self.region_pred(final_features).float()}
        for name, logits in output_dict.items():
            if not torch.isfinite(logits).all(): output_dict[name] = torch.nan_to_num(logits, nan=0., posinf=0., neginf=0.); logger.warning(f"NaN/Inf in final decoder logits '{name}'. Replaced.")
        return output_dict

class WuBuNestingSequenceModel(nn.Module):
    def __init__(self, wubu_config: Dict, sequence_config: Dict, feature_config: Dict):
        super().__init__()
        self.wubu_config, self.sequence_config, self.feature_config = wubu_config, sequence_config, feature_config
        self.local_hidden_size = sequence_config["local_hidden_size"]
        self.decoder_memory_dim = sum(wubu_config["hyperbolic_dims"])
        if sequence_config["decoder_memory_dim"] != self.decoder_memory_dim: logger.warning(f"Seq config decoder_memory_dim != sum WuBu dims. Using {self.decoder_memory_dim}."); self.sequence_config["decoder_memory_dim"] = self.decoder_memory_dim
        self.context_window, self.encoder_max_seq_len, self.decoder_max_seq_len = sequence_config["context_window"], sequence_config.get("encoder_max_seq_len",1024), sequence_config.get("decoder_max_seq_len",2048)
        first_lvl_c = wubu_config["initial_curvatures"][0] if wubu_config["initial_curvatures"] else 1.0
        self.shared_manifold=PoincareBall(c=first_lvl_c); logger.info(f"WuBuModel Shared Manifold (Embeddings): {self.shared_manifold.name}")
        self.local_encoder=WuBuLocalEncoder(hidden_size=self.local_hidden_size, num_layers=sequence_config["num_encoder_layers"], num_heads=sequence_config["num_encoder_heads"], manifold=self.shared_manifold, dropout=wubu_config.get("dropout",0.1), max_seq_len=self.encoder_max_seq_len, feature_config=self.feature_config)
        self.wubu_model=FullyHyperbolicWuBuNestingModel(input_dim=self.local_hidden_size, output_dim=self.decoder_memory_dim, config=self.wubu_config)
        self.local_decoder=WuBuLocalDecoder(hidden_size=self.local_hidden_size, global_tangent_dim=self.decoder_memory_dim, num_layers=sequence_config["num_decoder_layers"], num_heads=sequence_config["num_decoder_heads"], manifold=self.shared_manifold, dropout=wubu_config.get("dropout",0.1), max_decode_len=self.decoder_max_seq_len, feature_config=self.feature_config)
        logger.info("WuBuNestingSequenceModel (PyTorch Spatial) Initialized.")

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dev, dtype = batch_data['context_nucleotides'].device, self.local_encoder.positional_encoding.weight.dtype
        dummy_logits = lambda: {"nucleotide_logits": torch.zeros((batch_data['context_nucleotides'].shape[0], batch_data['target_nucleotides'].shape[1], self.feature_config['nucleotide_vocab_size']), device=dev, dtype=torch.float32)}
        try:
            pad_idx = NUCLEOTIDE_MAP.get('N', 4)
            input_padding_mask = (batch_data['context_nucleotides'] == pad_idx) if torch.any(batch_data['context_nucleotides'] == pad_idx) else None
            enc_tan=self.local_encoder(batch_data)
            if not torch.isfinite(enc_tan).all(): logger.error("Encoder output NaN/Inf. Zeroing."); enc_tan=torch.nan_to_num(enc_tan,nan=0., posinf=0., neginf=0.)
        except Exception as enc_err: logger.error(f"Error in Spatial Encoder: {enc_err}",exc_info=True); return dummy_logits()
        try:
            dec_mem_tan=self.wubu_model(x_tangent_in=enc_tan,padding_mask=input_padding_mask)
            if not torch.isfinite(dec_mem_tan).all(): logger.error("WuBu Core output NaN/Inf. Zeroing."); dec_mem_tan=torch.nan_to_num(dec_mem_tan,nan=0., posinf=0., neginf=0.)
        except Exception as wubu_err: logger.error(f"Error in WuBu Core Model: {wubu_err}",exc_info=True); return dummy_logits()
        try:
            output_dict=self.local_decoder(batch_data_target=batch_data, memory_tangent=dec_mem_tan, memory_key_padding_mask=input_padding_mask)
        except Exception as dec_err: logger.error(f"Error in Spatial Decoder: {dec_err}",exc_info=True); return dummy_logits()
        return output_dict

    @staticmethod
    def compute_loss(model_output: Dict[str, torch.Tensor], batch_data: Dict[str, torch.Tensor], feature_config: Dict, loss_weights: Dict = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_weights = loss_weights or {'nucleotide': 1.0, 'structure': 0.2, 'region': 0.1, 'predicted_structure_weight': 0.5}
        alpha, beta, w_pred = loss_weights.get('structure', 0.2), loss_weights.get('region', 0.1), loss_weights.get('predicted_structure_weight', 0.5)
        nuc_logits, tgt_nucleotides = model_output['nucleotide_logits'], batch_data['target_nucleotides']
        B, S, _ = nuc_logits.shape; pad_idx = NUCLEOTIDE_MAP.get('N', 4)
        tgt_padding_mask = (tgt_nucleotides == pad_idx) if torch.any(tgt_nucleotides == pad_idx) else None
        loss_nuc = torch.tensor(0.0, device=nuc_logits.device, dtype=torch.float32)
        if S > 1:
            logits_shift, targets_shift = nuc_logits[:, :-1, :].contiguous().view(-1, feature_config['nucleotide_vocab_size']), tgt_nucleotides[:, 1:].contiguous().view(-1)
            targets_shift = torch.clamp(targets_shift, 0, feature_config['nucleotide_vocab_size'] - 1)
            loss_nuc_full = F.cross_entropy(logits_shift.float(), targets_shift.long(), label_smoothing=0.1, reduction='none')
            if tgt_padding_mask is not None: mask_shift = tgt_padding_mask[:, 1:].contiguous().view(-1); mask_bool = ~mask_shift.bool(); loss_nuc_full = loss_nuc_full.masked_fill(~mask_bool, 0.); loss_nuc = loss_nuc_full.sum() / mask_bool.sum().clamp(min=1.)
            else: loss_nuc = loss_nuc_full.mean()
        loss_nuc = loss_nuc.to(nuc_logits.dtype)
        loss_struct = torch.tensor(0.0, device=nuc_logits.device, dtype=nuc_logits.dtype)
        if alpha > 0 and 'structure_logits' in model_output and 'target_structure_symbols' in batch_data:
            struct_logits, tgt_struct, struct_sources = model_output['structure_logits'], batch_data['target_structure_symbols'], batch_data.get('target_structure_sources')
            if struct_logits.shape[1] > 1:
                struct_logits_shift, tgt_struct_shift = struct_logits[:, :-1, :].contiguous().view(-1, feature_config['structure_vocab_size']), tgt_struct[:, 1:].contiguous().view(-1)
                tgt_struct_shift = torch.clamp(tgt_struct_shift, 0, feature_config['structure_vocab_size'] - 1)
                loss_struct_full = F.cross_entropy(struct_logits_shift.float(), tgt_struct_shift.long(), reduction='none')
                if struct_sources is not None:
                    struct_sources_shift = struct_sources[:, 1:].contiguous().view(-1)
                    source_weights = torch.where((struct_sources_shift == STRUCTURE_SOURCE_MAP['PredictedMFE']), torch.tensor(w_pred, device=struct_sources_shift.device), torch.ones_like(struct_sources_shift, dtype=torch.float32))
                    loss_struct_full *= source_weights
                if tgt_padding_mask is not None: mask_shift = tgt_padding_mask[:, 1:].contiguous().view(-1); mask_bool = ~mask_shift.bool(); loss_struct_full = loss_struct_full.masked_fill(~mask_bool, 0.); loss_struct = loss_struct_full.sum() / mask_bool.sum().clamp(min=1.)
                else: loss_struct = loss_struct_full.mean()
            loss_struct = loss_struct.to(nuc_logits.dtype)
        loss_region = torch.tensor(0.0, device=nuc_logits.device, dtype=nuc_logits.dtype)
        if beta > 0 and 'region_logits' in model_output and 'target_region_types' in batch_data:
            region_logits, tgt_region = model_output['region_logits'], batch_data['target_region_types']
            if region_logits.shape[1] > 1:
                region_logits_shift, tgt_region_shift = region_logits[:, :-1, :].contiguous().view(-1, feature_config['region_vocab_size']), tgt_region[:, 1:].contiguous().view(-1)
                tgt_region_shift = torch.clamp(tgt_region_shift, 0, feature_config['region_vocab_size'] - 1)
                loss_region_full = F.cross_entropy(region_logits_shift.float(), tgt_region_shift.long(), reduction='none')
                if tgt_padding_mask is not None: mask_shift = tgt_padding_mask[:, 1:].contiguous().view(-1); mask_bool = ~mask_shift.bool(); loss_region_full = loss_region_full.masked_fill(~mask_bool, 0.); loss_region = loss_region_full.sum() / mask_bool.sum().clamp(min=1.)
                else: loss_region = loss_region_full.mean()
            loss_region = loss_region.to(nuc_logits.dtype)
        total_loss = loss_nuc + alpha * loss_struct + beta * loss_region
        if not torch.isfinite(total_loss): logger.error(f"NaN/Inf in total loss ({total_loss}). Nuc={loss_nuc}, Struct={loss_struct}, Region={loss_region}. High loss returned."); total_loss = torch.tensor(100.0, device=total_loss.device, dtype=total_loss.dtype)
        loss_dict = {'total_loss': total_loss.item(), 'nucleotide_loss': loss_nuc.item() if torch.isfinite(loss_nuc) else float('inf'), 'structure_loss': loss_struct.item() if torch.isfinite(loss_struct) else float('inf'), 'region_loss': loss_region.item() if torch.isfinite(loss_region) else float('inf')}
        return total_loss, loss_dict

    @torch.no_grad()
    def generate(self, *args, **kwargs): logger.warning("Generation not fully implemented."); return torch.tensor([[]])

# =====================================================================
# Riemannian Optimizer
# =====================================================================
class RiemannianEnhancedSGD(torch.optim.Optimizer):
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
                        continue

                    if not torch.isfinite(r_grad).all():
                        logger.error(f"Non-finite RGrad {p.shape} on {manifold.name}. Skip.")
                        self.gradient_stats.non_finite_grads_in_step += 1
                        if 'momentum_buffer' in state:
                            del state['momentum_buffer']
                        continue

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
                        continue
                else:
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
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, train_loader: DataLoader, val_loader: Optional[DataLoader], grad_accum_steps: int = 1, use_amp: bool = True, log_interval: int = 10, save_interval: int = 1000, checkpoint_dir: str = "checkpoints", wandb_enabled: bool = False, max_grad_norm: float = 1.0, rank: int = 0, world_size: int = 1, detect_anomaly: bool = False, feature_config: Dict = None, loss_weights: Dict = None):
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
        self.is_main = self.rank == 0
        self.wandb_enabled = wandb_enabled and WANDB_AVAILABLE and self.is_main
        self.max_grad_norm=max_grad_norm
        self.global_step=0
        self.current_epoch=0
        self.last_val_metrics:Optional[Dict[str,float]]=None
        self.detect_anomaly=detect_anomaly
        self.feature_config=feature_config or {}
        self.loss_weights=loss_weights or {}
        self.has_grad_stats = hasattr(self.optimizer, 'gradient_stats') and isinstance(getattr(self.optimizer, 'gradient_stats', None), GradientStats)
        self.has_q_controller = hasattr(self.optimizer, 'q_controller') and isinstance(getattr(self.optimizer, 'q_controller', None), HAKMEMQController)
        self.wandb_run=wandb.run if self.wandb_enabled and wandb is not None else None
        logger.info(f"Trainer(Hybrid) Rank {rank}/{world_size}: Device={device}, AMP={self.use_amp}, Accum={self.grad_accum_steps}, MaxNorm={self.max_grad_norm}, Anomaly={self.detect_anomaly}, QCtrl={self.has_q_controller}")
        if self.is_main:
            os.makedirs(self.checkpoint_dir,exist_ok=True)

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        moved_batch = {}
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                try:
                    moved_batch[key] = tensor.to(self.device, non_blocking=True)
                except Exception as move_err:
                    logger.error(f"Failed to move tensor '{key}' to {self.device}: {move_err}. Skipping.")
                    return None
            else:
                logger.warning(f"Non-tensor item '{key}' in batch dict. Keeping as is.")
                moved_batch[key] = tensor
        return moved_batch

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
        total_loss_cycle_dict = defaultdict(float)
        optim_steps_epoch=0
        micro_steps_cycle=0
        micro_batches_processed = 0
        approx_optim_steps=None
        approx_micro_batches=None
        try:
            dset_len=0
            sampler=getattr(self.train_loader, 'sampler', None)
            dset=getattr(self.train_loader, 'dataset', None)
            if hasattr(dset,'__len__') and len(dset) > 0:
                L = len(dset)
                if isinstance(sampler, DistributedSampler) and isinstance(dset, IterableDataset):
                     dset_len_per_worker = L
                     approx_micro_batches = dset_len_per_worker
                     approx_optim_steps = max(1, approx_micro_batches // self.grad_accum_steps)
                else:
                     dset_len = max(1, L // self.world_size) if self.world_size > 1 else L
                     approx_micro_batches = dset_len
                     approx_optim_steps = max(1, approx_micro_batches // self.grad_accum_steps)
            else:
                logger.info(f"Rank {self.rank}: Cannot accurately estimate epoch length (IterableDataset without reliable len?).")
                approx_micro_batches = None
                approx_optim_steps = None
        except Exception as e:
            logger.warning(f"Could not estimate epoch length: {e}")
            approx_micro_batches = None
            approx_optim_steps = None

        disable_tqdm = not self.is_main
        batch_iter=tqdm(self.train_loader, desc=f"Ep {self.current_epoch+1}|Opt 0/?", disable=disable_tqdm, total=approx_micro_batches, unit="batch", dynamic_ncols=True, leave=False)
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch_data_cpu in enumerate(batch_iter):
            micro_batches_processed += 1
            micro_steps_cycle += 1
            is_last_micro_step = (micro_steps_cycle % self.grad_accum_steps == 0)
            is_last_batch_epoch = (approx_micro_batches is not None and micro_batches_processed >= approx_micro_batches)
            should_optim_step = is_last_micro_step or is_last_batch_epoch
            sync_ctx=contextlib.nullcontext()
            anomaly_ctx=contextlib.nullcontext()
            if self.world_size > 1 and isinstance(self.model, DDP) and not should_optim_step:
                sync_ctx = self.model.no_sync()
            if self.detect_anomaly:
                anomaly_ctx = torch.autograd.detect_anomaly(check_nan=True)

            loss = None
            current_loss_dict = {}
            try:
                batch_data = self._move_batch_to_device(batch_data_cpu)
                if batch_data is None or 'context_nucleotides' not in batch_data or batch_data['context_nucleotides'].numel() == 0:
                    logger.warning(f"Rank {self.rank}: Skipping invalid/empty batch at micro-step {micro_steps_cycle}.")
                    if should_optim_step:
                        logger.warning(f"Rank {self.rank}: Skipping optim step G{self.global_step} due to bad batch before step.")
                        self.optimizer.zero_grad(set_to_none=True)
                        total_loss_cycle_dict.clear()
                        micro_steps_cycle = 0
                        should_optim_step = False
                    continue

                with sync_ctx, anomaly_ctx:
                    amp_dtype = torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
                    with amp.autocast(device_type=self.device.type,dtype=amp_dtype,enabled=self.use_amp):
                        model_output = self.model(batch_data)
                        if model_output is None or 'nucleotide_logits' not in model_output:
                             raise RuntimeError("Model forward returned None or invalid output dict (missing 'nucleotide_logits')")
                        loss, current_loss_dict = WuBuNestingSequenceModel.compute_loss(model_output, batch_data, self.feature_config, self.loss_weights)
                    if loss is None or not torch.isfinite(loss): raise ValueError(f"Non-finite or None loss computed: {loss}")
                    loss_scaled = loss / self.grad_accum_steps
                self.scaler.scale(loss_scaled).backward()
                if 'total_loss' in current_loss_dict and np.isfinite(current_loss_dict['total_loss']):
                    for k, v in current_loss_dict.items():
                        if np.isfinite(v): total_loss_cycle_dict[k] += v
                else:
                    logger.warning(f"Rank {self.rank}: Non-finite total loss ({current_loss_dict.get('total_loss')}) at GStep {self.global_step}, MStep {micro_steps_cycle}. Not accumulating stats.")
            except Exception as batch_ex:
                logger.error(f"MicroStep Error G{self.global_step} M{micro_steps_cycle} R{self.rank}: {batch_ex}",exc_info=True)
                total_loss_cycle_dict.clear()
                micro_steps_cycle = 0
                should_optim_step = False
                try: self.optimizer.zero_grad(set_to_none=True); logger.warning(f"Rank {self.rank}: Zeroed gradients after micro-step error.")
                except Exception as zge: logger.error(f"Error zeroing gradients after micro-step error: {zge}")
                if 'CUDA out of memory' in str(batch_ex) and torch.cuda.is_available(): logger.warning("Clearing CUDA cache after OOM."); torch.cuda.empty_cache()
                continue

            if should_optim_step:
                avg_loss_cycle = total_loss_cycle_dict['total_loss'] / micro_steps_cycle if micro_steps_cycle > 0 else 0.
                avg_loss_cycle_dict = {k: v / micro_steps_cycle for k, v in total_loss_cycle_dict.items()} if micro_steps_cycle > 0 else {}
                optim_skipped,unclipped_norm,is_clipped,clip_ratio,grad_norm_error = False,0.,False,None,False
                try: self.scaler.unscale_(self.optimizer)
                except Exception as unscale_error: logger.error(f"AMP Unscale Error G{self.global_step} R{self.rank}: {unscale_error}. Skipping optim step."); optim_skipped = True
                if not optim_skipped and self.max_grad_norm > 0:
                    params_with_grad = [p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
                    if params_with_grad:
                        try:
                            total_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=float('inf'), norm_type=2.0)
                            unclipped_norm = total_norm.item()
                            if not np.isfinite(unclipped_norm): logger.warning(f"Rank {self.rank}: Non-finite grad norm ({unclipped_norm}) BEFORE clip. Skipping optim."); optim_skipped = True; grad_norm_error = True
                            elif unclipped_norm > self.max_grad_norm: torch.nn.utils.clip_grad_norm_(params_with_grad, self.max_grad_norm, norm_type=2.0); is_clipped = True; clip_ratio = self.max_grad_norm / (unclipped_norm + EPS)
                        except Exception as norm_error: logger.error(f"Grad Norm/Clip Error G{self.global_step} R{self.rank}: {norm_error}. Skipping optim."); optim_skipped = True; grad_norm_error = True; unclipped_norm = float('inf')
                    else: unclipped_norm = 0.0
                elif not optim_skipped:
                    params_with_grad = [p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
                    if params_with_grad:
                        try: unclipped_norm = torch.norm(torch.stack([torch.norm(p.grad.detach().float(), 2.0) for p in params_with_grad]), 2.0).item()
                        except Exception as norm_calc_err: logger.error(f"Error calculating grad norm (no clip): {norm_calc_err}"); unclipped_norm = float('inf'); grad_norm_error = True
                        if not np.isfinite(unclipped_norm): grad_norm_error = True; logger.warning(f"Rank {self.rank}: Non-finite grad norm ({unclipped_norm}) calculated (no clip).")
                    else: unclipped_norm = 0.0
                if self.has_grad_stats:
                    if grad_norm_error: self.optimizer.gradient_stats.non_finite_grads_in_step += 1
                    self.optimizer.gradient_stats.record_gradient(unclipped_norm, is_clipped, clip_ratio)
                if not optim_skipped and self.has_q_controller:
                    try:
                        q_ctrl = self.optimizer.q_controller
                        self.optimizer.set_current_loss(avg_loss_cycle if np.isfinite(avg_loss_cycle) else None)
                        grp0 = self.optimizer.param_groups[0]; cur_lr, cur_mom = grp0['lr'], grp0.get('momentum', 0.)
                        q_grad_norm = unclipped_norm if np.isfinite(unclipped_norm) else 100.0
                        q_state = q_ctrl.get_state(lr=cur_lr, momentum=cur_mom, grad_norm=q_grad_norm, loss=(avg_loss_cycle if np.isfinite(avg_loss_cycle) else None))
                        if q_ctrl.prev_state is not None and q_ctrl.prev_action is not None and q_state is not None and q_ctrl.prev_loss is not None:
                           reward = q_ctrl.compute_reward(avg_loss_cycle if np.isfinite(avg_loss_cycle) else None, q_ctrl.prev_loss, q_grad_norm)
                           if np.isfinite(reward): q_ctrl.update(q_ctrl.prev_state, q_ctrl.prev_action, reward, q_state)
                           else: logger.warning(f"QCtrl non-finite reward ({reward}). Skip Q update.")
                        q_ctrl.prev_loss = avg_loss_cycle if np.isfinite(avg_loss_cycle) else q_ctrl.prev_loss; q_ctrl.prev_state = q_state
                        q_ctrl.prev_action = q_ctrl.choose_action(q_state) if q_state is not None else {'lr_scale': 1.0, 'momentum_scale': 1.0}
                        if q_ctrl.prev_action:
                             for group in self.optimizer.param_groups:
                                base_lr, base_mom = group.get('base_lr', group['lr']), group.get('base_momentum', group['momentum'])
                                group['lr'] = float(np.clip(base_lr * q_ctrl.prev_action.get('lr_scale', 1.0), 1e-9, 1.0))
                                group['momentum'] = float(np.clip(base_mom * q_ctrl.prev_action.get('momentum_scale', 1.0), 0.1, 0.9999))
                    except Exception as q_err: logger.error(f"Q-Ctrl Update/Action Error G{self.global_step} R{self.rank}: {q_err}",exc_info=False);
                    if self.has_q_controller: q_ctrl = self.optimizer.q_controller; q_ctrl.prev_state, q_ctrl.prev_action, q_ctrl.prev_loss = None,None,None
                if not optim_skipped: self.scaler.step(self.optimizer); self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                grad_stats = self.optimizer.gradient_stats.record_step(self.global_step, skipped=optim_skipped) if self.has_grad_stats else {}
                if not optim_skipped:
                    optim_steps_epoch += 1; self.global_step += 1
                    if self.is_main:
                        opt_step_str=f"{optim_steps_epoch}/{(approx_optim_steps or '?')}"
                        batch_iter.set_description(f"Ep {self.current_epoch+1}|Opt {opt_step_str}")
                        cur_lr = self.optimizer.param_groups[0]['lr']
                        applied_norm = min(unclipped_norm, self.max_grad_norm) if is_clipped and np.isfinite(unclipped_norm) else (unclipped_norm if np.isfinite(unclipped_norm) else -1.0)
                        batch_iter.set_postfix(Loss=f"{avg_loss_cycle:.3f}", LR=f"{cur_lr:.3e}", Grad=f"{applied_norm:.2f}", Scale=f"{self.scaler.get_scale():.0f}", refresh=False)
                        if self.global_step % self.log_interval == 0:
                            cur_mom=self.optimizer.param_groups[0].get('momentum',0.); q_info = self.optimizer.get_q_info() if self.has_q_controller else {}; hyp_stats = self._get_hyperbolic_stats()
                            log_data = {"Epoch": self.current_epoch + 1, "Step": self.global_step, "Train Loss (Total)": avg_loss_cycle, "LR": cur_lr, "Momentum": cur_mom, "Grad Norm (Applied)": applied_norm, "Grad Norm (Unclipped Max)": grad_stats.get('max_gradient', -1.0), "Clip %": grad_stats.get('clip_percentage', 0.), "NonFinite Grads (Step)": grad_stats.get('non_finite_grads', 0), "Optim Skipped": grad_stats.get('step_skipped', False), "AMP Scale": self.scaler.get_scale()}
                            log_data.update({f"Train Loss ({k.replace('_loss','').capitalize()})": v for k, v in avg_loss_cycle_dict.items() if k != 'total_loss'})
                            log_data.update({f"Hyp/{k}":v for k,v in hyp_stats.items()}); log_data.update({f"QCtrl/{k}":v for k,v in q_info.items() if k!='last_action'})
                            if q_info.get('last_action'): log_data["QCtrl/LR_Scale"]=q_info['last_action'].get('lr_scale',1.); log_data["QCtrl/Mom_Scale"]=q_info['last_action'].get('momentum_scale',1.)
                            log_parts=[f"S{self.global_step}",f"Ep{self.current_epoch+1} Opt{optim_steps_epoch}",f"Loss {log_data['Train Loss (Total)']:.4f}",f"LR {log_data['LR']:.3e}",f"Grad {log_data['Grad Norm (Applied)']:.2f}",f"Scale {log_data['AMP Scale']:.0f}"]
                            if hyp_stats: log_parts.append(f"Crv[0] {hyp_stats.get('L0_Curv',0):.3g}")
                            if self.has_q_controller and q_info.get('last_action'): log_parts.append(f"QScale(L/M) {log_data.get('QCtrl/LR_Scale',1.):.2f}/{log_data.get('QCtrl/Mom_Scale',1.):.2f}")
                            if grad_stats.get('clip_percentage',0.) > 1.: log_parts.append(f"Clip% {log_data['Clip %']:.1f}")
                            if grad_stats.get('non_finite_grads',0) > 0: log_parts.append(f"NFGrads {log_data['NonFinite Grads (Step)']}")
                            if grad_stats.get('step_skipped', False): log_parts.append("SKIPPED")
                            logger.info(" | ".join(log_parts))
                            if self.wandb_enabled and self.wandb_run:
                                try:
                                    wandb_log_data = {f"train/{k.replace(' ', '_')}": v for k, v in log_data.items() if not k.startswith(('Hyp/','QCtrl/'))}
                                    wandb_log_data.update({f"train_hyp/{k.replace('Hyp/','')}": v for k, v in log_data.items() if k.startswith('Hyp/')})
                                    wandb_log_data.update({f"train_qctrl/{k.replace('QCtrl/','')}": v for k,v in log_data.items() if k.startswith('QCtrl/')})
                                    for loss_name, loss_val in avg_loss_cycle_dict.items():
                                         if loss_name != 'total_loss': wandb_log_data[f"train_loss/{loss_name.replace('_loss','')}"] = loss_val
                                    wandb.log(wandb_log_data, step=self.global_step)
                                except Exception as wbe: logger.error(f"Wandb train log failed: {wbe}")
                        if self.is_main and self.save_interval > 0 and self.global_step % self.save_interval == 0: self._save_checkpoint(is_intermediate=True, metrics=avg_loss_cycle_dict)
                else:
                     if self.is_main: batch_iter.set_postfix(Loss="SKIP", LR=f"{self.optimizer.param_groups[0]['lr']:.3e}", Grad="SKIP", Scale=f"{self.scaler.get_scale():.0f}", refresh=False)
                     if self.is_main and self.global_step % self.log_interval == 0: logger.warning(f"S{self.global_step} | Ep{self.current_epoch+1} Opt{optim_steps_epoch} | OPTIM STEP SKIPPED (GradNormError: {grad_norm_error}, UnscaleError: {'unscale_error' in locals()})")
                total_loss_cycle_dict.clear(); micro_steps_cycle = 0
        if self.is_main and hasattr(batch_iter, 'close'): batch_iter.close()
        if self.world_size > 1: logger.debug(f"Rank {self.rank} end-of-train-epoch barrier."); torch.distributed.barrier(); logger.debug(f"Rank {self.rank} passed barrier.")

    @torch.no_grad()
    def _validate(self)->Dict[str,float]:
        if self.val_loader is None: return {}
        self.model.eval(); approx_val_batches=None
        try:
            val_dset_len=0; sampler=getattr(self.val_loader, 'sampler', None); dset=getattr(self.val_loader, 'dataset', None)
            if hasattr(dset,'__len__'): L=len(dset); val_dset_len = max(1, L // self.world_size) if self.world_size > 1 else L
            if val_dset_len > 0: approx_val_batches = val_dset_len
        except Exception as e: logger.warning(f"Could not estimate validation length: {e}")
        val_iter=tqdm(self.val_loader, desc=f"Validating Ep {self.current_epoch+1}", disable=not self.is_main, total=approx_val_batches, unit="batch", leave=False)
        epoch_loss_sums, epoch_loss_counts = defaultdict(float), defaultdict(int)
        for batch_data_cpu in val_iter:
            try:
                batch_data = self._move_batch_to_device(batch_data_cpu)
                if batch_data is None or 'context_nucleotides' not in batch_data or batch_data['context_nucleotides'].numel() == 0: continue
                amp_dtype=torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
                with torch.no_grad(), amp.autocast(device_type=self.device.type,dtype=amp_dtype,enabled=self.use_amp):
                    m_eval = self.model.module if isinstance(self.model, DDP) else self.model
                    model_output = m_eval(batch_data)
                    val_loss_weights = self.loss_weights.copy() # Use original weights for aux losses
                    loss, current_loss_dict = WuBuNestingSequenceModel.compute_loss(model_output, batch_data, self.feature_config, val_loss_weights)
                if loss is not None and torch.isfinite(loss):
                    for k, v in current_loss_dict.items():
                        if np.isfinite(v): epoch_loss_sums[k] += v; epoch_loss_counts[k] += 1
                else: logger.warning(f"Rank {self.rank}: Non-finite validation loss ({loss}).")
            except Exception as ve: logger.error(f"Rank {self.rank} Validation batch error: {ve}",exc_info=False); continue
        final_metrics = {}
        if self.world_size > 1:
            try:
                sorted_keys = sorted(epoch_loss_sums.keys())
                local_sums = torch.tensor([epoch_loss_sums.get(k, 0.0) for k in sorted_keys], dtype=torch.float64, device=self.device)
                local_counts = torch.tensor([epoch_loss_counts.get(k, 0) for k in sorted_keys], dtype=torch.float64, device=self.device)
                torch.distributed.all_reduce(local_sums, op=torch.distributed.ReduceOp.SUM); torch.distributed.all_reduce(local_counts, op=torch.distributed.ReduceOp.SUM)
                global_sums, global_counts = local_sums.cpu().numpy(), local_counts.cpu().numpy()
                if self.is_main:
                    global_loss_sums, global_loss_counts_dict = {k: s for k, s in zip(sorted_keys, global_sums)}, {k: int(c) for k, c in zip(sorted_keys, global_counts)}
                    for k in global_loss_sums: final_metrics[f"val_{k}"] = global_loss_sums[k] / global_loss_counts_dict[k] if global_loss_counts_dict[k] > 0 else float('inf')
            except Exception as gather_error: logger.error(f"Rank {self.rank}: Validation loss aggregation failed: {gather_error}. Reporting local metrics (if main).")
            if self.is_main:
                for k in epoch_loss_sums: final_metrics[f"val_{k}"] = epoch_loss_sums[k] / epoch_loss_counts[k] if epoch_loss_counts[k] > 0 else float('inf')
        else:
            for k in epoch_loss_sums: final_metrics[f"val_{k}"] = epoch_loss_sums[k] / epoch_loss_counts[k] if epoch_loss_counts[k] > 0 else float('inf')
        if self.is_main and 'val_total_loss' in final_metrics:
            avg_total_loss = final_metrics['val_total_loss']; ppl = float('inf')
            if np.isfinite(avg_total_loss):
                try: ppl=math.exp(min(avg_total_loss, 700))
                except OverflowError: logger.warning(f"PPL overflow (avg_loss={avg_total_loss}).")
                except Exception as ppl_err: logger.warning(f"Error calculating perplexity: {ppl_err}")
            final_metrics['val_perplexity'] = ppl; self.last_val_metrics=final_metrics
            log_parts = [f"Validation Ep {self.current_epoch+1}"] + [f"{k.replace('val_', '').capitalize()}: {v:.4f}" for k, v in final_metrics.items() if 'loss' in k]
            if 'val_perplexity' in final_metrics: log_parts.append(f"Perplexity: {final_metrics['val_perplexity']:.2f}")
            logger.info(" | ".join(log_parts))
            if self.wandb_enabled and self.wandb_run:
                try:
                    wblog={f"val/{k.replace('val_','')}":v for k,v in final_metrics.items()}; wblog["epoch"] = self.current_epoch+1
                    wblog.update({f"val_hyp/{k}":v for k,v in self._get_hyperbolic_stats().items()})
                    wandb.log(wblog, step=self.global_step)
                except Exception as wbe: logger.error(f"Wandb validation log failed: {wbe}")
        if hasattr(val_iter,'close'): val_iter.close()
        if self.world_size > 1: logger.debug(f"Rank {self.rank} end-of-validation barrier."); torch.distributed.barrier(); logger.debug(f"Rank {self.rank} passed barrier.")
        return final_metrics if self.is_main else {}

    def _save_checkpoint(self, is_intermediate: bool=False, metrics: Optional[Dict]=None):
        if not self.is_main or self.save_interval < 0: return
        state_indicator = f"step_{self.global_step}" if is_intermediate else f"epoch_{self.current_epoch+1}_final"
        metric_str = ""; current_metrics = metrics if metrics is not None else self.last_val_metrics
        primary_metric_key = 'val_total_loss' if current_metrics and 'val_total_loss' in current_metrics else 'total_loss'
        if current_metrics and primary_metric_key in current_metrics:
            metric_val = current_metrics[primary_metric_key]
            if metric_val is not None and np.isfinite(metric_val): metric_str = f"_{primary_metric_key.replace('_loss','')}{f'{metric_val:.2e}' if abs(metric_val) < 1e-3 and metric_val != 0 else f'{metric_val:.3f}'}"
        filename = f"checkpoint_{state_indicator}{metric_str}.pt"; filepath = os.path.join(self.checkpoint_dir, filename)
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        clean_state_dict = {k: v for k, v in model_to_save.state_dict().items() if not k.endswith('.manifold')}
        checkpoint_data = {'epoch': self.current_epoch, 'global_step': self.global_step, 'model_state_dict': clean_state_dict, 'optimizer_state_dict': self.optimizer.state_dict(), 'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None, 'metrics': current_metrics, 'amp_enabled': self.use_amp, 'args': getattr(self,'args',None), 'wubu_config': getattr(model_to_save,'wubu_config',{}), 'sequence_config': getattr(model_to_save,'sequence_config',{}), 'feature_config': getattr(model_to_save,'feature_config',{}), 'loss_weights': self.loss_weights}
        if self.has_q_controller and self.optimizer.q_controller:
            try:
                q_ctrl = self.optimizer.q_controller
                q_table_serializable = {str(state): {param: arr.tolist() if isinstance(arr, np.ndarray) else arr for param, arr in action_dict.items()} for state, action_dict in q_ctrl.q_table.items()}
                checkpoint_data['q_controller_state'] = {'q_table': q_table_serializable, 'epsilon': q_ctrl.epsilon, 'access_count': {str(k):v for k,v in q_ctrl.q_table_access_count.items()}, 'creation_time': {str(k):v for k,v in q_ctrl.q_table_creation_time.items()}, 'loss_window': list(q_ctrl.loss_window), 'grad_norm_window': list(q_ctrl.grad_norm_window), 'performance_window': list(q_ctrl.performance_window), 'stable_steps': q_ctrl.stable_steps, 'oscillation_counter': q_ctrl.oscillation_counter, 'prev_loss': q_ctrl.prev_loss, 'prev_state': str(q_ctrl.prev_state) if q_ctrl.prev_state else None, 'prev_action': q_ctrl.prev_action, 'action_ranges': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in q_ctrl.action_ranges.items()}, 'num_actions': q_ctrl.num_actions}
            except Exception as q_save_err: logger.error(f"Error preparing Q-Controller state for saving: {q_save_err}")
        temp_filepath = filepath + ".tmp." + str(random.randint(1000,9999))
        try: torch.save(checkpoint_data, temp_filepath); os.replace(temp_filepath, filepath); logger.info(f"Checkpoint saved: {filepath}")
        except Exception as save_error: logger.error(f"Failed to save checkpoint {filepath}: {save_error}", exc_info=True); _try_remove_file(temp_filepath)

    def load_checkpoint(self, filepath: str)->int:
        if not os.path.exists(filepath): logger.error(f"Checkpoint file not found: {filepath}"); return 0
        try:
            checkpoint_data = torch.load(filepath, map_location='cpu'); logger.info(f"Loading checkpoint: {os.path.basename(filepath)}")
            model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
            if checkpoint_data.get('wubu_config') != model_to_load.wubu_config: logger.warning("Checkpoint WuBu config differs.")
            if checkpoint_data.get('sequence_config') != model_to_load.sequence_config: logger.warning("Checkpoint Sequence config differs.")
            if checkpoint_data.get('feature_config') != model_to_load.feature_config: logger.warning("Checkpoint Feature config differs."); self.feature_config = checkpoint_data.get('feature_config', self.feature_config)
            incompatible = model_to_load.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
            if incompatible.missing_keys: logger.warning(f"Ckpt Load - Missing model keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys: logger.warning(f"Ckpt Load - Unexpected model keys: {incompatible.unexpected_keys}")
            model_to_load.to(self.device); logger.info("Model state loaded.")
            if 'optimizer_state_dict' in checkpoint_data:
                try:
                    self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                try: state[k] = v.to(self.device)
                                except Exception as move_err: logger.warning(f"Could not move optim state tensor {k} to {self.device}: {move_err}")
                    logger.info("Optimizer state loaded.")
                except Exception as optim_load_err: logger.warning(f"Failed load optim state: {optim_load_err}. Resetting."); self.optimizer.state = defaultdict(dict)
            else: logger.warning("Optim state not found. Starting fresh.")
            saved_amp_enabled = checkpoint_data.get('amp_enabled', False)
            if self.use_amp:
                if 'scaler_state_dict' in checkpoint_data and checkpoint_data['scaler_state_dict'] is not None and saved_amp_enabled:
                    try: self.scaler.load_state_dict(checkpoint_data['scaler_state_dict']); logger.info("AMP scaler state loaded.")
                    except Exception as scaler_load_err: logger.warning(f"Failed load AMP scaler state: {scaler_load_err}. Resetting."); self.scaler = amp.GradScaler(enabled=self.use_amp)
                elif not saved_amp_enabled: logger.warning("Ckpt saved w/o AMP, but AMP enabled. Fresh scaler.")
            elif saved_amp_enabled: logger.warning("Ckpt saved w/ AMP, but AMP disabled. Ignoring scaler.")
            start_epoch = checkpoint_data.get('epoch', -1) + 1; self.global_step = checkpoint_data.get('global_step', 0)
            self.current_epoch = start_epoch - 1 if start_epoch > 0 else 0; self.last_val_metrics = checkpoint_data.get('metrics'); self.loss_weights = checkpoint_data.get('loss_weights', self.loss_weights)
            if self.last_val_metrics: logger.info(f"Restored last metrics: {self.last_val_metrics}")
            logger.info(f"Restored loss weights: {self.loss_weights}")
            if self.has_q_controller and self.optimizer.q_controller and 'q_controller_state' in checkpoint_data:
                q_state = checkpoint_data['q_controller_state']; logger.info("Loading Q-Controller state...")
                try:
                    q_ctrl = self.optimizer.q_controller
                    loaded_action_ranges = q_state.get('action_ranges', {}); current_action_ranges = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in q_ctrl.action_ranges.items()}
                    if current_action_ranges != loaded_action_ranges: logger.warning("QCtrl Action Ranges differ. Resetting Q-state."); q_ctrl.q_table,q_ctrl.q_table_access_count,q_ctrl.q_table_creation_time = {},defaultdict(int),{}
                    else:
                         q_ctrl.q_table = {tuple(eval(k_str)) if isinstance(eval(k_str), list) else eval(k_str) : {param: np.array(arr, dtype=np.float32) for param, arr in v_dict.items()} for k_str, v_dict in q_state.get('q_table', {}).items()}
                         q_ctrl.q_table_access_count = defaultdict(int, {tuple(eval(k_str)) if isinstance(eval(k_str), list) else eval(k_str): v for k_str, v in q_state.get('access_count', {}).items()})
                         q_ctrl.q_table_creation_time = {tuple(eval(k_str)) if isinstance(eval(k_str), list) else eval(k_str): v for k_str, v in q_state.get('creation_time', {}).items()}
                    q_ctrl.epsilon = q_state.get('epsilon', q_ctrl.epsilon)
                    q_ctrl.loss_window = deque(q_state.get('loss_window', []), maxlen=q_ctrl.loss_window.maxlen)
                    q_ctrl.grad_norm_window = deque(q_state.get('grad_norm_window', []), maxlen=q_ctrl.grad_norm_window.maxlen)
                    q_ctrl.performance_window = deque(q_state.get('performance_window', []), maxlen=q_ctrl.performance_window.maxlen)
                    q_ctrl.stable_steps,q_ctrl.oscillation_counter,q_ctrl.prev_loss=q_state.get('stable_steps',0),q_state.get('oscillation_counter',0),q_state.get('prev_loss')
                    prev_state_loaded = q_state.get('prev_state'); q_ctrl.prev_state = tuple(eval(prev_state_loaded)) if isinstance(prev_state_loaded, str) else tuple(prev_state_loaded) if prev_state_loaded else None
                    q_ctrl.prev_action, q_ctrl.num_actions = q_state.get('prev_action'), q_state.get('num_actions', q_ctrl.num_actions)
                    logger.info("Q-Controller state loaded (or reset if incompatible).")
                except Exception as q_load_err: logger.warning(f"Failed load Q-Controller state: {q_load_err}. Resetting.",exc_info=False)
                if self.has_q_controller and self.optimizer.q_controller: q_ctrl=self.optimizer.q_controller; q_ctrl.q_table,q_ctrl.q_table_access_count,q_ctrl.q_table_creation_time,q_ctrl.prev_loss,q_ctrl.prev_state,q_ctrl.prev_action={},defaultdict(int),{},None,None,None; q_ctrl.loss_window.clear();q_ctrl.grad_norm_window.clear();q_ctrl.performance_window.clear()
            elif self.has_q_controller: logger.warning("QCtrl enabled, but no state found in ckpt. Starting fresh.")
            logger.info(f"Ckpt '{os.path.basename(filepath)}' loaded. Resuming Epoch {start_epoch} (GStep {self.global_step}).")
            return start_epoch
        except Exception as load_error:
            logger.error(f"Failed load ckpt '{filepath}': {load_error}", exc_info=True)
            self.global_step,self.current_epoch,self.last_val_metrics = 0,0,None; self.optimizer.state = defaultdict(dict)
            if self.use_amp: self.scaler = amp.GradScaler(enabled=self.use_amp)
            if self.has_q_controller and self.optimizer.q_controller: q_ctrl=self.optimizer.q_controller; q_ctrl.q_table,q_ctrl.q_table_access_count,q_ctrl.q_table_creation_time,q_ctrl.prev_loss,q_ctrl.prev_state,q_ctrl.prev_action={},defaultdict(int),{},None,None,None; q_ctrl.loss_window.clear();q_ctrl.grad_norm_window.clear();q_ctrl.performance_window.clear()
            return 0

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
            # Set epoch for samplers and datasets
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
            val_metrics = self._validate() # Returns metrics only on rank 0
            val_duration = time.monotonic() - val_start_time
            if self.is_main and val_metrics:
                logger.info(f"Epoch {epoch+1} Validation finished in {timedelta(seconds=val_duration)}.")

            if self.is_main and self.save_interval >= 0:
                # Pass validation metrics for checkpoint naming/logging
                self._save_checkpoint(is_intermediate=False, metrics=val_metrics)

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
    "num_levels": 3, "hyperbolic_dims": [128, 96, 64], "initial_curvatures": [1.0, 0.7, 0.4],
    "initial_scales": [1.0, 1.2, 1.5], "initial_spread_values": [0.1, 0.2, 0.3],
    "boundary_points_per_level": [16, 12, 8], "learnable_curvature": True, "learnable_scales": True,
    "learnable_spread": True, "curvature_min_value": 0.01, "scale_min_value": 0.1, "spread_min_value": 0.01,
    "use_level_descriptors": True, "use_level_spread": True, "level_descriptor_init_scale": 1e-5,
    "transform_types": ["mlp", "mlp"], "transform_hidden_dims": [96, 64],
    "relative_vector_aggregation": "mean", "aggregation_method": "concat_tangent", "dropout": 0.1,
    "use_tangent_flow": True, "tangent_flow_type": "mlp", "tangent_flow_hidden_dim_ratio": 0.5, "tangent_flow_scale": 0.1
}
DEFAULT_CONFIG_SEQUENCE = {
    "local_hidden_size": 256, "decoder_memory_dim": sum(DEFAULT_CONFIG_WUBU["hyperbolic_dims"]), "context_window": 512,
    "num_encoder_layers": 6, "num_encoder_heads": 8, "num_decoder_layers": 6, "num_decoder_heads": 8,
    "encoder_max_seq_len": 2048, "decoder_max_seq_len": 2048,
}
DEFAULT_CONFIG_FEATURE = {
    "nucleotide_vocab_size": NUCLEOTIDE_VOCAB_SIZE, "structure_vocab_size": STRUCTURE_VOCAB_SIZE, "region_vocab_size": REGION_VOCAB_SIZE,
    "structure_source_vocab_size": STRUCTURE_SOURCE_VOCAB_SIZE, "dataset_source_vocab_size": DATASET_SOURCE_VOCAB_SIZE, "codon_vocab_size": CODON_VOCAB_SIZE
}
DEFAULT_CONFIG_LOSS = {'nucleotide': 1.0, 'structure': 0.2, 'region': 0.1, 'predicted_structure_weight': 0.5}
DEFAULT_CONFIG_QLEARN = {
    "learning_rate": 0.015, "discount": 0.90, "epsilon": 0.15, "epsilon_decay": 0.99985, "min_epsilon": 0.005,
    "lr_scale_options": [0.95, 0.98, 1.0, 1.02, 1.05], "momentum_scale_options": [0.98, 0.99, 1.0, 1.005, 1.01], "max_q_table_size": 15000
}

# =====================================================================
# Argument Parsing
# =====================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBu Nesting Model Trainer (Hybrid Spatial/Hyperbolic - PyTorch Fallback)")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--use_combined_dataset", action="store_true")
    parser.add_argument("--max_combined_datasets", type=int, default=None)
    parser.add_argument("--balanced_sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_hybrid_pytorch")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--gencode_annotation_db", type=str, default=None)
    parser.add_argument("--refseq_annotation_db", type=str, default=None)
    parser.add_argument("--context_size", type=int, default=DEFAULT_CONFIG_SEQUENCE["context_window"])
    parser.add_argument("--local_hidden_size", type=int, default=DEFAULT_CONFIG_SEQUENCE["local_hidden_size"])
    parser.add_argument("--num_encoder_layers", type=int, default=DEFAULT_CONFIG_SEQUENCE["num_encoder_layers"])
    parser.add_argument("--num_decoder_layers", type=int, default=DEFAULT_CONFIG_SEQUENCE["num_decoder_layers"])
    parser.add_argument("--num_encoder_heads", type=int, default=DEFAULT_CONFIG_SEQUENCE["num_encoder_heads"])
    parser.add_argument("--num_decoder_heads", type=int, default=DEFAULT_CONFIG_SEQUENCE["num_decoder_heads"])
    parser.add_argument("--encoder_max_len", type=int, default=DEFAULT_CONFIG_SEQUENCE["encoder_max_seq_len"])
    parser.add_argument("--decoder_max_len", type=int, default=DEFAULT_CONFIG_SEQUENCE["decoder_max_seq_len"])
    parser.add_argument("--wubu_levels", type=int, default=DEFAULT_CONFIG_WUBU["num_levels"])
    parser.add_argument("--wubu_dims", nargs='+', type=int, default=DEFAULT_CONFIG_WUBU["hyperbolic_dims"])
    parser.add_argument("--structure_loss_weight", type=float, default=DEFAULT_CONFIG_LOSS["structure"])
    parser.add_argument("--region_loss_weight", type=float, default=DEFAULT_CONFIG_LOSS["region"])
    parser.add_argument("--predicted_structure_weight", type=float, default=DEFAULT_CONFIG_LOSS["predicted_structure_weight"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--target_effective_batch_size", type=int, default=None)
    parser.add_argument("--creative_batching", action="store_true")
    parser.add_argument("--creative_batching_vram_gb", type=float, default=40.0)
    parser.add_argument("--creative_batching_safety_factor", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_q_learning", action="store_true")
    default_num_workers = min(os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1, 4)
    parser.add_argument("--num_workers", type=int, default=default_num_workers)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distributed_backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--wandb_project", type=str, default="WuBuNestingHybridPyTorch")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true")
    args = parser.parse_args()
    if not args.use_combined_dataset and args.dataset_name is None: parser.error("Must specify --dataset_name or --use_combined_dataset.")
    if args.use_combined_dataset and args.dataset_name is not None: logger.warning(f"Ignoring --dataset_name='{args.dataset_name}'."); args.dataset_name = None
    if len(args.wubu_dims) != args.wubu_levels:
        logger.warning(f"Mismatch: --wubu_levels={args.wubu_levels} vs --wubu_dims ({len(args.wubu_dims)}). Adjusting..."); default_dims = DEFAULT_CONFIG_WUBU["hyperbolic_dims"]
        adjusted_dims = args.wubu_dims[:args.wubu_levels]; args.wubu_dims = adjusted_dims + [default_dims[len(adjusted_dims)] if len(adjusted_dims) < len(default_dims) else (adjusted_dims[-1] if adjusted_dims else 64)] * (args.wubu_levels - len(adjusted_dims))
        logger.warning(f"Using adjusted wubu_dims: {args.wubu_dims}")
    if args.local_hidden_size % args.num_encoder_heads != 0: raise ValueError(f"Encoder hidden size ({args.local_hidden_size}) not div by heads ({args.num_encoder_heads})")
    if args.local_hidden_size % args.num_decoder_heads != 0: raise ValueError(f"Decoder hidden size ({args.local_hidden_size}) not div by heads ({args.num_decoder_heads})")
    if args.num_workers < 0: args.num_workers = 0
    if args.num_workers == 0: args.prefetch_factor = None
    if not torch.cuda.is_available() and args.pin_memory: args.pin_memory = False
    args.gff_db_paths = {}
    if args.gencode_annotation_db: args.gff_db_paths['gencode'] = args.gencode_annotation_db
    if args.refseq_annotation_db: args.gff_db_paths['refseq'] = args.refseq_annotation_db
    args.loss_weights = {'nucleotide': 1.0, 'structure': args.structure_loss_weight, 'region': args.region_loss_weight, 'predicted_structure_weight': args.predicted_structure_weight}
    if args.creative_batching and args.target_effective_batch_size is None: args.target_effective_batch_size = args.batch_size * args.grad_accum_steps; logger.info(f"CB: Default target_effective_batch_size={args.target_effective_batch_size}")
    elif args.creative_batching: logger.info(f"CB enabled with target_effective_batch_size={args.target_effective_batch_size}.")
    return args

# =====================================================================
# Distributed Setup Utilities
# =====================================================================
def setup_distributed(backend='nccl'):
    if is_initialized():
        logger.warning("DDP already initialized.")
        rank = get_rank(); world_size = get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count() if torch.cuda.is_available() else 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        return device, rank, local_rank, world_size
    try:
        rank, local_rank, world_size = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
        master_addr, master_port = os.environ.get('MASTER_ADDR', 'localhost'), os.environ.get('MASTER_PORT', '12355')
        if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
            torch.cuda.set_device(local_rank); device = torch.device(f"cuda:{local_rank}")
            if backend.lower() != 'nccl': logger.warning(f"CUDA available, but backend is '{backend}'. NCCL recommended.")
        else:
            device = torch.device("cpu")
            if backend.lower() == 'nccl': logger.warning("CUDA unavailable/insufficient. NCCL needs CUDA. Switching to 'gloo'."); backend = 'gloo'
            else: logger.info(f"Using CPU with '{backend}' backend.")
        print(f"Rank {rank} Initializing DDP: Backend={backend}, Addr={master_addr}:{master_port}, WorldSize={world_size}")
        init_process_group(backend=backend, init_method=f'tcp://{master_addr}:{master_port}', world_size=world_size, rank=rank, timeout=timedelta(seconds=1800))
        logger.info(f"DDP Rank {rank}/{world_size} initialized. Backend: {backend}. Device: {device}.")
        logger.debug(f"Rank {rank} entering barrier after init."); torch.distributed.barrier(); logger.debug(f"Rank {rank} passed barrier.")
        return device, rank, local_rank, world_size
    except KeyError as e: logger.error(f"DDP env var missing: {e}. Use torchrun."); raise RuntimeError(f"DDP env var missing: {e}") from e
    except Exception as e: logger.error(f"DDP init failed: {e}", exc_info=True); raise RuntimeError("DDP init failed") from e

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
        try: device, rank, local_rank, world_size = setup_distributed(backend=args.distributed_backend)
        except Exception as e: print(f"FATAL: DDP setup failed: {e}", file=sys.stderr); sys.exit(1)
    elif torch.cuda.is_available(): device = torch.device("cuda:0")
    else: logger.info("CUDA not available, running on CPU."); args.distributed_backend = 'gloo' if args.distributed_backend == 'nccl' else args.distributed_backend

    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = f'%(asctime)s - R{rank} - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level_int, format=log_format, force=True); logger.setLevel(log_level_int)
    for _lg_name in ["requests", "urllib3", "PIL", "matplotlib", "h5py"]: logging.getLogger(_lg_name).setLevel(logging.WARNING)

    if rank == 0:
        logger.info("=====================================================================")
        logger.info(" WuBu Nesting Model Trainer (Hybrid Spatial/Hyperbolic V3 - PyTorch Fallback)")
        logger.info("=====================================================================")
        logger.info(f"Run Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Platform: {platform.system()} ({platform.release()}) | Host: {socket.gethostname()}")
        logger.info(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        logger.info(f"BioPython: {BIOPYTHON_AVAILABLE} | ViennaRNA: {VIENNARNA_AVAILABLE} | NetworkX: {NETWORKX_AVAILABLE} | h5py: {H5PY_AVAILABLE} | WandB: {WANDB_AVAILABLE}")
        logger.info(f"Distributed: {is_distributed} (World: {world_size}) | Device: {device}")
        if torch.cuda.is_available(): logger.info(f"CUDA Device: {torch.cuda.get_device_name(local_rank if is_distributed else 0)}")
        logger.info(f"Use Combined HDF5: {args.use_combined_dataset} | Balanced Sampling: {args.balanced_sampling if args.use_combined_dataset else 'N/A'}")
        logger.info(f"Creative Batching: {args.creative_batching} (VRAM Target: {args.creative_batching_vram_gb} GB, Safety: {args.creative_batching_safety_factor})")
        logger.info(f"Arguments: {vars(args)}")
        logger.info("=====================================================================")

    seed_offset = args.seed + rank; torch.manual_seed(seed_offset); np.random.seed(seed_offset); random.seed(seed_offset)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_offset)

    processed_h5_path, dataset_class, dataset_args = None, None, {}
    if rank == 0:
        if args.use_combined_dataset:
            logger.info(f"Rank 0: Prep COMBINED HDF5 dataset (Max: {args.max_combined_datasets})...")
            processed_h5_path = prepare_combined_dataset_h5(args.data_dir, max_datasets=args.max_combined_datasets, gff_db_paths=args.gff_db_paths)
        elif args.dataset_name:
            logger.info(f"Rank 0: Prep SINGLE HDF5 dataset '{args.dataset_name}'...")
            prep_result = prepare_dataset_h5(args.dataset_name, args.data_dir, gff_db_paths=args.gff_db_paths)
            processed_h5_path = prep_result[0] if prep_result else None
        else: logger.error("Rank 0: No dataset specified. Exiting."); cleanup_distributed() if world_size > 1 else None; sys.exit(1)
        if processed_h5_path is None or not os.path.exists(processed_h5_path): logger.error(f"Rank 0: HDF5 Dataset prep failed. Exiting."); cleanup_distributed() if world_size > 1 else None; sys.exit(1)
        else: logger.info(f"Rank 0: HDF5 Dataset ready at '{processed_h5_path}'")

    if world_size > 1:
        logger.debug(f"Rank {rank} waiting at dataset path barrier..."); path_list = [processed_h5_path] if rank == 0 else [None]
        torch.distributed.broadcast_object_list(path_list, src=0); processed_h5_path = path_list[0]; torch.distributed.barrier()
        logger.debug(f"Rank {rank} passed path barrier. Path: {processed_h5_path}")
        if processed_h5_path is None: logger.error(f"Rank {rank}: No valid HDF5 path from Rank 0. Exiting."); cleanup_distributed(); sys.exit(1)
    elif processed_h5_path is None: logger.error("HDF5 Dataset prep failed. Exiting."); sys.exit(1)

    if args.use_combined_dataset: dataset_class, dataset_args = BalancedWuBuNestingDatasetH5, {"h5_file_path": processed_h5_path, "context_size": args.context_size, "balanced_sampling": args.balanced_sampling, "metadata_path": os.path.join(args.data_dir, COMBINED_DATA_INFO_FILE)}
    else: dataset_class, dataset_args = WuBuNestingDatasetH5, {"h5_file_path": processed_h5_path, "context_size": args.context_size}

    effective_batch_size, adjusted_batch_size, adjusted_accum_steps = args.batch_size * args.grad_accum_steps, args.batch_size, args.grad_accum_steps
    if args.creative_batching and torch.cuda.is_available():
        try:
            vram_gb, safety_factor = args.creative_batching_vram_gb, max(1.1, args.creative_batching_safety_factor)
            fixed_mem_gb_estimate = vram_gb * 0.10; bytes_per_element = 2 if args.use_amp else 4; overhead_factor = 30
            max_len = max(args.encoder_max_len, args.decoder_max_len, args.context_size)
            sample_mem_mb = (max_len * args.local_hidden_size * overhead_factor * bytes_per_element) / (1024 * 1024)
            if sample_mem_mb > 0:
                available_mb_for_batches = (vram_gb - fixed_mem_gb_estimate) * 0.70 * 1024
                optimal_gpu_batch_size = max(4, min(512, int(available_mb_for_batches / (sample_mem_mb * safety_factor))))
                target_eff_batch = args.target_effective_batch_size if args.target_effective_batch_size is not None else effective_batch_size
                adjusted_accum_steps = max(1, round(target_eff_batch / (optimal_gpu_batch_size * world_size)))
                adjusted_batch_size = max(1, target_eff_batch // (adjusted_accum_steps * world_size))
                if adjusted_batch_size != args.batch_size or adjusted_accum_steps != args.grad_accum_steps: logger.warning(f"CB Applied: BS per GPU: {args.batch_size}->{adjusted_batch_size}, Accum: {args.grad_accum_steps}->{adjusted_accum_steps} (EffBS: {adjusted_batch_size * adjusted_accum_steps * world_size})")
                else: logger.info("CB: No adjustment needed.")
            else: logger.warning("CB: Could not estimate mem/sample. Using original settings.")
        except Exception as cb_err: logger.error(f"CB calc failed: {cb_err}. Using original settings.", exc_info=True)
    elif rank == 0: logger.info("Creative batching disabled or not applicable.")

    try:
        train_dataset, val_dataset = dataset_class(**dataset_args), dataset_class(**dataset_args)
        train_dataset.set_seed(args.seed + rank); val_dataset.set_seed(args.seed + world_size + rank)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed + rank, drop_last=True) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_distributed else None
        rank_worker_offset = rank * args.num_workers; g = torch.Generator(); g.manual_seed(args.seed + rank)
        worker_init_fn = functools.partial(seed_worker, base_seed=args.seed, rank_offset=rank_worker_offset)
        dl_args = {"batch_size":adjusted_batch_size, "shuffle":False, "num_workers":args.num_workers, "pin_memory":args.pin_memory, "prefetch_factor":args.prefetch_factor if args.num_workers > 0 else None, "worker_init_fn":worker_init_fn if args.num_workers > 0 else None, "generator":g if args.num_workers > 0 else None, "persistent_workers":args.num_workers > 0}
        train_loader = DataLoader(train_dataset, sampler=train_sampler, drop_last=is_distributed, **dl_args)
        val_loader = DataLoader(val_dataset, sampler=val_sampler, drop_last=False, **dl_args)
        train_len_est, val_len_est = len(train_loader) if hasattr(train_loader, '__len__') else 'N/A', len(val_loader) if hasattr(val_loader, '__len__') else 'N/A'
        logger.info(f"Rank {rank}: Dataloaders created (Train Est: {train_len_est}, Val Est: {val_len_est}, GPU BS: {adjusted_batch_size})")
    except Exception as e: logger.error(f"Rank {rank}: Failed create datasets/loaders: {e}", exc_info=True); cleanup_distributed() if is_distributed else None; sys.exit(1)

    wubu_config = DEFAULT_CONFIG_WUBU.copy(); wubu_config["num_levels"] = args.wubu_levels; wubu_config["hyperbolic_dims"] = args.wubu_dims
    num_levels, num_transforms = wubu_config["num_levels"], max(0, wubu_config["num_levels"] - 1)
    def _resize_config_list(cd, key, tlen, dlist):
        clist = cd.get(key, []);
        if len(clist) == tlen: return
        logger.debug(f"Adjusting WuBu cfg '{key}' to len {tlen}.")
        rlist = dlist[:tlen]; rlist.extend([rlist[-1] if rlist else (dlist[0] if dlist else None)] * (tlen - len(rlist))); cd[key] = rlist
    for k, default_list in [("initial_curvatures", DEFAULT_CONFIG_WUBU["initial_curvatures"]), ("initial_scales", DEFAULT_CONFIG_WUBU["initial_scales"]), ("initial_spread_values", DEFAULT_CONFIG_WUBU["initial_spread_values"]), ("boundary_points_per_level", DEFAULT_CONFIG_WUBU["boundary_points_per_level"])]: _resize_config_list(wubu_config, k, num_levels, default_list)
    for k, default_list in [("transform_types", DEFAULT_CONFIG_WUBU["transform_types"]), ("transform_hidden_dims", DEFAULT_CONFIG_WUBU["transform_hidden_dims"])]: _resize_config_list(wubu_config, k, num_transforms, default_list)

    sequence_config = DEFAULT_CONFIG_SEQUENCE.copy(); sequence_config.update({"local_hidden_size": args.local_hidden_size, "decoder_memory_dim": sum(wubu_config["hyperbolic_dims"]), "context_window": args.context_size, "num_encoder_layers": args.num_encoder_layers, "num_decoder_layers": args.num_decoder_layers, "num_encoder_heads": args.num_encoder_heads, "num_decoder_heads": args.num_decoder_heads, "encoder_max_seq_len": args.encoder_max_len, "decoder_max_seq_len": args.decoder_max_len})
    feature_config = DEFAULT_CONFIG_FEATURE.copy()

    try: model = WuBuNestingSequenceModel(wubu_config=wubu_config, sequence_config=sequence_config, feature_config=feature_config); model.to(device)
    except Exception as e: logger.error(f"Rank {rank}: Failed model init: {e}", exc_info=True); cleanup_distributed() if is_distributed else None; sys.exit(1)
    if rank == 0: logger.info(f"Model: {type(model).__name__}, Total Params: {sum(p.numel() for p in model.parameters()):,}, Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if is_distributed: model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None, output_device=local_rank if device.type == 'cuda' else None, find_unused_parameters=False, broadcast_buffers=True); logger.info(f"Rank {rank}: Model wrapped with DDP.")
    q_config = DEFAULT_CONFIG_QLEARN if not args.disable_q_learning else None
    try: optimizer = RiemannianEnhancedSGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, q_learning_config=q_config); logger.info(f"Rank {rank}: Optimizer: {type(optimizer).__name__} (QCtrl: {q_config is not None})")
    except Exception as e: logger.error(f"Rank {rank}: Failed optim init: {e}", exc_info=True); cleanup_distributed() if is_distributed else None; sys.exit(1)

    if rank == 0 and not args.disable_wandb and WANDB_AVAILABLE:
        try:
            run_ts = datetime.now().strftime('%Y%m%d_%H%M'); ds_flag = "Comb" if args.use_combined_dataset else (args.dataset_name[:8] if args.dataset_name else "Unk"); cb_flag = "_CB" if args.creative_batching else ""; q_flag = "_NoQ" if args.disable_q_learning else ""
            eff_bs = adjusted_batch_size*adjusted_accum_steps*world_size; wandb_run_name = f"wubuHybPT_{ds_flag}_L{args.wubu_levels}H{args.local_hidden_size}_EBS{eff_bs}{cb_flag}{q_flag}_{run_ts}"
            run_config = vars(args).copy(); run_config.update({'adjusted_batch_size_per_gpu': adjusted_batch_size, 'adjusted_grad_accum_steps': adjusted_accum_steps, 'effective_batch_size': eff_bs}); run_config.update({f"wubu_{k}": v for k,v in wubu_config.items()}); run_config.update({f"seq_{k}": v for k,v in sequence_config.items()}); run_config.update({f"feat_{k}": v for k,v in feature_config.items()})
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=run_config, name=wandb_run_name, job_type="train", resume="allow"); logger.info("WandB initialized.")
        except Exception as e: logger.error(f"WandB init failed: {e}. Disabling.", exc_info=True); args.disable_wandb = True
    elif rank == 0 and not args.disable_wandb: logger.warning("WandB requested but not found. Disabling."); args.disable_wandb = True

    try:
        trainer = Trainer(model=model, optimizer=optimizer, device=device, train_loader=train_loader, val_loader=val_loader, grad_accum_steps=adjusted_accum_steps, use_amp=args.use_amp, log_interval=args.log_interval, save_interval=args.save_interval, checkpoint_dir=args.checkpoint_dir, wandb_enabled=(rank == 0 and not args.disable_wandb), max_grad_norm=args.max_grad_norm, rank=rank, world_size=world_size, detect_anomaly=args.detect_anomaly, feature_config=feature_config, loss_weights=args.loss_weights)
        trainer.args = args; logger.info(f"Rank {rank}: Trainer initialized.")
    except Exception as e:
        logger.error(f"Rank {rank}: Failed Trainer init: {e}", exc_info=True)
        if rank == 0 and not args.disable_wandb and WANDB_AVAILABLE and wandb.run: wandb.finish(exit_code=1)
        cleanup_distributed() if is_distributed else None; sys.exit(1)

    start_epoch = 0
    if args.load_checkpoint:
        try:
            if rank == 0: logger.info(f"Attempting load ckpt: {args.load_checkpoint}"); start_epoch = trainer.load_checkpoint(args.load_checkpoint); logger.info(f"Rank 0: Ckpt loaded. Resume epoch {start_epoch+1}.")
            if world_size > 1: start_epoch_tensor = torch.tensor(start_epoch, dtype=torch.int, device=device); torch.distributed.broadcast(start_epoch_tensor, src=0); start_epoch = start_epoch_tensor.item()
            if rank > 0: logger.info(f"Rank {rank}: Loading checkpoint {args.load_checkpoint}..."); _ = trainer.load_checkpoint(args.load_checkpoint)
            if world_size > 1: torch.distributed.barrier()
        except FileNotFoundError: logger.error(f"Ckpt file not found: {args.load_checkpoint}. Start fresh.")
        except Exception as e: logger.error(f"Failed load ckpt: {e}. Start fresh.", exc_info=True); start_epoch = 0; trainer.optimizer.state = defaultdict(dict); trainer.global_step,trainer.current_epoch,trainer.last_val_metrics = 0,0,None
        if trainer.use_amp: trainer.scaler = amp.GradScaler(enabled=trainer.use_amp)
        if trainer.has_q_controller and trainer.optimizer.q_controller: q_ctrl=trainer.optimizer.q_controller; q_ctrl.q_table,q_ctrl.q_table_access_count,q_ctrl.q_table_creation_time,q_ctrl.prev_loss,q_ctrl.prev_state,q_ctrl.prev_action={},defaultdict(int),{},None,None,None; q_ctrl.loss_window.clear();q_ctrl.grad_norm_window.clear();q_ctrl.performance_window.clear()

    try:
        if rank == 0: logger.info("Starting main training loop...")
        trainer.train(epochs=args.epochs, start_epoch=start_epoch)
        if rank == 0: logger.info("Training finished successfully.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted.")
        if rank == 0 and args.save_interval >= 0: logger.info("Saving interrupt ckpt..."); trainer._save_checkpoint(is_intermediate=True, metrics=trainer.last_val_metrics)
    except Exception as e:
        logger.error(f"Unhandled training exception Rank {rank}: {e}", exc_info=True)
        if rank == 0 and not args.disable_wandb and WANDB_AVAILABLE and wandb.run: logger.error("Finishing WandB run with error."); wandb.finish(exit_code=1)
        cleanup_distributed() if is_distributed else None; sys.exit(1)

    cleanup_distributed() if is_distributed else None
    if rank == 0 and not args.disable_wandb and WANDB_AVAILABLE and wandb.run: logger.info("Finishing WandB run normally."); wandb.finish()
    logger.info(f"Rank {rank}: Script execution finished.")

if __name__ == "__main__":
    run()