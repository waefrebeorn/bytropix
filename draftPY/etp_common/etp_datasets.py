import logging
import os
from typing import List, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to import h5py for HDF5 support
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    logging.warning("h5py library not found. HDF5 file support will be disabled.")


def load_embeddings_from_file(filepath: str) -> List[np.ndarray]:
    """
    Loads embeddings from a file (.npz or .hdf5/.h5).

    Args:
        filepath: Path to the embedding file.

    Returns:
        A list of NumPy arrays, where each array is an embedding.
    
    Raises:
        FileNotFoundError: If the filepath does not exist.
        ValueError: If the file type is unsupported or loading fails.
    """
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    embeddings: List[np.ndarray] = []
    file_extension = os.path.splitext(filepath)[1].lower()
    logging.info(f"Attempting to load embeddings from {filepath} (type: {file_extension})")

    if file_extension == ".npz":
        try:
            with np.load(filepath, allow_pickle=True) as data:
                # Check if it was saved as individual args ('arr_0', 'arr_1', ...)
                # or as a single named array containing a list/object
                if len(data.files) == 1 and isinstance(data[data.files[0]], (list, np.ndarray)) and data[data.files[0]].dtype == 'object':
                     # This case handles np.savez_compressed(output_path, embeddings=list_of_arrays)
                     # or if the single array is an object array of arrays.
                    loaded_obj = data[data.files[0]]
                    if isinstance(loaded_obj, list): # list of arrays
                        embeddings = [np.asarray(emb) for emb in loaded_obj]
                    elif loaded_obj.ndim > 0 and isinstance(loaded_obj[0], np.ndarray): # object array of arrays
                        embeddings = [np.asarray(emb) for emb in loaded_obj]
                    else: # single array, needs to be listified
                         embeddings = [np.asarray(loaded_obj)]

                elif len(data.files) > 0 and all(f.startswith('arr_') for f in data.files):
                    # This case handles np.savez_compressed(output_path, *list_of_arrays)
                    # Files are named 'arr_0', 'arr_1', ...
                    embeddings = [data[f] for f in sorted(data.files, key=lambda x: int(x.split('_')[1]))]
                elif len(data.files) > 0 : # try to load all arrays in the npz
                    embeddings = [data[f] for f in data.files]
                else:
                    logging.warning(f"NPZ file {filepath} is empty or has an unexpected structure.")
                    return [] # Return empty list if no compatible data found

                # Ensure all loaded embeddings are numpy arrays
                embeddings = [np.asarray(emb) for emb in embeddings if emb is not None]
                logging.info(f"Successfully loaded {len(embeddings)} embeddings from NPZ file {filepath}.")
        except Exception as e:
            logging.error(f"Error loading NPZ file {filepath}: {e}")
            raise ValueError(f"Could not load embeddings from NPZ file {filepath}: {e}")

    elif file_extension in [".hdf5", ".h5"]:
        if not H5PY_AVAILABLE:
            logging.error("h5py library is required to load HDF5 files, but it's not installed.")
            raise ImportError("h5py library is required for HDF5 support.")
        try:
            with h5py.File(filepath, 'r') as hf:
                # Assuming embeddings are stored as datasets named 'embedding_0', 'embedding_1', ...
                # or just any dataset in the file.
                dataset_keys = sorted(list(hf.keys()), key=lambda x: int(x.split('_')[1]) if x.startswith('embedding_') and x.split('_')[1].isdigit() else 0)
                for key in dataset_keys:
                    embeddings.append(hf[key][:]) # [:] loads the dataset into a NumPy array
            logging.info(f"Successfully loaded {len(embeddings)} embeddings from HDF5 file {filepath}.")
        except Exception as e:
            logging.error(f"Error loading HDF5 file {filepath}: {e}")
            raise ValueError(f"Could not load embeddings from HDF5 file {filepath}: {e}")
    
    # Fallback for directory of .npy files (as discussed in requirements)
    elif os.path.isdir(filepath):
        logging.info(f"{filepath} is a directory. Attempting to load .npy files from it.")
        try:
            npy_files = sorted([f for f in os.listdir(filepath) if f.endswith(".npy")])
            if not npy_files:
                logging.warning(f"No .npy files found in directory {filepath}.")
                return []
            for f_name in npy_files:
                npy_path = os.path.join(filepath, f_name)
                embeddings.append(np.load(npy_path, allow_pickle=True))
            logging.info(f"Successfully loaded {len(embeddings)} embeddings from .npy files in {filepath}.")
        except Exception as e:
            logging.error(f"Error loading .npy files from directory {filepath}: {e}")
            raise ValueError(f"Could not load embeddings from directory {filepath}: {e}")

    else:
        logging.error(f"Unsupported file type: {file_extension} for file {filepath}")
        raise ValueError(f"Unsupported file type: {file_extension}. Please use .npz, .hdf5, .h5, or a directory of .npy files.")

    return embeddings


class DeepSeekR1EmbeddingDataset(Dataset):
    """
    A PyTorch Dataset for loading pairs of sentence embeddings from two sources.
    """
    def __init__(self, embeddings_file_A: str, embeddings_file_B: str):
        """
        Args:
            embeddings_file_A: Path to the first embedding file (e.g., corpus_A.npz).
            embeddings_file_B: Path to the second embedding file (e.g., corpus_B.npz).
        """
        logging.info(f"Initializing DeepSeekR1EmbeddingDataset with files: {embeddings_file_A}, {embeddings_file_B}")
        try:
            self.embeddings_A = load_embeddings_from_file(embeddings_file_A)
            self.embeddings_B = load_embeddings_from_file(embeddings_file_B)
        except Exception as e:
            logging.error(f"Failed to load embeddings during dataset initialization: {e}")
            # Propagate the error or handle it by setting embeddings to empty lists
            # For now, let's re-raise to make it clear initialization failed.
            raise

        if not self.embeddings_A:
            logging.warning(f"Embeddings file A ({embeddings_file_A}) resulted in an empty list of embeddings.")
        if not self.embeddings_B:
            logging.warning(f"Embeddings file B ({embeddings_file_B}) resulted in an empty list of embeddings.")
        
        if not self.embeddings_A and not self.embeddings_B:
             logging.error("Both embedding sources are empty. Dataset will be empty.")
        elif not self.embeddings_A or not self.embeddings_B:
            logging.warning("One of the embedding sources is empty. The dataset will cycle the non-empty source against an empty one if accessed directly, or behave according to __len__.")


    def __len__(self) -> int:
        """
        Returns the maximum length of the two embedding lists.
        This allows iteration up to the length of the larger dataset.
        """
        len_A = len(self.embeddings_A) if self.embeddings_A else 0
        len_B = len(self.embeddings_B) if self.embeddings_B else 0
        
        if len_A == 0 and len_B == 0:
            return 0
        return max(len_A, len_B)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, None]]:
        """
        Returns a dictionary containing embeddings from source A and source B.
        Uses modulo arithmetic to cycle through the smaller dataset if lengths differ.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            A dictionary: {'source_A': embedding_A, 'source_B': embedding_B}
            Returns None for a source if its embedding list is empty.
        """
        len_A = len(self.embeddings_A) if self.embeddings_A else 0
        len_B = len(self.embeddings_B) if self.embeddings_B else 0

        if len_A == 0 and len_B == 0:
            # This case should ideally be handled by __len__ being 0,
            # but as a safeguard for direct __getitem__ calls on an empty dataset:
            logging.warning("Attempting to get item from an empty dataset (both sources are empty).")
            return {'source_A': None, 'source_B': None}

        embedding_A = self.embeddings_A[idx % len_A] if len_A > 0 else None
        embedding_B = self.embeddings_B[idx % len_B] if len_B > 0 else None
        
        # Ensure returned values are numpy arrays if not None
        if embedding_A is not None:
            embedding_A = np.asarray(embedding_A, dtype=np.float32) # Standardize dtype
        if embedding_B is not None:
            embedding_B = np.asarray(embedding_B, dtype=np.float32)

        return {'source_A': embedding_A, 'source_B': embedding_B}


if __name__ == '__main__':
    logging.info("Starting example usage of etp_datasets.")

    # Create dummy embedding files for testing
    dummy_dir = "draftPY/dummy_embeddings_ds"
    os.makedirs(dummy_dir, exist_ok=True)

    # --- Test Case 1: NPZ files (saved with *args) ---
    npz_file_A_star = os.path.join(dummy_dir, "corpus_A_star.npz")
    npz_file_B_star = os.path.join(dummy_dir, "corpus_B_star.npz")
    
    embeddings_A_list_star = [np.random.rand(10).astype(np.float32) for _ in range(5)]
    embeddings_B_list_star = [np.random.rand(10).astype(np.float32) for _ in range(3)]
    np.savez_compressed(npz_file_A_star, *embeddings_A_list_star)
    np.savez_compressed(npz_file_B_star, *embeddings_B_list_star)
    logging.info(f"Created dummy NPZ (*args) files: {npz_file_A_star}, {npz_file_B_star}")

    # --- Test Case 2: NPZ files (saved as a single list/object array) ---
    npz_file_A_obj = os.path.join(dummy_dir, "corpus_A_obj.npz")
    embeddings_A_list_obj = [np.random.rand(10).astype(np.float32) for _ in range(5)]
    np.savez_compressed(npz_file_A_obj, embeddings=np.array(embeddings_A_list_obj, dtype=object)) # Save as object array
    logging.info(f"Created dummy NPZ (object array) file: {npz_file_A_obj}")


    # --- Test Case 3: HDF5 files ---
    if H5PY_AVAILABLE:
        hdf5_file_A = os.path.join(dummy_dir, "corpus_A.h5")
        hdf5_file_B = os.path.join(dummy_dir, "corpus_B.h5")
        with h5py.File(hdf5_file_A, 'w') as hf:
            for i, emb in enumerate(embeddings_A_list_star):
                hf.create_dataset(f"embedding_{i}", data=emb)
        with h5py.File(hdf5_file_B, 'w') as hf:
            for i, emb in enumerate(embeddings_B_list_star):
                hf.create_dataset(f"embedding_{i}", data=emb)
        logging.info(f"Created dummy HDF5 files: {hdf5_file_A}, {hdf5_file_B}")
    else:
        logging.info("Skipping HDF5 dummy file creation as h5py is not available.")
        hdf5_file_A, hdf5_file_B = None, None # To avoid issues later if they are used

    # --- Test Case 4: Directory of .npy files ---
    npy_dir_A = os.path.join(dummy_dir, "corpus_A_npy")
    npy_dir_B = os.path.join(dummy_dir, "corpus_B_npy")
    os.makedirs(npy_dir_A, exist_ok=True)
    os.makedirs(npy_dir_B, exist_ok=True)
    for i, emb in enumerate(embeddings_A_list_star):
        np.save(os.path.join(npy_dir_A, f"emb_{i}.npy"), emb)
    for i, emb in enumerate(embeddings_B_list_star):
        np.save(os.path.join(npy_dir_B, f"emb_{i}.npy"), emb)
    logging.info(f"Created dummy .npy directories: {npy_dir_A}, {npy_dir_B}")


    # Test load_embeddings_from_file
    logging.info("\n--- Testing load_embeddings_from_file ---")
    try:
        loaded_npz_star = load_embeddings_from_file(npz_file_A_star)
        assert len(loaded_npz_star) == 5
        assert isinstance(loaded_npz_star[0], np.ndarray)
        logging.info(f"Successfully loaded NPZ (*args): {len(loaded_npz_star)} embeddings.")

        loaded_npz_obj = load_embeddings_from_file(npz_file_A_obj)
        assert len(loaded_npz_obj) == 5
        assert isinstance(loaded_npz_obj[0], np.ndarray)
        logging.info(f"Successfully loaded NPZ (object array): {len(loaded_npz_obj)} embeddings.")

        if H5PY_AVAILABLE and hdf5_file_A:
            loaded_hdf5 = load_embeddings_from_file(hdf5_file_A)
            assert len(loaded_hdf5) == 5
            assert isinstance(loaded_hdf5[0], np.ndarray)
            logging.info(f"Successfully loaded HDF5: {len(loaded_hdf5)} embeddings.")
        
        loaded_npy_dir = load_embeddings_from_file(npy_dir_A)
        assert len(loaded_npy_dir) == 5
        assert isinstance(loaded_npy_dir[0], np.ndarray)
        logging.info(f"Successfully loaded .npy from directory: {len(loaded_npy_dir)} embeddings.")

    except Exception as e:
        logging.error(f"Error during load_embeddings_from_file tests: {e}", exc_info=True)


    # Test DeepSeekR1EmbeddingDataset
    logging.info("\n--- Testing DeepSeekR1EmbeddingDataset (NPZ *args) ---")
    try:
        dataset_npz_star = DeepSeekR1EmbeddingDataset(npz_file_A_star, npz_file_B_star)
        logging.info(f"Dataset length (NPZ *args): {len(dataset_npz_star)}")
        assert len(dataset_npz_star) == 5 # max(5, 3)
        
        sample = dataset_npz_star[0]
        assert 'source_A' in sample and 'source_B' in sample
        assert sample['source_A'].shape == (10,)
        assert sample['source_B'].shape == (10,)
        logging.info(f"Sample 0 (NPZ *args): source_A shape {sample['source_A'].shape}, source_B shape {sample['source_B'].shape}")

        # Test modulo access
        sample_cycled = dataset_npz_star[4] # A[4], B[4%3=1]
        assert np.array_equal(sample_cycled['source_A'], embeddings_A_list_star[4])
        assert np.array_equal(sample_cycled['source_B'], embeddings_B_list_star[1])
        logging.info("Modulo access test passed for NPZ (*args) dataset.")

    except Exception as e:
        logging.error(f"Error during DeepSeekR1EmbeddingDataset (NPZ *args) tests: {e}", exc_info=True)

    if H5PY_AVAILABLE and hdf5_file_A and hdf5_file_B:
        logging.info("\n--- Testing DeepSeekR1EmbeddingDataset (HDF5) ---")
        try:
            dataset_hdf5 = DeepSeekR1EmbeddingDataset(hdf5_file_A, hdf5_file_B)
            logging.info(f"Dataset length (HDF5): {len(dataset_hdf5)}")
            assert len(dataset_hdf5) == 5 # max(5, 3)
            
            sample_hdf5 = dataset_hdf5[0]
            assert 'source_A' in sample_hdf5 and 'source_B' in sample_hdf5
            assert sample_hdf5['source_A'].shape == (10,)
            logging.info(f"Sample 0 (HDF5): source_A shape {sample_hdf5['source_A'].shape}, source_B shape {sample_hdf5['source_B'].shape}")
            logging.info("HDF5 dataset test passed.")
        except Exception as e:
            logging.error(f"Error during DeepSeekR1EmbeddingDataset (HDF5) tests: {e}", exc_info=True)
    else:
        logging.info("Skipping HDF5 dataset test as h5py is not available or files not created.")

    logging.info("\n--- Testing DeepSeekR1EmbeddingDataset (NPY directory) ---")
    try:
        dataset_npy_dir = DeepSeekR1EmbeddingDataset(npy_dir_A, npy_dir_B)
        logging.info(f"Dataset length (NPY dir): {len(dataset_npy_dir)}")
        assert len(dataset_npy_dir) == 5 # max(5, 3)
        
        sample_npy = dataset_npy_dir[0]
        assert 'source_A' in sample_npy and 'source_B' in sample_npy
        assert sample_npy['source_A'].shape == (10,)
        logging.info(f"Sample 0 (NPY dir): source_A shape {sample_npy['source_A'].shape}, source_B shape {sample_npy['source_B'].shape}")
        logging.info("NPY directory dataset test passed.")
    except Exception as e:
        logging.error(f"Error during DeepSeekR1EmbeddingDataset (NPY dir) tests: {e}", exc_info=True)

    # Test with one empty dataset
    logging.info("\n--- Testing DeepSeekR1EmbeddingDataset (One Empty NPZ) ---")
    empty_npz = os.path.join(dummy_dir, "empty.npz")
    np.savez_compressed(empty_npz) # Create an empty npz
    try:
        dataset_one_empty = DeepSeekR1EmbeddingDataset(npz_file_A_star, empty_npz)
        logging.info(f"Dataset length (one empty): {len(dataset_one_empty)}")
        assert len(dataset_one_empty) == 5 # max(5,0)
        sample_one_empty = dataset_one_empty[0]
        assert sample_one_empty['source_A'] is not None
        assert sample_one_empty['source_B'] is None
        logging.info(f"Sample 0 (one empty): source_A shape {sample_one_empty['source_A'].shape}, source_B is {sample_one_empty['source_B']}")
        logging.info("One empty dataset test passed.")
    except Exception as e:
        logging.error(f"Error during one empty dataset test: {e}", exc_info=True)
    
    # Test with two empty datasets
    logging.info("\n--- Testing DeepSeekR1EmbeddingDataset (Two Empty NPZs) ---")
    try:
        dataset_both_empty = DeepSeekR1EmbeddingDataset(empty_npz, empty_npz)
        logging.info(f"Dataset length (both empty): {len(dataset_both_empty)}")
        assert len(dataset_both_empty) == 0
        # Accessing __getitem__ on an empty dataset should be handled gracefully if __len__ is 0
        # However, direct call for idx > 0 would be an error.
        # DataLoader would not call __getitem__ if __len__ is 0.
        logging.info("Two empty datasets test passed (length is 0).")

    except Exception as e:
        logging.error(f"Error during two empty datasets test: {e}", exc_info=True)


    logging.info("Example usage of etp_datasets finished.")
    # Consider cleaning up dummy files/dirs, but for now, leave them for inspection.
    # import shutil
    # shutil.rmtree(dummy_dir)
    # logging.info(f"Cleaned up dummy directory: {dummy_dir}")
