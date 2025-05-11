import gffutils
import os
import argparse
import logging
import time
import sys
import requests # For downloading
from tqdm import tqdm # For progress bar
import shutil # For moving downloaded file

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# Default filenames - ADJUST THESE if your downloaded files have different names
DEFAULT_GENCODE_GFF_FILENAME = "gencode.v46.annotation.gff3.gz" # Example filename
DEFAULT_REFSEQ_GFF_FILENAME = "GRCh38_latest_genomic.gff.gz"   # Example filename

# --- Download URLs ---
# !!! VERIFY AND UPDATE THESE URLS FOR THE DESIRED VERSIONS !!!
# Example URLs (might become outdated):
GENCODE_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gff3.gz"
REFSEQ_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_genomic.gff.gz"

# --- Helper Functions ---

def download_file(url: str, dest_path: str, chunk_size=8192):
    """Downloads a file from a URL to a destination path with progress."""
    temp_dest_path = dest_path + ".part"
    try:
        logger.info(f"Attempting download: {url} -> {os.path.basename(dest_path)}")
        response = requests.get(url, stream=True, timeout=600) # 10 min timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        total_size = int(response.headers.get('content-length', 0))

        with open(temp_dest_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave=False # Don't leave progress bar after completion
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    size = f.write(chunk)
                    bar.update(size)

        # Check if download was complete (sometimes servers don't provide content-length)
        if total_size != 0 and os.path.getsize(temp_dest_path) != total_size:
             logger.error(f"Download incomplete for {os.path.basename(dest_path)}. Expected {total_size}, got {os.path.getsize(temp_dest_path)}.")
             os.remove(temp_dest_path)
             return False

        shutil.move(temp_dest_path, dest_path)
        logger.info(f"Download complete: {os.path.basename(dest_path)}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed for {url}: {e}")
        if os.path.exists(temp_dest_path):
            try:
                os.remove(temp_dest_path)
            except OSError:
                pass
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during download of {url}: {e}", exc_info=True)
        if os.path.exists(temp_dest_path):
             try:
                os.remove(temp_dest_path)
             except OSError:
                pass
        return False


def create_gffutils_db(gff_filepath: str, db_filepath: str):
    """
    Creates a gffutils database from a GFF file if the DB doesn't exist.

    Args:
        gff_filepath: Path to the input GFF3 file (can be gzipped).
        db_filepath: Path where the output database (.db) should be created.
    """
    # GFF file existence is checked before calling this function now

    if os.path.exists(db_filepath):
        logger.info(f"Database already exists: {db_filepath}. Skipping creation.")
        return True

    logger.info(f"Creating gffutils database '{os.path.basename(db_filepath)}' from '{os.path.basename(gff_filepath)}'...")
    start_time = time.time()
    try:
        # Use force=True to overwrite any potentially incomplete .tmp file
        # merge_strategy helps handle duplicate IDs sometimes found in GFF files
        # disable_infer options can speed up creation if exact gene/transcript boundaries aren't needed from gffutils itself
        db = gffutils.create_db(gff_filepath,
                                dbfn=db_filepath,
                                force=True,
                                keep_order=True,
                                merge_strategy='merge', # Try 'merge' first, 'create_unique' if merge fails
                                sort_attribute_values=True,
                                disable_infer_genes=True,
                                disable_infer_transcripts=True,
                                verbose=False) # Set verbose=True for more detailed progress
        end_time = time.time()
        logger.info(f"Database '{os.path.basename(db_filepath)}' created successfully in {end_time - start_time:.2f} seconds.")
        return True
    except Exception as e:
        logger.error(f"ERROR creating database '{os.path.basename(db_filepath)}': {e}", exc_info=True)
        # Clean up potentially incomplete db file on error
        if os.path.exists(db_filepath):
            try:
                os.remove(db_filepath)
                logger.info(f"Removed incomplete database file: {db_filepath}")
            except OSError as rm_err:
                logger.error(f"Failed to remove incomplete database file {db_filepath}: {rm_err}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download GFFs and create gffutils databases for WuBu Trainer.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where GFF files will be downloaded/checked and .db files created (e.g., 'data_h5').")
    parser.add_argument("--gencode_gff_file", type=str, default=DEFAULT_GENCODE_GFF_FILENAME, help="Filename for the GENCODE GFF3 file.")
    parser.add_argument("--refseq_gff_file", type=str, default=DEFAULT_REFSEQ_GFF_FILENAME, help="Filename for the RefSeq GFF3 file.")
    parser.add_argument("--gencode_url", type=str, default=GENCODE_URL, help="URL to download GENCODE GFF3 from.")
    parser.add_argument("--refseq_url", type=str, default=REFSEQ_URL, help="URL to download RefSeq GFF3 from.")
    parser.add_argument("--skip_gencode", action="store_true", help="Skip processing GENCODE.")
    parser.add_argument("--skip_refseq", action="store_true", help="Skip processing RefSeq.")


    args = parser.parse_args()

    logger.info(f"Annotation DB Preparation Script Started.")
    logger.info(f"Output Directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    gencode_ready = False
    refseq_ready = False

    # --- Process GENCODE ---
    if not args.skip_gencode:
        gencode_gff_path = os.path.join(args.output_dir, args.gencode_gff_file)
        gencode_db_path = os.path.join(args.output_dir, "gencode_annotations.db")
        gff_exists = os.path.exists(gencode_gff_path)

        if not gff_exists:
            logger.info(f"GENCODE GFF file '{args.gencode_gff_file}' not found. Attempting download...")
            if args.gencode_url:
                gff_exists = download_file(args.gencode_url, gencode_gff_path)
            else:
                logger.error("GENCODE URL not provided. Cannot download.")

        if gff_exists:
            gencode_ready = create_gffutils_db(gencode_gff_path, gencode_db_path)
        else:
            logger.warning(f"Could not find or download GENCODE GFF file: {args.gencode_gff_file}")
            # Still check if DB exists from a previous run
            if os.path.exists(gencode_db_path):
                 logger.info(f"Using existing GENCODE DB: {gencode_db_path}")
                 gencode_ready = True
            else:
                 gencode_ready = False
    else:
        logger.info("Skipping GENCODE processing.")
        gencode_ready = True # Consider skipped as "ready" for exit code logic

    # --- Process RefSeq ---
    if not args.skip_refseq:
        refseq_gff_path = os.path.join(args.output_dir, args.refseq_gff_file)
        refseq_db_path = os.path.join(args.output_dir, "refseq_annotations.db")
        gff_exists = os.path.exists(refseq_gff_path)

        if not gff_exists:
            logger.info(f"RefSeq GFF file '{args.refseq_gff_file}' not found. Attempting download...")
            if args.refseq_url:
                gff_exists = download_file(args.refseq_url, refseq_gff_path)
            else:
                 logger.error("RefSeq URL not provided. Cannot download.")

        if gff_exists:
            refseq_ready = create_gffutils_db(refseq_gff_path, refseq_db_path)
        else:
            logger.warning(f"Could not find or download RefSeq GFF file: {args.refseq_gff_file}")
            # Still check if DB exists
            if os.path.exists(refseq_db_path):
                 logger.info(f"Using existing RefSeq DB: {refseq_db_path}")
                 refseq_ready = True
            else:
                 refseq_ready = False
    else:
        logger.info("Skipping RefSeq processing.")
        refseq_ready = True # Consider skipped as "ready"

    logger.info("Annotation DB Preparation Script Finished.")

    # Exit with error code if any required DB creation/download failed
    if not gencode_ready or not refseq_ready:
        logger.error("One or more required annotation databases could not be created or found/downloaded.")
        sys.exit(1) # Exit code 1 indicates failure
    else:
        logger.info("All available/required annotation databases are ready.")
        sys.exit(0) # Exit code 0 indicates success


if __name__ == "__main__":
    main()
