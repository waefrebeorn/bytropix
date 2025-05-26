import argparse
import logging
import os
import random
from typing import List, Set, Any, Optional, Tuple, Dict

# Attempt to import mteb
try:
    import mteb
    from datasets import Dataset as HFDataset, IterableDataset as HFIterableDataset, DatasetDict as HFDatasetDict # For type checking
except ImportError:
    print("The 'mteb' library or 'datasets' library is not installed. Please install them with 'pip install mteb datasets'")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MTEB_to_Two_TXTs_Converter")

def sanitize_text(text: str) -> str:
    """Cleans text by stripping, replacing internal newlines/tabs with spaces, and ensuring no excessive spacing."""
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    cleaned = ' '.join(cleaned.split()) # Normalize multiple spaces to single space
    return cleaned

def save_texts_to_file(texts: List[str], output_filepath: str):
    """Saves a list of texts to a file, one sanitized text per line."""
    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir: 
             os.makedirs(output_dir, exist_ok=True)
        
        unique_sanitized_texts = {sanitize_text(text) for text in texts if text and text.strip()}
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for text in sorted(list(unique_sanitized_texts)): 
                if text: 
                    f.write(text + '\n')
        logger.info(f"Successfully saved {len(unique_sanitized_texts)} unique, sanitized texts to {output_filepath}")
    except Exception as e:
        logger.error(f"Error saving texts to {output_filepath}: {e}", exc_info=True)

def extract_texts_from_mteb_dataset_item(item: Any) -> List[str]:
    """Extracts texts if the item itself is a string or a dict containing known text keys."""
    texts_found: List[str] = []
    if isinstance(item, str):
        texts_found.append(item)
    elif isinstance(item, dict):
        # Prioritize common single text fields
        for key in ["text", "sentence", "query", "passage", "document", "content", "title"]:
            if key in item and isinstance(item[key], str):
                texts_found.append(item[key])
        # Then check for paired text fields
        for key1, key2 in [("sentence1", "sentence2"), ("query1", "query2"), ("anchor", "positive"), ("anchor", "negative")]:
            if key1 in item and isinstance(item[key1], str):
                texts_found.append(item[key1])
            if key2 in item and isinstance(item[key2], str):
                texts_found.append(item[key2])
        # Handle lists of texts within a dict field (e.g., multiple passages for a query)
        for key in ["positive", "negative", "passages", "documents", "corpus"]: # common list holding structures
            if key in item and isinstance(item[key], list):
                for sub_item in item[key]:
                    if isinstance(sub_item, str):
                        texts_found.append(sub_item)
                    elif isinstance(sub_item, dict) and "text" in sub_item and isinstance(sub_item["text"], str):
                         texts_found.append(sub_item["text"])
    return texts_found

def extract_texts_from_mteb_data_structure(dataset_split_data: Any, task_name_for_log: str) -> Set[str]:
    """More robustly extracts text from various MTEB dataset structures."""
    extracted_texts_set: Set[str] = set()

    if isinstance(dataset_split_data, (HFDataset, HFIterableDataset)):
        logger.debug(f"Task '{task_name_for_log}': Processing HF Dataset (type {type(dataset_split_data)}) with columns: {dataset_split_data.column_names}")
        
        # More comprehensive list of potential text columns
        potential_text_columns = [
            "text", "sentence", "sentence1", "sentence2",
            "query", "passage", "document", "documents", "content", "title",
            "anchor", "positive", "negative", "answers", "questions",
            "premise", "hypothesis", "claim", "evidence" 
            # Add more as identified from MTEB task structures
        ]
        # Columns that might contain lists of texts or dicts with text
        potential_list_or_dict_cols = ["passages", "positive_passages", "negative_passages", "corpus", "evidences"]


        # First, try to extract from known single-text or paired-text columns
        for col_name in dataset_split_data.column_names:
            if col_name in potential_text_columns:
                logger.debug(f"Task '{task_name_for_log}': Extracting from column: {col_name}")
                try:
                    for item_data in dataset_split_data[col_name]: # This iterates through the column directly
                        if isinstance(item_data, str):
                            extracted_texts_set.add(item_data)
                        elif isinstance(item_data, dict): # e.g. if a cell itself is a dict like {'text': '...'}
                            extracted_texts_set.update(extract_texts_from_mteb_dataset_item(item_data))
                        elif isinstance(item_data, list): # e.g. if a cell contains a list of passages
                             for sub_item_in_list in item_data:
                                extracted_texts_set.update(extract_texts_from_mteb_dataset_item(sub_item_in_list))

                except Exception as e:
                    logger.warning(f"Task '{task_name_for_log}': Error directly processing column '{col_name}'. Error: {e}. Will rely on row-wise iteration.")
        
        # Fallback to row-wise iteration to catch more complex structures or unlisted columns
        logger.debug(f"Task '{task_name_for_log}': Performing general row-wise extraction.")
        for row_item in dataset_split_data: # row_item is usually a dict
            extracted_texts_set.update(extract_texts_from_mteb_dataset_item(row_item))


    elif isinstance(dataset_split_data, dict): # e.g., BEIR format {'corpus': {id: text}, 'queries': {id: text}}
        logger.debug(f"Task '{task_name_for_log}': Processing generic dict. Keys: {list(dataset_split_data.keys())}")
        for key, value in dataset_split_data.items():
            if isinstance(value, dict): # e.g. corpus: {doc_id: text_or_dict}, queries: {q_id: text}
                for sub_key, sub_value in value.items():
                    extracted_texts_set.update(extract_texts_from_mteb_dataset_item(sub_value))
            elif isinstance(value, list): # e.g. list of dicts for classification/STS
                for item in value:
                    extracted_texts_set.update(extract_texts_from_mteb_dataset_item(item))
            elif isinstance(value, str): # Direct text value
                 extracted_texts_set.add(value)
    elif isinstance(dataset_split_data, list):
        logger.debug(f"Task '{task_name_for_log}': Processing generic list of length {len(dataset_split_data)}.")
        for item in dataset_split_data:
            extracted_texts_set.update(extract_texts_from_mteb_dataset_item(item))
    else:
        logger.warning(f"Task '{task_name_for_log}': Unrecognized data structure type: {type(dataset_split_data)}.")

    return extracted_texts_set

def get_mteb_task_objects_for_corpus(
    explicit_task_names: List[str],
    benchmark_names_for_fallback: List[str],
    target_lang_codes: List[str],
    corpus_label: str
) -> List[Any]:
    task_objects_for_corpus: List[Any] = []

    # Normalize target language codes (e.g., 'en' -> 'eng')
    normalized_target_langs = []
    if target_lang_codes:
        for lang_code in target_lang_codes:
            lc = lang_code.lower()
            if lc == "en": normalized_target_langs.append("eng")
            # Add more normalizations if needed (e.g., "de" -> "deu")
            else: normalized_target_langs.append(lc)
    logger.info(f"Corpus {corpus_label}: Normalized target languages for matching: {normalized_target_langs}")

    if explicit_task_names:
        logger.info(f"Corpus {corpus_label}: Using explicitly provided task names: {explicit_task_names}")
        # Use mteb.get_tasks for robust loading and language filtering
        try:
            task_objects_for_corpus = mteb.get_tasks(tasks=explicit_task_names, languages=target_lang_codes if target_lang_codes else None)
            logger.info(f"Corpus {corpus_label}: Successfully fetched {len(task_objects_for_corpus)} task objects for explicit names (after MTEB's lang filter).")
            
            # Additional check if mteb.get_tasks didn't filter perfectly or if lang names are tricky
            if target_lang_codes and task_objects_for_corpus:
                filtered_objects = []
                for task_obj in task_objects_for_corpus:
                    task_meta_langs_lower = [ln.lower() for ln in task_obj.metadata.languages]
                    if any(norm_target_ln in task_meta_langs_lower for norm_target_ln in normalized_target_langs):
                        filtered_objects.append(task_obj)
                    else:
                        logger.warning(f"Corpus {corpus_label}: Task '{task_obj.metadata.name}' (Langs: {task_obj.metadata.languages}) removed after explicit language check against target: {target_lang_codes}.")
                task_objects_for_corpus = filtered_objects
                logger.info(f"Corpus {corpus_label}: Refined to {len(task_objects_for_corpus)} task objects after explicit language check.")

        except Exception as e:
            logger.error(f"Corpus {corpus_label}: Error fetching explicit tasks {explicit_task_names} via mteb.get_tasks: {e}. Will try one by one.", exc_info=True)
            # Fallback to trying one by one for more granular error reporting if mteb.get_tasks fails
            task_objects_for_corpus = []
            for task_name_candidate in explicit_task_names:
                try:
                    task_obj = mteb.get_task(task_name_candidate)
                    lang_match = not normalized_target_langs or \
                                 any(norm_ln in [meta_ln.lower() for meta_ln in task_obj.metadata.languages] for norm_ln in normalized_target_langs)
                    if lang_match:
                        task_objects_for_corpus.append(task_obj)
                    else:
                        logger.warning(f"Corpus {corpus_label}: Explicit task '{task_name_candidate}' (Langs: {task_obj.metadata.languages}) skipped: language mismatch with target {target_lang_codes}.")
                except Exception as e_single:
                    logger.warning(f"Corpus {corpus_label}: Could not get explicit task '{task_name_candidate}'. Error: {e_single}")
            logger.info(f"Corpus {corpus_label}: Fetched {len(task_objects_for_corpus)} task objects via individual get_task calls.")


    elif benchmark_names_for_fallback:
        logger.info(f"Corpus {corpus_label}: No explicit tasks. Getting tasks from benchmarks: {benchmark_names_for_fallback} for languages: {target_lang_codes}")
        all_benchmark_tasks_temp = []
        for benchmark_name in benchmark_names_for_fallback:
            try:
                benchmark_task_list_or_config = mteb.get_benchmark(benchmark_name, languages=target_lang_codes if target_lang_codes else None)
                if benchmark_task_list_or_config: # This should be a list of task objects or dicts that MTEB can process
                    # If get_benchmark returns task names or configs, they might need to be instantiated by mteb.get_tasks
                    # For MTEB 1.38.x, get_benchmark should return list of task objects directly
                    if all(hasattr(t, 'metadata') for t in benchmark_task_list_or_config): # Check if they look like task objects
                        all_benchmark_tasks_temp.extend(benchmark_task_list_or_config)
                    else: # If it's a list of names/configs, pass through mteb.get_tasks
                        tasks_from_bm_config = mteb.get_tasks(tasks=benchmark_task_list_or_config, languages=target_lang_codes if target_lang_codes else None)
                        all_benchmark_tasks_temp.extend(tasks_from_bm_config)
                    logger.info(f"Corpus {corpus_label}: Added tasks from benchmark '{benchmark_name}'.")
                else:
                    logger.warning(f"Corpus {corpus_label}: Benchmark '{benchmark_name}' returned no tasks/configs.")
            except Exception as e:
                logger.warning(f"Corpus {corpus_label}: Could not get tasks for benchmark '{benchmark_name}'. Error: {e}")
        
        unique_tasks_by_name = {task.metadata.name: task for task in all_benchmark_tasks_temp if hasattr(task, 'metadata')}
        task_objects_for_corpus = list(unique_tasks_by_name.values())
        logger.info(f"Corpus {corpus_label}: Total unique task objects from benchmarks: {len(task_objects_for_corpus)}")
    else:
        logger.warning(f"Corpus {corpus_label}: Neither explicit tasks nor benchmarks provided. No MTEB tasks will be sourced.")

    if not task_objects_for_corpus:
        logger.warning(f"Corpus {corpus_label}: No MTEB task objects to process based on criteria.")
    
    return task_objects_for_corpus


def process_corpus(
    corpus_label: str,
    mteb_task_objects_to_process: List[Any], 
    target_langs: List[str], # Mainly for logging, filtering should have happened
    splits_preference: List[str], 
    max_total_texts: int,
    output_filepath: str,
    shuffle_output: bool
):
    logger.info(f"===== Processing Corpus {corpus_label} =====")
    
    if not mteb_task_objects_to_process:
        logger.warning(f"No MTEB task objects provided for Corpus {corpus_label}. Skipping.")
        return

    corpus_collected_texts: Set[str] = set()

    for task_idx, task in enumerate(mteb_task_objects_to_process):
        task_name = task.metadata.name
        task_type = task.metadata.type.lower()
        logger.info(f"Corpus {corpus_label}: Processing task {task_idx+1}/{len(mteb_task_objects_to_process)}: {task_name} (Type: {task_type}, Actual Task Langs: {task.metadata.languages})")

        try:
            task.load_data() 
            logger.debug(f"Corpus {corpus_label}, Task {task_name}: task.load_data() called successfully.")

            texts_found_in_this_task = False
            potential_data_sources: List[Tuple[str, Any]] = [] 

            # 1. Check task attributes for loaded data
            # For Retrieval: task.corpus (DatasetDict/Dataset), task.queries (DatasetDict/Dataset)
            # For others: task.dataset (DatasetDict/Dataset)
            
            sources_to_investigate : Dict[str, Any] = {}
            if hasattr(task, "corpus") and task.corpus is not None:
                sources_to_investigate["corpus"] = task.corpus
            if hasattr(task, "queries") and task.queries is not None:
                 sources_to_investigate["queries"] = task.queries
            if hasattr(task, "dataset") and task.dataset is not None: # This might overlap with corpus/queries for some tasks
                 sources_to_investigate["dataset"] = task.dataset

            if not sources_to_investigate:
                logger.warning(f"Corpus {corpus_label}, Task {task_name}: No data attributes (corpus, queries, dataset) found or they are None. Skipping task.")
                continue

            for source_attr_name, data_container in sources_to_investigate.items():
                if isinstance(data_container, (HFDataset, HFIterableDataset)): # Direct dataset
                    potential_data_sources.append((f"{source_attr_name}-main", data_container))
                elif isinstance(data_container, HFDatasetDict): # Dict of datasets (splits)
                    for split_pref in splits_preference:
                        if split_pref in data_container and data_container[split_pref] is not None:
                            potential_data_sources.append((f"{source_attr_name}-{split_pref}", data_container[split_pref]))
                            logger.debug(f"Added {source_attr_name}-{split_pref} for task {task_name}")
                            # Often, we just need one good split from each container (corpus, queries, dataset)
                            # If specific logic for train/test/dev is needed per container, this needs adjustment.
                            # For now, take the first preferred split found in each container.
                            break 
                    else: # If no preferred splits found, try any split
                        if data_container: # If HFDatasetDict is not empty
                            first_available_split = next(iter(data_container))
                            if data_container[first_available_split] is not None:
                                potential_data_sources.append((f"{source_attr_name}-{first_available_split}", data_container[first_available_split]))
                                logger.debug(f"Using first available split '{first_available_split}' from {source_attr_name} for task {task_name}")
                elif isinstance(data_container, (list, dict)) and data_container: # Raw list/dict from less standard tasks
                     potential_data_sources.append((f"{source_attr_name}-raw", data_container))


            if not potential_data_sources:
                logger.warning(f"Corpus {corpus_label}, Task {task_name}: Could not identify usable data sources from task attributes. Skipping.")
                continue
            
            unique_data_object_ids = set() # To avoid processing the exact same HF Dataset object multiple times
            for source_desc, data_object in potential_data_sources:
                if data_object is None: continue
                obj_id = id(data_object)
                if obj_id in unique_data_object_ids:
                    logger.debug(f"Skipping already processed data object for source '{source_desc}' in task '{task_name}'.")
                    continue
                unique_data_object_ids.add(obj_id)

                logger.info(f"Corpus {corpus_label}, Task {task_name}: Extracting from source '{source_desc}'.")
                texts_from_source_set = extract_texts_from_mteb_data_structure(data_object, f"{task_name}-{source_desc}")
                
                if texts_from_source_set:
                    num_previously_collected = len(corpus_collected_texts)
                    corpus_collected_texts.update(texts_from_source_set)
                    num_newly_added = len(corpus_collected_texts) - num_previously_collected
                    logger.info(f"Corpus {corpus_label}, Task {task_name}, Source '{source_desc}': Found {len(texts_from_source_set)} texts. Added {num_newly_added} new (Total for corpus: {len(corpus_collected_texts)}).")
                    texts_found_in_this_task = True
            
            if not texts_found_in_this_task:
                logger.warning(f"Corpus {corpus_label}, Task {task_name}: No texts extracted from any identified sources/attributes.")

            if max_total_texts > 0 and len(corpus_collected_texts) >= max_total_texts:
                logger.info(f"Corpus {corpus_label}: Reached max_total_texts limit ({max_total_texts}). Stopping.")
                break
        except Exception as e:
            logger.error(f"Corpus {corpus_label}: Error processing task {task_name}: {e}", exc_info=True)
            
    final_texts_to_save = list(corpus_collected_texts)
    if shuffle_output:
        logger.info(f"Corpus {corpus_label}: Shuffling {len(final_texts_to_save)} collected texts.")
        random.shuffle(final_texts_to_save)

    if max_total_texts > 0 and len(final_texts_to_save) > max_total_texts:
        logger.info(f"Corpus {corpus_label}: Limiting total unique texts from {len(final_texts_to_save)} to {max_total_texts}.")
        final_texts_to_save = final_texts_to_save[:max_total_texts]
            
    if final_texts_to_save:
        save_texts_to_file(final_texts_to_save, output_filepath)
    else:
        logger.warning(f"Corpus {corpus_label}: No texts collected. Output file '{output_filepath}' will be empty or not created.")
    logger.info(f"===== Finished Processing Corpus {corpus_label}. Total unique texts for corpus: {len(corpus_collected_texts)}, Saved: {len(final_texts_to_save)} =====")


def main():
    parser = argparse.ArgumentParser(description="Convert MTEB task datasets to plain text files for Corpora A and B.")
    
    default_tasks_A = [ 
        "Banking77Classification", "EmotionClassification", "ImdbClassification", 
        "AmazonReviewsClassification", # MTEB should handle 'en' via languages arg to get_tasks
        "STSBenchmark", "STS12", "SICK-R",
        "NFCorpus", "SciFact", "FEVER",
        "SprintDuplicateQuestions", "TwitterSemEval2015", "AskUbuntuDupQuestions"
    ]
    default_tasks_B = [ 
        "ArxivClassification", "DBpediaClassification", "YahooAnswersTopicsClassification",
        "MassiveIntentClassification", 
        "STS13", "STS14", "STS15",
        "ArguAna", "MSMARCO", "HotpotQA", "QuoraRetrieval", "TRECCOVID",
        "TwitterURLCorpus", "PawsXPairClassification" 
    ]

    parser.add_argument("--tasks_A", nargs="*", default=default_tasks_A, help="MTEB task names for Corpus A.")
    parser.add_argument("--benchmark_keywords_A", nargs="*", default=[], help="Fallback keywords to filter MTEB tasks for Corpus A if --tasks_A is empty.")
    parser.add_argument("--output_file_A", type=str, default="mteb_corpus_A.txt")
    parser.add_argument("--max_texts_A", type=int, default=150000)

    parser.add_argument("--tasks_B", nargs="*", default=default_tasks_B, help="MTEB task names for Corpus B.")
    parser.add_argument("--benchmark_keywords_B", nargs="*", default=[], help="Fallback keywords to filter MTEB tasks for Corpus B if --tasks_B is empty.")
    parser.add_argument("--output_file_B", type=str, default="mteb_corpus_B.txt")
    parser.add_argument("--max_texts_B", type=int, default=150000)

    parser.add_argument("--splits", nargs="+", default=["train", "test", "validation", "dev"])
    parser.add_argument("--task_langs", nargs="+", default=["en"])
    parser.add_argument("--shuffle_output", action="store_true", default=True)

    args = parser.parse_args()

    if args.output_file_A:
        task_objects_A = get_mteb_task_objects_for_corpus(args.tasks_A, args.benchmark_keywords_A, args.task_langs, "A")
        process_corpus("A", task_objects_A, args.task_langs, args.splits, args.max_texts_A, args.output_file_A, args.shuffle_output)

    if args.output_file_B:
        task_objects_B = get_mteb_task_objects_for_corpus(args.tasks_B, args.benchmark_keywords_B, args.task_langs, "B")
        process_corpus("B", task_objects_B, args.task_langs, args.splits, args.max_texts_B, args.output_file_B, args.shuffle_output)

    logger.info("MTEB to TXT (A & B) conversion process finished.")

if __name__ == "__main__":
    main()