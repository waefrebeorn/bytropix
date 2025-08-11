# corpus_converter.py
# An Alchemical Transmuter: This script reads an old-format corpus file
# (e.g., backupcorpus.py) and converts its contents into the new, structured
# format, ready to be merged with data from the corpus_builder.

import importlib.util
import sys
import textwrap
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "backupcorpus.py"
OUTPUT_FILE = "modifiedbackupcorpus.py"

# --- HELPER FUNCTIONS (from the main trainer) ---

def clean_text_for_fstring(text: str) -> str:
    """Cleans text to be safely embedded in a Python triple-quoted f-string."""
    return text.strip().replace('"""', '\\"\\"\\"')

def distill_text_from_corpus(data: any) -> str:
    """Recursively extracts all string values from a nested data structure."""
    if isinstance(data, str):
        return data + "\n"
    elif isinstance(data, dict):
        return "".join(distill_text_from_corpus(v) for v in data.values())
    elif isinstance(data, list):
        return "".join(distill_text_from_corpus(item) for item in data)
    return ""

# --- NEW FORMATTERS ---

def format_lore_entry(name, value, index):
    """Formats a generic, non-conversational lore entry."""
    variable_name = f"LORE_{name}_{index}"
    distilled_content = clean_text_for_fstring(distill_text_from_corpus(value))
    
    return (
        f'{variable_name} = {{\n'
        f'    "TASK_TYPE": "Foundational Text",\n'
        f'    "SOURCE": "{name}",\n'
        f'    "NARRATIVE_TEXT": """{distilled_content}"""\n'
        f'}}\n\n'
    )

def format_dialogus_entry(entry, index):
    """Formats a conversational entry from the Dialogus Exempla corpus."""
    variable_name = f"LORE_DIALOGUS_{index}"
    instruction = clean_text_for_fstring(entry["user_prompt"])
    response = clean_text_for_fstring(entry["assistant_response"])
    source_name = entry.get("name", "Dialogus Exempla")
    
    return (
        f'{variable_name} = {{\n'
        f'    "TASK_TYPE": "Instruction Following",\n'
        f'    "SOURCE": "{source_name}",\n'
        f'    "INSTRUCTION": """{instruction}""",\n'
        f'    "RESPONSE": """{response}"""\n'
        f'}}\n\n'
    )

# --- MAIN CONVERSION LOGIC ---

def convert_corpus():
    """Reads the input file, converts its data, and writes to the output file."""
    print(f"--- Corpus Converter Initialized ---")
    print(f"Reading old-format data from '{INPUT_FILE}'...")

    try:
        # Dynamically import the user's corpus file
        spec = importlib.util.spec_from_file_location("old_corpus", INPUT_FILE)
        old_corpus = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_corpus)
    except FileNotFoundError:
        print(f"\n[FATAL ERROR] Input file '{INPUT_FILE}' not found.")
        print("Please rename your old corpus file to 'backupcorpus.py'.")
        return
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not load '{INPUT_FILE}'. Error: {e}")
        return

    # Find all ALL_CAPS variables to process
    corpus_vars = {name: getattr(old_corpus, name) for name in dir(old_corpus) if not name.startswith('_') and name.isupper()}
    
    if not corpus_vars:
        print(f"\n[ERROR] No ALL_CAPS variables found in '{INPUT_FILE}'. Nothing to convert.")
        return

    print(f"Found {len(corpus_vars)} top-level variables. Converting to new format...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# modifiedbackupcorpus.py\n")
        f.write(f"# Auto-converted from '{INPUT_FILE}'.\n")
        f.write("# Copy the contents of this file into your main CORPUS.PY to merge.\n\n")

        global_index = 0
        for name, value in tqdm(corpus_vars.items(), desc="Converting Lore"):
            # Special handling for the conversational corpus
            if "DIALOGUS_EXEMPLA" in name:
                # This assumes a structure like: LIST -> DICT -> 'conversation_templates' -> LIST of DICTS
                try:
                    # Handle if it's a list of dicts or just a dict
                    templates = value[0]['conversation_templates'] if isinstance(value, list) else value['conversation_templates']
                    for template in templates:
                        f.write(format_dialogus_entry(template, global_index))
                        global_index += 1
                except (KeyError, TypeError, IndexError) as e:
                    print(f"\n[WARNING] Could not parse conversational structure in '{name}'. Treating as generic lore. Error: {e}")
                    f.write(format_lore_entry(name, value, global_index))
                    global_index += 1
            # Generic handler for all other lore files
            else:
                f.write(format_lore_entry(name, value, global_index))
                global_index += 1
                
    print(f"\n--- Success! ---")
    print(f"Converted data has been written to '{OUTPUT_FILE}'.")
    print("You can now copy its contents and paste them into your main CORPUS.PY file.")

if __name__ == "__main__":
    convert_corpus()