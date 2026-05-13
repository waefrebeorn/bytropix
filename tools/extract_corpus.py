#!/usr/bin/env python3
"""
Extract NARRATIVE_TEXT from CORPUS.py using ast parsing.
"""
import ast
import re

input_path = "ENCODERS/phase3-generative/CORPUS.py"
output_path = "data/corpus_raw.txt"

# Read the file
with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

# Strategy: find all dict assignments by looking for lines like VAR_NAME = {
# then use regex to find NARRATIVE_TEXT values within each dict
# Actually, the simplest robust approach is to find all "NARRATIVE_TEXT": followed by
# a string, handling escaped quotes properly.

# Let's use a line-by-line approach combining each dict block
# and using ast.literal_eval for the value

# Find all NARRATIVE_TEXT occurrences and their values by tracking braces
narratives = []

# Simple line-based parser
lines = content.split('\n')
i = 0
while i < len(lines):
    line = lines[i]
    # Look for "NARRATIVE_TEXT":
    idx = line.find('"NARRATIVE_TEXT"')
    if idx >= 0:
        # Found it - find the colon
        colon = line.find(':', idx)
        if colon >= 0:
            # Extract the value part (after colon)
            value_part = line[colon+1:].strip()
            
            # Determine quote style
            if value_part.startswith('"""'):
                # Triple-quoted - need to find closing """
                value_part = value_part[3:]
                full_text = value_part
                # If closing """ is on same line or later lines
                while '"""' not in full_text and i < len(lines):
                    i += 1
                    if i < len(lines):
                        full_text += '\n' + lines[i]
                val = full_text[:full_text.index('"""')]
                narratives.append(val.strip())
            elif value_part.startswith("'''"):
                value_part = value_part[3:]
                full_text = value_part
                while "'''" not in full_text and i < len(lines):
                    i += 1
                    if i < len(lines):
                        full_text += '\n' + lines[i]
                val = full_text[:full_text.index("'''")]
                narratives.append(val.strip())
            elif value_part.startswith("'") or value_part.startswith('"'):
                quote = value_part[0]
                # Single-line string
                rest = value_part[1:]
                # Find the closing quote, handling escapes
                j = 0
                while j < len(rest):
                    if rest[j] == '\\' and j+1 < len(rest):
                        j += 2
                        continue
                    if rest[j] == quote:
                        val = rest[:j]
                        narratives.append(val.strip())
                        break
                    j += 1
    i += 1

print(f"Extracted {len(narratives)} narrative texts")

with open(output_path, "w", encoding="utf-8") as f:
    for i, text in enumerate(narratives):
        if i > 0:
            f.write("\n")
        f.write(text)

total_chars = sum(len(t) for t in narratives)
print(f"Written to {output_path}")
print(f"Total characters: {total_chars}")
