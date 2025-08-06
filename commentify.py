import argparse
import os

def add_comment_prefix(input_path, output_path, prefix):
    """
    Reads a file line-by-line, adds a prefix, and writes to a new file.
    This is memory-efficient and works for large files.
    """
    print(f"Processing '{input_path}'...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            line_count = 0
            for line in f_in:
                # The `line` from the file iterator includes the newline character `\n`.
                # Prepending the prefix here preserves the original line endings and spacing.
                f_out.write(f"{prefix}{line}")
                line_count += 1
        
        print(f"Success! Processed {line_count} lines.")
        print(f"Commented version saved to '{output_path}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    """
    Parses command-line arguments and runs the main script logic.
    """
    parser = argparse.ArgumentParser(
        description="A script to add a prefix to every line of a text file, effectively commenting it out.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input_file",
        help="The path to the input text file (e.g., tinyshakespeare.txt)."
    )

    parser.add_argument(
        "-o", "--output",
        default=None,
        help="The path for the output file. If not provided, defaults to 'commented_[input_file_name]'."
    )

    parser.add_argument(
        "-p", "--prefix",
        default="# ",
        help="The string to prepend to each line."
    )

    args = parser.parse_args()

    # If no output file is specified, create a default name.
    output_file = args.output
    if output_file is None:
        base_name = os.path.basename(args.input_file)
        output_file = f"commented_{base_name}"

    add_comment_prefix(args.input_file, output_file, args.prefix)

if __name__ == "__main__":
    main()