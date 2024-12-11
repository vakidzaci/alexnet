import os
from collections import defaultdict

def filter_lines_by_sentence_length(input_file, output_file, max_words, records_per_length):
    """
    Filters lines from an input file to an output file based on the number of words in the sentence.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        max_words (int): Maximum sentence length to consider (inclusive).
        records_per_length (int): Number of records to retain for each sentence length.
    """
    # Dictionary to store filtered lines for each sentence length
    sentence_length_dict = defaultdict(list)

    # Read and process the input file
    with open(input_file, "r") as infile:
        for line in infile:
            parts = line.strip().split(" ", 1)  # Split into path and text
            if len(parts) < 2:
                continue  # Skip lines without proper format

            image_path, text = parts[0], parts[1]
            sentence_length = len(text.split())

            # Add the line to the appropriate length bucket
            if 1 <= sentence_length <= max_words:
                sentence_length_dict[sentence_length].append(line)

    # Ensure we meet the required number of records for each sentence length
    total_records = 0
    with open(output_file, "w") as outfile:
        for sentence_length in range(1, max_words + 1):
            lines = sentence_length_dict[sentence_length]

            # Write exactly `records_per_length` lines for this sentence length
            selected_lines = lines[:records_per_length]
            outfile.writelines(selected_lines)
            total_records += len(selected_lines)

    print(f"Filtered {total_records} records written to {output_file}")

# Example usage
if __name__ == "__main__":
    input_file = "file.txt"  # Replace with the path to your input file
    output_file = "new_file.txt"  # Replace with the desired output file path

    max_words = 21
    records_per_length = 5000

    filter_lines_by_sentence_length(input_file, output_file, max_words, records_per_length)
    print(f"Processing complete. Check {output_file} for results.")
