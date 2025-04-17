import os
import json


def merge_jsonl_files(input_dir):
    # Ensure input is a directory
    if not os.path.isdir(input_dir):
        raise ValueError(f"{input_dir} is not a valid directory.")

    parent_dir = os.path.abspath(os.path.join(input_dir, os.pardir))
    merged_file_path = os.path.join(parent_dir, "merged_output.jsonl")

    with open(merged_file_path, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_dir):
            if filename.endswith(".jsonl"):
                file_path = os.path.join(input_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        line = line.strip()
                        if line:  # Avoid empty lines
                            try:
                                json_obj = json.loads(line)
                                outfile.write(json.dumps(json_obj) + "\n")
                            except json.JSONDecodeError as e:
                                print(f"Skipping invalid JSON line in {filename}: {e}")

    print(f"Merged file saved to: {merged_file_path}")
