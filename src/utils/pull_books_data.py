import os
import random
import math
import json

from datasets import load_dataset

###
### WARNING:
###     This script downloads a MASSIVE amount of data from the datasets library.
###     It takes a LONG time the first time you run it.
###

script_dir = os.path.dirname(os.path.abspath(__file__))
output_suffix = ".jsonl"


pull_total = 25000
randomize = True

# We are splitting this work among 4 members, so 4 divisions
divisions = 1

# Only use first 5 million entries. There are 29 million. You need this.
sample_size = 5_000_000

# We are using the Amazon Reviews dataset from https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
# To train our model, we are going to pull the first thousand reviews from the 'books' department of this dataset.
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Books", split="full", trust_remote_code=True)
selected = dataset.select(range(sample_size)).shuffle(seed=42)

def pull_filtered_reviews(_count, _max_rating, _divisions, _output_path):
    # Filter and slice the dataset to get only relevant reviews
    filtered_dataset = selected.filter(lambda x: x.get("rating", 0) <= _max_rating)
    selected_reviews = filtered_dataset.select(range(min(_count, len(filtered_dataset))))
    print("Filtered dataset size:", len(filtered_dataset))
    print("Sample entry:", filtered_dataset[0])  # Should be a dict

    # Write to files
    pulls_per_file = _count // divisions
    for i in range(_divisions):
        start = i * pulls_per_file
        end = (i + 1) * pulls_per_file if i < divisions - 1 else len(selected_reviews)

        os.makedirs(_output_path, exist_ok=True)
        out_file = os.path.join(_output_path, f"{i}{output_suffix}")

        with open(out_file, "w", encoding="utf-8") as f:
            subset = selected_reviews.select(range(start, end))
            for review in subset:
                f.write(json.dumps(review) + "\n")
        print(f"Wrote {end - start} reviews to {out_file}")


# Pull a set of both high and lower rated reviews to try and provide pre-balancing
pull_filtered_reviews(pull_total, 3, divisions, os.path.join(script_dir, "../books_data_low_review_25k"))
pull_filtered_reviews(pull_total, 5, divisions, os.path.join(script_dir, "../books_data_review_asst_25k"))