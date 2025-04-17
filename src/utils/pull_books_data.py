import os
import random
import math
import json

from datasets import load_dataset

###
### WARNING:
###     This script downloads a MASSIVE amount of data from datasets. It takes a LONG time. Please be aware of both.
###

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "../books_data_asst_review")
os.makedirs(output_path, exist_ok=True)

output_suffix = ".jsonl"

# We are pulling 1000 records from the dataset
pull_total = 4000

# Max rating to filter on (e.g., 3.0 will include 1-3 star reviews only)
max_rating = 5.0

# And randomizing the records we select
randomize = True

# We are splitting this work among 4 members
divisions = 4

# Only use first 1 million entries. There are 29 million. You need this.
sample_size = 5_000_000

# We are using the Amazon Reviews dataset from https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
# To train our model, we are going to pull the first thousand reviews from the 'books' department of this dataset.
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Books", split="full", trust_remote_code=True)
selected = dataset.select(range(sample_size)).shuffle(seed=42)

# Filter and slice the dataset to get only relevant reviews
filtered_dataset = selected.filter(lambda x: x.get("rating", 0) <= max_rating)
selected_reviews = filtered_dataset.select(range(min(pull_total, len(filtered_dataset))))
print("Filtered dataset size:", len(filtered_dataset))
print("Sample entry:", filtered_dataset[0])  # Should be a dict

# Write to files
pulls_per_file = pull_total // divisions
for i in range(divisions):
    start = i * pulls_per_file
    end = (i + 1) * pulls_per_file if i < divisions - 1 else len(selected_reviews)
    out_file = os.path.join(output_path, f"{i}{output_suffix}")
    with open(out_file, "w", encoding="utf-8") as f:
        subset = selected_reviews.select(range(start, end))
        for review in subset:
            f.write(json.dumps(review) + "\n")
    print(f"Wrote {end - start} reviews to {out_file}")