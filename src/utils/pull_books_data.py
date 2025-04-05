import os
import random
import math
import json

from datasets import load_dataset

outputPath = os.path.join(os.path.abspath(__file__), "../books_data")
outputSuffix = ".jsonl"

# We are pulling 1000 records from the dataset
pull_total = 1000

# And randomizing the records we select
randomize = True

# We are splitting this work among 4 members
divisions = 4

# We are using the Amazon Reviews dataset from https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
# To train our model, we are going to pull the first thousand reviews from the 'books' department of this dataset.
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Books", split="full", trust_remote_code=True)

filesOut = list()
for i in range(divisions):
    # Open each file we need to append to
    filesOut.append(open(outputPath+str(i)+outputSuffix, "w+", encoding="utf-8"))

pullsPerFile = pull_total / divisions
pulled = 0
selected = set()
if len(filesOut):
    while pulled < pull_total:
        target = random.randint(0, len(dataset) - 1)
        if target in selected:
            continue
        selected.add(target)

        targetFileNum = math.floor(pulled / pullsPerFile)
        targetFileOut = filesOut[targetFileNum]
        if targetFileOut:
            targetFileOut.write(json.dumps(dataset[target]) + '\n')
        else:
            raise IndexError

        if pulled % int(pullsPerFile) == 0:
            print(f"Working on file {targetFileNum}, processed {pulled} records so far out of {pull_total}.")
        pulled += 1

for f in filesOut:
    f.close()

