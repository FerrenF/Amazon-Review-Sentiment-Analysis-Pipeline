import os
import logging
from pathlib import Path

# Here lies variables that may be needed in multiple parts of the project.
# Keeping them in one place helps ensure ease of maintainability.

# Dynamically resolve the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Adjust depending on where this file is

OUTPUT_RAW_PATH = "./data/books_data/"


DATA_DIR = PROJECT_ROOT / "data"
UNPROCESSED_DATA_DIR = DATA_DIR / "unprocessed"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = PROJECT_ROOT / "output"

UNPROCESSED_EXT = ".jsonl"
PROCESSED_EXT = ".parquet"

LOGGING_NAME = "_log.txt"

PIPELINE_NAME = "amzn_review_sentiment"

