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

UNPROCESSED_EXT = ".jsonl"
PROCESSED_EXT = ".parquet"

# Logging
logging.basicConfig(
    level=logging.INFO,  # Can be DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s [%(levelname)s] %(message)s',
)
