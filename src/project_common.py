import os
import logging

# Here lies variables that may be needed in multiple parts of the project.
# Keeping them in one place helps ensure ease of maintainability.

script_dir = os.path.dirname(os.path.abspath(__file__))
outputPath = os.path.join(script_dir, "books_data/")

unprocessedDataDirectory = os.path.join(script_dir, "data/unprocessed/")
unprocessedDataExtensionType = ".jsonl"

processedDataDirectory = os.path.join(script_dir, "data/processed/")
processedDataExtensionType = ".parquet"

# Logging
logging.basicConfig(
    level=logging.INFO,  # Can be DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s [%(levelname)s] %(message)s',
)
