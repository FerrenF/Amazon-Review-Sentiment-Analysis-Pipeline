import logging
import pathlib
import pandas as pd
import fastparquet

from src.project_common import *
from src.core.stage import Stage


def load_dataset_from_parquet(stage: Stage, output_prefix: str, data: dict) -> dict:
    """
    Loads a processed dataset from a parquet file based on the stage and output prefix.

    :param stage: The Stage instance this data corresponds to.
    :param output_prefix: A string prefix used to locate the correct file.
    :param data: The input dictionary that will receive the loaded DataFrame.
    :return: The modified data dictionary with the 'dataset' key set.
    """

    output_filename = output_prefix + "_" + stage.name
    input_path = pathlib.Path(processedDataDirectory).joinpath(output_filename + processedDataExtensionType)

    if not input_path.exists():
        raise FileNotFoundError(f"Expected processed dataset file not found at {str(input_path)}")

    logging.info(f"Loading processed dataset from {str(input_path)}")

    try:
        df = pd.read_parquet(input_path, engine='pyarrow')
        data["dataset"] = df
    except Exception as e:
        logging.error(f"Failed to read parquet file at {str(input_path)}: {e}")
        raise

    return data
