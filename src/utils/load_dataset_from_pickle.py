import pathlib
import pickle

import pandas as pd

from common.project_common import *
from core.stage import Stage


def load_dataset_from_pickle(stage: Stage, output_prefix: str, data: dict) -> dict:
    output_filename = output_prefix + "_" + stage.name
    input_path = pathlib.Path(PROCESSED_DATA_DIR) / (output_filename + ".pkl")

    if not input_path.exists():
        raise FileNotFoundError(f"Expected pickled dataset file not found at {str(input_path)}")

    logging.info(f"Loading pickled dataset from {str(input_path)}")
    with open(input_path, "rb") as f:
        df = pickle.load(f)
        data["dataset"] = df
    return data
