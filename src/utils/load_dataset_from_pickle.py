import pathlib
import pickle

import pandas as pd

from common.project_common import *
from core.stage import Stage


def load_dataset_from_pickle( output_prefix: str, data: dict,stage: Stage = None, debug=False) -> dict:
    output_filename = output_prefix + (("_" + stage.name) if stage is not None else "")
    input_path = pathlib.Path(PROCESSED_DATA_DIR) / (output_filename + ".pkl")

    if not input_path.exists():
        raise FileNotFoundError(f"Expected pickled dataset file not found at {str(input_path)}")

    if debug:
        logging.info(f"Loading pickled dataset from {str(input_path)}")

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    if debug:
        logging.info("Loaded keys: " + ",".join(data.keys()))
    return data

