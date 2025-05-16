import pathlib
import pandas as pd

from common.project_common import *
from core.stage import Stage


def load_dataset_from_parquet( output_prefix: str, data: dict,stage: Stage=None, debug=False) -> dict:
    output_filename = output_prefix + (("_" + stage.name) if stage is not None else "")
    input_path = pathlib.Path(PROCESSED_DATA_DIR) / (output_filename + PROCESSED_EXT)

    if not input_path.exists():
        raise FileNotFoundError(f"Expected processed dataset file not found at {str(input_path)}")

    if debug:
        logging.info(f"Loading processed dataset from {str(input_path)}")

    df = pd.read_parquet(input_path, engine='pyarrow')
    data["dataset"] = df
    return data
