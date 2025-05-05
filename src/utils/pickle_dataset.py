import pathlib
import pickle

import pandas as pd
from common.project_common import *

from core.stage import Stage

def pickle_dataset(stage: Stage, output_prefix: str, data: dict) -> pathlib.Path | None:

    output_filename = output_prefix + "_" + stage.name
    output_dir = pathlib.Path(PROCESSED_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / (output_filename + ".pkl")
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

    # data.to_pickle(output_file)

    return output_file if output_file.exists() else None




