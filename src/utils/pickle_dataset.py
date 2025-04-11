import pathlib

import pandas as pd
from common.project_common import *

from core.stage import Stage

def pickle_dataset(stage: Stage, output_prefix: str, data: dict) -> pathlib.Path | None:

    if data.get("dataset") is None:
        return None

    df = data["dataset"]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The 'dataset' field of input given is not a pandas DataFrame type.")

    output_filename = output_prefix + "_" + stage.name
    output_dir = pathlib.Path(PROCESSED_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / (output_filename + ".pkl")
    df.to_pickle(output_file)

    return output_file if output_file.exists() else None




