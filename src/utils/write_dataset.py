import logging
import pathlib

import pandas as pd
import pyarrow as pa
import os
from src.project_common import *

from src.core.stage import Stage


def write_dataset(stage: Stage, output_prefix: str, data: dict) -> bool:

    """
       Writes the given dataset (stored in the 'dataset' key of the data dict) to a parquet file
       in the processed data directory using fastparquet.

       The output filename is based on the provided prefix and stage name.

       :param stage: The Stage instance this data corresponds to.
       :param output_prefix: A string prefix used to construct the output file name.
       :param data: A dictionary expected to contain a 'dataset' key holding a pandas DataFrame.
       :return: True if the file was successfully written and confirmed to exist, False otherwise.
       :raises TypeError: If 'dataset' exists but is not a pandas DataFrame.
       """

    if data["dataset"] is None:
        return False

    if not type(data["dataset"]) is pd.DataFrame:
        raise TypeError("The 'dataset' field of input given is not a pandas DataFrame type.")

    output_filename = output_prefix + "_" + stage.name
    output_dir = pathlib.Path(processedDataDirectory)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    output_file = output_dir.joinpath(output_filename + processedDataExtensionType)
    output_file.touch()

    data['dataset'].to_parquet(output_file, engine='pyarrow')
    #fastparquet.write(output_file, data['dataset'])

    if output_file.exists():
        return True

    logging.error(f"Fastparquet wrote to file {str(output_file)} but it's existence could not be confirmed "
                  f"after writing. Check for problems.")
    return False
