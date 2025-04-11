import pathlib

from core.step import Step
from common.project_common import *


class CleanDatasetStep(Step):
    name = "clean_dataset"

    def run(self, data: dict) -> dict:
        """
        This step cleans the data which has been written for each stage of the project during previous runs.
        That is, it deletes all the parquet files in the processed data directory. Beware.

        :param data: A dict, which may or may not be empty. It's ignored.
        :return: The same input dict.
        """

        path_object = pathlib.Path(PROCESSED_DATA_DIR)
        if not path_object.exists():
            raise FileNotFoundError(f"Input path for unprocessed data at {UNPROCESSED_DATA_DIR} does not exist.")

        # Iterate through objects in the processed data directory.
        for child in path_object.iterdir():

            # If the child's extension matches .parquet, then it's processed data. Terminate it.
            if child.match("*" + PROCESSED_EXT):
                logging.info(f"Deleting processed data parquet: {str(child)}")
                child.unlink(True)

        return data
