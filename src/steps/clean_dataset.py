import pathlib
from datetime import datetime

from core.step import Step
from common.project_common import *


class CleanDatasetStep(Step):
    name = "clean_dataset"

    def set_stats(self, data: dict):
        if 'time' not in data['stats']:
            data["stats"]["time"] = list()
        data["stats"]["time"].append((self.name, datetime.now()))
        data["stats"]["wiped"] = True
        return data

    def run(self, data: dict) -> dict:

        """
        This step cleans data which has been written for each stage the pipeline during previous runs.
        *It --deletes-- all (parquet files) and (pickles) in the processed data directory. Beware.*

        :param data: A dict, which may or may not be empty. It's ignored.
        :return: The same input dict. With everything else secretly missing.
        """

        path_object = pathlib.Path(PROCESSED_DATA_DIR)
        if not path_object.exists():
            path_object.mkdir(parents=True, exist_ok=True)

        # Iterate through objects in the processed data directory.

        # Catch .parquet files.
        for child in path_object.iterdir():
            if child.match("*" + PROCESSED_EXT):
                self.step_log(f"DELETED: {str(child)}")
                child.unlink(True)

        # Catch .pkl files.
        for child in path_object.iterdir():
            if child.match("*" + ".pkl"):
                self.step_log(f"DELETED: {str(child)}")
                child.unlink(True)

        return data
