import logging
import pathlib
from datetime import datetime

from core.step import Step
from utils import load_dataset_from_pickle, load_dataset_from_parquet
import pandas as pd

class LoadCheckpointIfExists(Step):
    def __init__(self, stage_name: str, checkpoint_dir: str = "processed", checkpoint_basename: str = "data"):
        """
        Checks for a checkpoint file (pickle or parquet) and loads it into the pipeline.

        :param stage name: The stage for which a checkpoint is being loaded.
        :param checkpoint_dir: Directory containing checkpoint files.
        :param checkpoint_basename: Base name of the checkpoint file (without stage name extension).
        """
        self.name = f"load_checkpoint_{checkpoint_basename}"
        self.description = "Loads a previously saved checkpoint to resume the pipeline early if possible."
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.checkpoint_basename = checkpoint_basename
        self.stage_name = stage_name

    def set_stats(self, data: dict):
        if 'time' not in data['stats']:
            data["stats"]["time"] = []
        data["stats"]["time"].append((self.name, datetime.now()))
        data["stats"]["checkpoint"] = {
            "used_checkpoint": False,
            "checkpoint_type": None
        }

    def run(self, data: dict) -> dict:
        pkl_path = self.checkpoint_dir / f"{self.checkpoint_basename}_{self.stage_name}.pkl"
        parquet_path = self.checkpoint_dir / f"{self.checkpoint_basename}_{self.stage_name}.parquet"

        checkpoint_path = None
        checkpoint_type = None

        if pkl_path.exists():
            checkpoint_path = pkl_path
            checkpoint_type = "pickle"
        elif parquet_path.exists():
            checkpoint_path = parquet_path
            checkpoint_type = "parquet"

        if checkpoint_path:
            self.step_log(f"Checkpoint found at {checkpoint_path}, loading...")

            try:
                if checkpoint_type == "pickle":
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                elif checkpoint_type == "parquet":                    
                    data["dataset"] = pd.read_parquet(parquet_path, engine='pyarrow')                   

                data["stats"]["checkpoint"]["checkpoint_type"] = checkpoint_type
                data["stats"]["checkpoint"]["used_checkpoint"] = True
                data["skip_to_stage"] = self.stage_name

            except Exception as e:
                self.step_log(f"Failed to load checkpoint: {e}")
                data["skip_to_stage"] = None
        else:
            self.step_log(f"No checkpoint found, proceeding with full execution.")
            data["skip_to_stage"] = None

        return data
