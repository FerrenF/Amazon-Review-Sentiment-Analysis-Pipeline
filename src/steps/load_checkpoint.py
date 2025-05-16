import logging
import pathlib
from datetime import datetime

from core.step import Step
from utils import load_dataset_from_pickle, load_dataset_from_parquet


class LoadCheckpointIfExists(Step):
    def __init__(self, stage_name: str, checkpoint_dir: str = "processed", checkpoint_basename: str = "data_checkpoint"):
        """
        Checks for a checkpoint file (pickle or parquet) and loads it into the pipeline.

        :param checkpoint_dir: Directory containing checkpoint files.
        :param checkpoint_basename: Base name of the checkpoint file (without extension).
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
        pkl_path = self.checkpoint_dir / f"{self.checkpoint_basename}.pkl"
        parquet_path = self.checkpoint_dir / f"{self.checkpoint_basename}.parquet"

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
                    data = load_dataset_from_pickle(
                        output_prefix=self.checkpoint_basename,
                        data=data
                    )
                else:
                    load_dataset_from_parquet(
                        output_prefix=self.checkpoint_basename,
                        data=data
                    )

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
