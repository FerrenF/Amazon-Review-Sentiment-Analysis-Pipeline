import logging
import pathlib

from core.stage import Stage
from core.step import Step
from utils import load_dataset_from_pickle
from utils import load_dataset_from_parquet


class LoadCheckpointIfExists(Step):

    def __init__(self, stage_name: str, prefix="data", is_pickle=False):
        self.name = f"load_checkpoint_{stage_name}"
        self.stage_name = stage_name
        self.prefix = prefix
        self.is_pickle = is_pickle

    def run(self, data: dict) -> dict:
        from common.project_common import PROCESSED_DATA_DIR, PROCESSED_EXT

        checkpoint_path = pathlib.Path(PROCESSED_DATA_DIR).joinpath(
            f"{self.prefix}_{self.stage_name}{PROCESSED_EXT if not self.is_pickle else '.pkl'}"
        )

        if checkpoint_path.exists():
            logging.info(
                f"Checkpoint for stage '{self.stage_name}' found at {checkpoint_path}, loading instead of rerunning stages up to checkpoint.")
            try:
                if self.is_pickle:
                    load_dataset_from_pickle(stage=Stage(self.stage_name, []),
                                                    output_prefix=self.prefix, data=data)
                else:
                    load_dataset_from_parquet(stage=Stage(self.stage_name, []), output_prefix=self.prefix,
                                                     data=data)
                data["skip_to_stage"] = self.stage_name
            except Exception as e:
                logging.error(f"Failed to load checkpoint for stage '{self.stage_name}': {e}")
                data["skip_to_stage"] = None
        else:
            logging.info(f"No checkpoint found for stage '{self.stage_name}', proceeding with full execution.")
            data["skip_to_stage"] = None

        return data
