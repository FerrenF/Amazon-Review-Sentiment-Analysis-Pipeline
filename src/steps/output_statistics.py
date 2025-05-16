import logging
import json
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd

from core.step import Step


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Series,)):
            return obj.to_dict()
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        return super().default(obj)


class OutputStatisticsStep(Step):
    name = "output_statistics"

    def __init__(self, save_to_file: bool = True,
                 output_dir: str = "output",
                 filename: str = "statistics.jsonl", append=True):
        self.save_to_file = save_to_file
        self.output_dir = pathlib.Path(output_dir)
        self.filename = filename
        self.append = append

    def set_stats(self, data: dict):
        if "stats" not in data:
            data["stats"] = {}
        if "time" not in data["stats"]:
            data["stats"]["time"] = []
        data["stats"]["time"].append((self.name, datetime.now()))

    def run(self, data: dict) -> dict:
        """
        Outputs the 'stats' key of data in JSONL format (one JSON object per line).
        Ensures all values are JSON serializable.
        """
        stats = data.get("stats", {})

        if self.save_to_file:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / self.filename

            try:
                with output_path.open("a" if self.append else "w+", encoding="utf-8") as f:
                    json.dump(stats, f, cls=NumpyEncoder)
                    f.write("\n")
                self.step_log(f"Written to {output_path}")
            except Exception as e:
                logging.error(f"Failed to write statistics: {e}")
