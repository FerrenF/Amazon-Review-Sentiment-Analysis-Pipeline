import logging
from datetime import datetime
import numpy as np
from scipy.sparse import issparse

from core.step import Step


class DropLabelsStep(Step):
    name = "drop_labels"
    description = "Drops labels."

    def __init__(self, labels_to_drop: list, target_key="label",  vector_key="vector_data"):
        """
        :param labels_to_drop: List of label values to remove from the dataset.
        :param target_key: Name of the key containing the labels.
        :param vector_key: Name of the key containing the vectors.

        This step will remove any rows from the dataset where the value in the target column matches one of the labels to drop. If vector key is supplied, it will also remove the
        associated indice also.

        Example:
            To remove all samples labeled 2 and 4:
                DropLabelsStep(labels_to_drop=[2, 4], target_column="label")
        """
        self.labels_to_drop = labels_to_drop
        self.target_column = target_key
        self.vector_key = vector_key

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")

        self.step_log(f'{self.friendly_name()} : {self.description}')
        original_count = len(df)

        # Identify rows to keep
        keep_mask = ~df[self.target_column].isin(self.labels_to_drop)
        # Filter dataset

        self.step_log(f"Dropping labels {self.labels_to_drop} from target '{self.target_column}'.")
        df = df.loc[keep_mask].reset_index(drop=True)

        dropped_count = original_count - len(df)
        self.step_log(f"Dropped {dropped_count} rows. Remaining: {len(df)}.")


        keep_indices = np.where(keep_mask.to_numpy())[0]
        if isinstance(self.vector_key, tuple):
            vd = data[self.vector_key[0]][self.vector_key[1]]
        else:
            vd = data[self.vector_key]

        if issparse(vd):
            vd = vd[keep_indices]
        else:
            vd = np.asarray(vd)[keep_indices]

        if isinstance(self.vector_key, tuple):
            data[self.vector_key[0]][self.vector_key[1]] = vd
        else:
            data[self.vector_key] = vd

        self.step_log(
            f"Filtered vector data in '{self.target_column}' to match dataset (new shape: {vd.shape})")

        data["dataset"] = df
        return data
