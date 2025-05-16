import logging
from datetime import datetime

from core.step import Step


class RemapLabelsStep(Step):
    name = "remap_labels"

    def __init__(self, mapping: dict, target_column="label"):
        """
        :param mapping: Dictionary of {old_label: new_label}
        :param target_column: Name of the column with the labels

        This class will iterate through the key, value  pairs in mapping, replacing occurrences of old_label with new_label within the data that is passed through this step.

        For testing a lower number of possible classifications. If I have 5 classes, 1, 2, 3, 4, and 5 but I no longer want to try and classify 2 and 4, then some of my options are to
        remap my old classifications into my new classification system. I could remap my system into three classes using the parameter:
                mapping = {
                    2: 3,
                    4: 3
                }
        I would be left with a dataset of only those data points labelled 1, 3 and 5 and my solution would be mapping affected rows towards the median. Conversely, I could map using:
                mapping = {
                    2: 1,
                    4: 5
                }
        Which would accomplish the same goal except it would replace the labels of affected rows with the more extreme neighbor.

        A third solution would be simply dropping the data completely. Use DropLabelsStep for that, though.
        """
        self.mapping = mapping
        self.target_column = target_column


    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))
        data["stats"]["remapping"] = self.mapping

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")

        self.step_log(f"Remapping labels in column '{self.target_column}' using mapping: {self.mapping}")

        # Perform mapping, default to original if not in mapping
        df[self.target_column] = df[self.target_column].map(lambda x: self.mapping.get(x, x))

        self.step_log(f"Label remapping complete.")

        data["dataset"] = df
        return data
