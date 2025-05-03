import logging
from core.step import Step


class RemapLabelsStep(Step):
    name = "remap_labels"

    def __init__(self, mapping: dict, target_column="label"):
        """
        :param mapping: Dictionary of {old_label: new_label}
        :param target_column: Name of the column with the labels

        For testing a lower number of possible classifications.
        """
        self.mapping = mapping
        self.target_column = target_column

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")

        logging.info(f"Remapping labels in column '{self.target_column}' using mapping: {self.mapping}")

        # Perform mapping, default to original if not in mapping
        df[self.target_column] = df[self.target_column].map(lambda x: self.mapping.get(x, x))

        logging.info(f"Label remapping complete.")

        data["dataset"] = df
        return data
