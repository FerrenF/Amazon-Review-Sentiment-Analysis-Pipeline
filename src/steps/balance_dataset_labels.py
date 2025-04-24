import logging
import pandas as pd
from sklearn.utils import resample
from core.step import Step

class BalanceLabelsStep(Step):
    name = "balance_labels_step"

    def __init__(self, sample_method: str = "undersample"):
        self.method = sample_method

    def run(self, data: dict) -> dict:
        """
        Balancs the dataset using either undersampling or oversampling (default = 'undersample', opts: 'oversample').
        Assumes a 'label' column is present.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        logging.info(f"Performing label balancing using '{self.method}' method...")

        # Find the majority and minority classes
        counts = df["label"].value_counts()
        logging.info(f"Label counts before balancing: \n{counts}")

        min_size = counts.min()
        max_size = counts.max()

        if self.method == "undersample":
            dfs = [resample(df[df["label"] == label],
                            replace=False,
                            n_samples=min_size,
                            random_state=42)
                   for label in df["label"].unique()]
        else:
            dfs = [resample(df[df["label"] == label],
                            replace=True,
                            n_samples=max_size,
                            random_state=42)
                   for label in df["label"].unique()]

        balanced_df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        logging.info(f"Label balancing complete using {self.method} method.")
        counts = balanced_df["label"].value_counts()
        logging.info(f"Label counts after balancing: \n{counts}")

        data["dataset"] = balanced_df
        return data
