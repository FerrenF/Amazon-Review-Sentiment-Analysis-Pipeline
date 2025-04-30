import logging
import pandas as pd
from sklearn.utils import resample
from core.step import Step

class BalanceLabelsStep(Step):
    name = "balance_labels_step"

    def __init__(self, sample_method: str = "undersample", targets="dataset"):
        self.method = sample_method

        if not isinstance(targets, tuple):
            raise ValueError("Targets parameter must be a tuple containing two values.")
        self.target_x, self.target_y = targets

    def run(self, data: dict) -> dict:
        """
        Balances the dataset using either undersampling or random oversampling (default = 'undersample', opts: 'oversample').
        Assumes a 'label' column is present.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        logging.info(f"Performing label balancing on [{self.target_x}] using '{self.method}' method...")

        X_train = data[self.target_x]
        y_train = data[self.target_y]

        df = pd.DataFrame(X_train)  # Assumes X_train is a list of lists or 2D array
        df["label"] = pd.Series(y_train)  # Add the label column

        # Get class counts
        counts = df["label"].value_counts()
        logging.info(f"Label counts before balancing: \n{counts}")

        min_size = counts.min()
        max_size = counts.max()

        if self.method == "undersample":
            dfs = [
                resample(df[df["label"] == label],
                         replace=False,
                         n_samples=min_size,
                         random_state=42)
                for label in df["label"].unique()
            ]
        else:
            dfs = [
                resample(df[df["label"] == label],
                         replace=True,
                         n_samples=max_size,
                         random_state=42)
                for label in df["label"].unique()
            ]


        balanced_df = pd.concat(dfs)

        # Shuffle the result...
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split back into X and y
        X_resampled = balanced_df.drop(columns=["label"])
        y_resampled = balanced_df["label"]

        logging.info(f"Label balancing complete using {self.method} method.")
        counts = y_resampled.value_counts()
        logging.info(f"Label counts after balancing: \n{counts}")

        data[self.target_x] = X_resampled
        data[self.target_y] = y_resampled
        return data
