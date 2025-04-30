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

        if hasattr(X_train, 'toarray'):
            df = pd.DataFrame(X_train.toarray())
        else:
            df = pd.DataFrame(X_train)

        df[self.target_y] = pd.Series(y_train).values

        # Get class counts
        counts = df[self.target_y].value_counts()
        logging.info(f"Label counts before balancing: \n{counts}")

        min_size = counts.min()
        max_size = counts.max()

        if self.method == "undersample":
            dfs = [
                resample(df[df[self.target_y] == label],
                         replace=False,
                         n_samples=min_size,
                         random_state=42)
                for label in df[self.target_y].unique()
            ]
        else:
            dfs = [
                resample(df[df[self.target_y] == label],
                         replace=True,
                         n_samples=max_size,
                         random_state=42)
                for label in df[self.target_y].unique()
            ]


        balanced_df = pd.concat(dfs, ignore_index=True)

        # Shuffle the result...
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split back into X and y
        X_resampled = balanced_df.drop(columns=[self.target_y])
        y_resampled = balanced_df[self.target_y]

        logging.info(f"Label balancing complete using {self.method} method.")
        counts = y_resampled.value_counts()
        logging.info(f"Label counts after balancing: \n{counts}")

        data[self.target_x] = X_resampled
        data[self.target_y] = y_resampled
        return data
