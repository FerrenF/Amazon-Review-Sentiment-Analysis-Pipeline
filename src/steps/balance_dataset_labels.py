import logging
import pandas as pd
from sklearn.utils import resample
from core.step import Step

class BalanceLabelsStep(Step):
    name = "balance_labels_step"

    def __init__(self, sample_method: str = "undersample", targets="dataset", text_col = None, oversample_cap: int = None):
        self.method = sample_method
        self.oversample_cap = oversample_cap
        self.text_col = text_col
        if not isinstance(targets, tuple):
            raise ValueError("Targets parameter must be a tuple containing two values.")
        self.target_x, self.target_y = targets

    def run(self, data: dict) -> dict:
        if self.target_x not in data or self.target_y not in data:
            raise ValueError(f"Expected keys {self.target_x} and {self.target_y} in data.")

        logging.info(f"Performing label balancing on [{self.target_x}] using '{self.method}' method...")

        X_train = data[self.target_x]
        y_train = data[self.target_y]

        # Get label counts
        from collections import Counter
        counts = Counter(y_train)
        logging.info(f"Label counts before balancing: \n{counts}")

        min_size = min(counts.values())
        max_size = max(counts.values())

        indices_per_class = {
            label: [i for i, y in enumerate(y_train) if y == label]
            for label in counts.keys()
        }

        sampled_indices = []
        for label, indices in indices_per_class.items():
            n_samples = min_size if self.method == "undersample" else max_size
            if self.oversample_cap is not None and self.method != "undersample":
                n_samples = min(n_samples, self.oversample_cap)

            sampled = resample(indices, replace=(self.method != "undersample"),
                               n_samples=n_samples, random_state=42)
            sampled_indices.extend(sampled)

        from sklearn.utils import shuffle
        sampled_indices = shuffle(sampled_indices, random_state=42)

        if hasattr(X_train, 'iloc'):  # pandas DataFrame
            X_resampled = X_train.iloc[sampled_indices]
        else:  # numpy array, scipy sparse, etc.
            X_resampled = X_train[sampled_indices]

        if isinstance(y_train, pd.Series):
            y_resampled = y_train.iloc[sampled_indices].tolist()
        else:
            y_resampled = [y_train[i] for i in sampled_indices]

        logging.info(f"Label balancing complete using {self.method} method.")
        counts_after = Counter(y_resampled)
        logging.info(f"Label counts after balancing: \n{counts_after}")

        data[self.target_x] = X_resampled
        data[self.target_y] = y_resampled

        if self.text_col and self.text_col in data:
            text_train = data[self.text_col]
            if hasattr(text_train, 'iloc'):
                text_resampled = text_train.iloc[sampled_indices]
            else:
                text_resampled = [text_train[i] for i in sampled_indices]
            data[self.text_col] = text_resampled

        return data

