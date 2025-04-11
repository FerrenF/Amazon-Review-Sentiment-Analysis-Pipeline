import logging
from sklearn.model_selection import train_test_split
from core.step import Step


class TrainTestSplitStep(Step):
    name = "train_test_split_step"

    def __init__(self, test_size, random_state, perform_stratification=True):
        self.test_size = test_size
        self.random_state = random_state
        self.perform_stratification = perform_stratification

    def run(self, data: dict) -> dict:
        """
        Splits the dataset into training and testing sets (default: 80/20).
        Uses the 'vector' column for features, 'label' for target,
        and tracks which text entries go into which split.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        X = df["vector"].tolist()
        y = df["label"].tolist()
        text = df["text"].tolist()

        logging.info("Splitting dataset into train/test sets...")

        (
            X_train, X_test,
            y_train, y_test,
            text_train, text_test
        ) = train_test_split(
            X, y, text,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.perform_stratification else None
        )

        data["X_train"] = X_train
        data["X_test"] = X_test
        data["y_train"] = y_train
        data["y_test"] = y_test
        data["text_train"] = text_train
        data["text_test"] = text_test

        logging.info(f"Train/test split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
        return data
