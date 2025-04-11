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
        Uses the 'vector' column for features and 'label' for target.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]
        X = df["vector"].tolist()
        y = df["label"].tolist()

        logging.info("Splitting dataset into train/test sets...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify= y if self.perform_stratification else None
        )

        data["X_train"] = X_train
        data["X_test"] = X_test
        data["y_train"] = y_train
        data["y_test"] = y_test

        logging.info("Train/test split complete.")
        return data
