import logging

from sklearn.linear_model import Ridge
from core.step import Step


class RidgeRegressionStep(Step):
    name = "ridge_regression"

    def __init__(self, **params):
        self.model = Ridge(**params)

    def run(self, data: dict) -> dict:
        """
        Trains a Ridge Regression model and stores it in the data dictionary.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in the data dictionary.")
        if "X_train" not in data:
            raise ValueError("No train data found in the data dictionary.")
        if "y_train" not in data:
            raise ValueError("No training labels found in the data dictionary.")

        X = data["X_train"]
        y = data["y_train"]

        logging.info(f"Training {self.name} model...")
        self.model.fit(X, y)
        data["model"] = self.model
        logging.info(f"Training complete for {self.name}.")

        return data
