import logging
from datetime import datetime

from sklearn.linear_model import Lasso
from core.step import Step


class LassoRegressionStep(Step):
    name = "lasso_regression"

    def __init__(self, **params):
        self.model = Lasso(**params)

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))
        data["stats"]["model"] = "Lasso Regression"

    def run(self, data: dict) -> dict:
        """
        Trains a Lasso Regression model and stores it in the data dictionary.
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
