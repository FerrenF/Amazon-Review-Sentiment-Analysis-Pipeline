import logging

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from core.step import Step


class LinearRegressionStep(Step):
    name = "linear_regression"
    def __init__(self, **params):
        self.model = LinearRegression(**params)

    def run(self, data: dict) -> dict:
        """
        Trains a Linear Regression model and stores it in the data dictionary.
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

        # Store the model in data for later use
        data["model"] = self.model
        logging.info(f"Training complete for {self.name}.")

        return data