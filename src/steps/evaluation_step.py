import logging

import numpy as np
from sklearn.metrics import mean_squared_error
from core.step import Step


class EvaluationStep(Step):
    name = "evaluation"
    def __init__(self, metrics=None):
        if metrics is None:
            metrics = ["mse"]
        self.metrics = metrics

    def run(self, data: dict) -> dict:
        """
        Evaluates the model using the vectorized test data and specified metrics.
        Assumes that the 'model' key contains the trained model,
        and the dataset includes 'label' and 'vector' vectorized data columns.
        """
        if "model" not in data:
            raise ValueError("No model found in the data dictionary.")

        model = data["model"]
        if "X_test" not in data:
            raise ValueError("No test dataset found in the data dictionary.")

        if "y_test" not in data:
            raise ValueError("No test labels found in the data dictionary.")

        X_test = data["X_test"]
        y_test = data["y_test"]

        logging.info(f"Evaluating model...")

        y_pred = model.predict(X_test)

        # Calculate and log metrics
        results = {}
        if "mse" in self.metrics:
            mse = mean_squared_error(y_test, y_pred)
            results["mse"] = mse
            logging.info(f"Mean Squared Error (MSE): {mse:.4f}")

        # Add more metrics here as needed

        # Store results in data for future use
        data["evaluation_results"] = results
        return data
