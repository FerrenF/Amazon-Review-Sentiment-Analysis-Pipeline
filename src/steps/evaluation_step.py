import logging

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
        y_pred = np.clip(y_pred, -1, 1)
        y_pred_rounded =  np.clip(np.round(y_pred), -1, 1).astype(int)

        # Calculate and log metrics
        results = {}
        if "mse" in self.metrics:

            baseline = np.mean(data['y_train'])
            baseline_mse = mean_squared_error(y_test, [baseline] * len(y_test))
            print(f"Baseline MSE: {baseline_mse}")

            mse = mean_squared_error(y_test, y_pred_rounded)
            results["mse"] = mse
            logging.info(f"Mean Squared Error (MSE): {mse:.4f}")

        if "mae" in self.metrics:
            mae = mean_absolute_error(y_test, y_pred_rounded)
            results["mae"] = mae
            logging.info(f"Mean Absolute Error (MAE): {mae:.4f}")

        if "r2" in self.metrics:
            r2 = r2_score(y_test, y_pred_rounded)
            results["r2"] = r2
            logging.info(f"R² Score: {r2:.4f}")

        if "mape" in self.metrics:
            y_test_safe = np.where(y_test == 0, 1e-8, y_pred_rounded)  # avoid division by zero
            mape = np.mean(np.abs((y_test - y_pred_rounded) / y_test_safe)) * 100
            results["mape"] = mape
            logging.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

        # Add more metrics here as needed

        # Store results in data for future use
        data["evaluation_results"] = results
        return data
