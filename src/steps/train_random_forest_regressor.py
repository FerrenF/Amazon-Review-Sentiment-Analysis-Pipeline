import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from core.step import Step


class RandomForestRegressionStep(Step):
    name = "random_forest_regression"
    def __init__(self, **params):
        self.model = RandomForestRegressor(**params)

    def run(self, data: dict) -> dict:
        """
        Trains a Random Forest Regression model using the vectorized text data and stores it in the data dictionary.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in the data dictionary.")

        df = data["dataset"]
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
