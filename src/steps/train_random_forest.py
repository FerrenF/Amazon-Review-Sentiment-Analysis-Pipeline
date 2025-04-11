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
        Trains a Random Forest Regression model using vectorized text data and stores it in the data dictionary.
        The dataset should include a 'vector' column (features) and a 'target' column (labels).
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in the data dictionary.")

        df = data["dataset"]
        X = df["vector"].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(
            self.vector_size)).tolist()  # Vectorized data as features
        y = df["target"]  # Assuming 'target' column exists

        logging.info(f"Training {self.name} model...")
        self.model.fit(X, y)

        # Store the model in data for later use
        data["model"] = self.model
        logging.info(f"Training complete for {self.name}.")

        return data
