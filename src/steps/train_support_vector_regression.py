import logging
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from core.step import Step

class SVRStep(Step):

    name = "svr"

    def __init__(self, name: str = "svr", kernel='rbf', C=1.0, epsilon=0.1, grid_search=False, param_grid=None):
        """
          Trains a Support Vector Machine for a **Regression** model using vectorized text data and stores it in the data dictionary.
          Optionally performs grid search for hyperparameter tuning.

          :param grid_search: Whether to use GridSearchCV for hyperparameter tuning.
          :param param_grid: The grid of parameters to search over.
          :param params: Default parameters for SVR if grid_search is False.
          """
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        self.grid_search = grid_search
        self.param_grid = param_grid if param_grid else {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'poly']  # Optionally add 'linear' if linearity is suspected
        }

    def run(self, data: dict) -> dict:

        if "dataset" not in data:
            raise ValueError("No dataset found in the data dictionary.")
        if "X_train" not in data:
            raise ValueError("No train data found in the data dictionary.")
        if "y_train" not in data:
            raise ValueError("No training labels found in the data dictionary.")

        X = data["X_train"]
        y = data["y_train"]

        logging.info(f"Training {self.name} model...")

        if self.grid_search:
            # Perform GridSearchCV for hyperparameter tuning
            logging.info("Starting grid search for hyperparameter tuning...")
            grid_search = GridSearchCV(self.model, self.param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            logging.info(f"Grid search completed. Best parameters: {grid_search.best_params_}")
        else:
            # If no grid search, train the model with initial parameters
            self.model.fit(X, y)

        # Store the model in data for later use
        data["model"] = self.model
        logging.info(f"Training complete for {self.name}.")

        return data
