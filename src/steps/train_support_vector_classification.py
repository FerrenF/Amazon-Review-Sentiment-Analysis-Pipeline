import logging
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from core.step import Step


class SupportVectorClassificationStep(Step):
    name = "support_vector_classification"

    def __init__(self, grid_search: bool = False, param_grid: dict = None, **params):
        """
        Trains a Support Vector Machine for a **Classification** model using vectorized text data and stores it in the data dictionary.
        Optionally performs grid search for hyperparameter tuning.

        :param grid_search: Whether to use GridSearchCV for hyperparameter tuning.
        :param param_grid: The grid of parameters to search over.
        :param params: Default parameters for SVC if grid_search is False.
        """
        self.grid_search = grid_search
        self.param_grid = param_grid or {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"]
        }
        self.model = SVC(**params)

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in the data dictionary.")
        if "X_train" not in data or "y_train" not in data:
            raise ValueError("Training data or labels missing from the data dictionary.")

        X = data["X_train"]
        y = data["y_train"]

        if self.grid_search:
            logging.info(f"Running grid search for {self.name}...")
            grid_search = GridSearchCV(SVC(), self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            data["hyperparameters"] = grid_search.best_params_
            logging.info(f"Best parameters from grid search: {grid_search.best_params_}")
        else:
            data["hyperparameters"] = "default"
            logging.info(f"Training {self.name} model...")
            self.model.fit(X, y)

        data["model"] = self.model
        logging.info(f"Training complete for {self.name}.")

        return data
