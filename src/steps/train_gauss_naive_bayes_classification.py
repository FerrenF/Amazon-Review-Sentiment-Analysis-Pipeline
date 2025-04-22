import logging
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from core.step import Step


class GaussNaiveBayesClassificationStep(Step):
    name = "gauss_naive_bayes_classification"

    def __init__(self, grid_search: bool = False, param_grid: dict = None, **params):
        """

        :param grid_search: Whether to perform a GridSearchCV to optimize hyperparameters (usually unnecessary).
        :param param_grid: Grid of parameters for GridSearchCV.
        :param params: Default model parameters.
        """
        self.grid_search = grid_search
        self.param_grid = param_grid or {
            "var_smoothing": [1e-9, 1e-8, 1e-7]  # This is the only tunable param for GaussianNB
        }
        self.model = GaussianNB(**params)

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in the data dictionary.")
        if "X_train" not in data or "y_train" not in data:
            raise ValueError("Training data or labels missing from the data dictionary.")

        X = data["X_train"]
        y = data["y_train"]

        if self.grid_search:
            logging.info(f"Running grid search for {self.name}...")
            grid_search = GridSearchCV(GaussianNB(), self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
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
