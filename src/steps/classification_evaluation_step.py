import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from core.step import Step


class ClassificationEvaluationStep(Step):
    name = "classification_evaluation"
    description = "Performs evaluation on classification task"

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))

    def __init__(self, metrics: Optional[List[str]] = None, average: str = "macro"):
        """
        :param metrics: List of metrics to compute. Options: accuracy, precision, recall, f1, confusion_matrix, report
        :param average: Averaging method for multi-class metrics (e.g., 'macro', 'micro', 'weighted').
        """
        self.metrics = metrics or ["accuracy"]
        self.average = average

    def run(self, data: dict) -> dict:

        if "model" not in data:
            raise ValueError("No model found in the data dictionary.")
        if "X_test" not in data or "y_test" not in data:
            raise ValueError("Missing test set or labels in the data dictionary.")

        model = data["model"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        self.step_log(f'{self.friendly_name()} : {self.description}')
        y_pred = model.predict(X_test)

        results = {}

        if "accuracy" in self.metrics:
            acc = accuracy_score(y_test, y_pred)
            results["accuracy"] = acc
            self.step_log(f"Accuracy: {acc:.4f}")

        if "precision" in self.metrics:
            prec = precision_score(y_test, y_pred, average=self.average, zero_division=0)
            results["precision"] = prec
            self.step_log(f"Precision ({self.average}): {prec:.4f}")

        if "recall" in self.metrics:
            rec = recall_score(y_test, y_pred, average=self.average, zero_division=0)
            results["recall"] = rec
            self.step_log(f"Recall ({self.average}): {rec:.4f}")

        if "f1" in self.metrics:
            f1 = f1_score(y_test, y_pred, average=self.average, zero_division=0)
            results["f1"] = f1
            self.step_log(f"F1 Score ({self.average}): {f1:.4f}")

        if "confusion_matrix" in self.metrics:
            cm = confusion_matrix(y_test, y_pred)
            results["confusion_matrix"] = cm.tolist()
            self.step_log(f"Confusion Matrix:\n{cm}")

        if "report" in self.metrics:
            report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            results["classification_report"] = report
            self.step_log("Classification Report:\n" + classification_report(y_test, y_pred, zero_division=0))

        data["stats"]["evaluation_results"] = results
        return data
