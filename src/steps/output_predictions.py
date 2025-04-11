import logging
import json
import pathlib

import numpy as np

from core.step import Step

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


class OutputPredictionsStep(Step):
    name = "output_predictions"

    def __init__(self, output_key: str = "predictions", save_to_file: bool = True, output_dir: str = "output", filename: str = "predictions.json"):
        self.output_key = output_key
        self.save_to_file = save_to_file
        self.output_dir = pathlib.Path(output_dir)
        self.filename = filename

    def run(self, data: dict) -> dict:
        """
        Collects predictions with original text and true labels for inspection.
        Optionally saves results to a JSON file.
        """
        if "model" not in data or "X_test" not in data or "y_test" not in data:
            raise ValueError("Missing model, X_test, or y_test in data.")

        if "dataset" not in data or "text" not in data["dataset"]:
            raise ValueError("Missing dataset text in data['dataset']['text'].")

        model = data["model"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        text_train = data["text_train"]
        text_test = data["text_test"]

        y_pred = model.predict(X_test)

        predictions = []
        for i in range(len(X_test)):
            predictions.append({
                "text": text_test[i],
                "true_label": y_test[i],
                "predicted_label": y_pred[i]
            })

        data[self.output_key] = predictions

        if self.save_to_file:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / self.filename
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
            logging.info(f"Saved {len(predictions)} predictions to {output_path}")

        else:
            logging.info(f"Stored {len(predictions)} predictions in memory under '{self.output_key}'.")

        return data
