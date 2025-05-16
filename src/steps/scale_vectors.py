import logging
from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from core.step import Step


class ScaleVectorsStep(Step):
    name = "scale_vectors"

    def __init__(self, method='minmax', target_column='vector', output_column=None):
        """
        Initializes a scaling step using the specified method.

        :param method: Type of scaler to use. Options: 'minmax', 'standard'
        :param target_column: Name of the column in data["dataset"] to scale.
        :param output_column: Name of the column to store the scaled vectors. Defaults to target_column.
        """
        if method not in ['minmax', 'standard']:
            raise ValueError("Unsupported scaling method. Use 'minmax' or 'standard'.")

        self.method = method
        self.scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
        self.target_column = target_column
        self.output_column = output_column if output_column is not None else target_column

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))
        data["stats"]["scaling"] = self.method

    def run(self, data: dict) -> dict:
        """
        Scales the specified column in the DataFrame by applying feature-wise scaling.
        Assumes that each entry is a 1D array of numeric features.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        if self.target_column not in df.columns:
            logging.warning(f"No '{self.target_column}' column found, skipping scaling.")
            return data

        self.step_log(f"Scaling vectors in column '{self.target_column}' using {self.method} scaler...")

        # Convert list/array of vectors into 2D array
        vectors = np.array(df[self.target_column].tolist())
        scaled_vectors = self.scaler.fit_transform(vectors)

        # Store scaled vectors in output column
        if isinstance(self.output_column, tuple):
            data[self.output_column[0]][self.output_column[1]] = list(scaled_vectors)
        else:
            data[self.output_column] = list(scaled_vectors)

        self.step_log(f"Scaling complete. Scaled vectors stored in column '{self.output_column}'.")
        return data
