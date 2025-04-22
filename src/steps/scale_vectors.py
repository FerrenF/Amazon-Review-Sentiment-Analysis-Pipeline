import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from core.step import Step


class ScaleVectorsStep(Step):
    name = "scale_vectors"

    def __init__(self, method='minmax'):
        """
        Initializes a scaling step using the specified method.

        Note: It's probably not a good idea to do both normalization and scaling.

        :param method: Type of scaler to use. Options: 'minmax', 'standard'
        """
        if method not in ['minmax', 'standard']:
            raise ValueError("Unsupported scaling method. Use 'minmax' or 'standard'.")

        self.method = method
        self.scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()

    def run(self, data: dict) -> dict:
        """
        Scales the 'vector' column in the DataFrame by applying feature-wise scaling.
        Assumes that each vector is a 1D array of numeric features.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        if "vector" not in df.columns:
            logging.warning("No 'vector' column found, skipping scaling.")
            return data

        logging.info(f"Scaling vectors using {self.method} scaler...")

        # Convert list/array of vectors into 2D array
        vectors = np.array(df["vector"].tolist())
        scaled_vectors = self.scaler.fit_transform(vectors)

        # Update the DataFrame with scaled vectors
        df["vector"] = list(scaled_vectors)
        data["dataset"] = df

        logging.info("Scaling complete.")
        return data
