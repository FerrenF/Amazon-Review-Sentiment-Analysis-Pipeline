import logging
import numpy as np
from sklearn.preprocessing import Normalizer
from core.step import Step


class NormalizeVectorsStep(Step):
    name = "normalize_vectors"

    def __init__(self, norm='l2'):
        """
        Initializes a Normalization Step with the desired normalization type.
        Supported norm types: 'l1', 'l2' or 'max'.

        :param norm: The type of normalization. Default is 'l2'.
        """
        self.norm = norm
        self.normalizer = Normalizer(norm=self.norm)

    def run(self, data: dict) -> dict:
        """
        Normalizes the 'vector' column in the DataFrame by scaling each vector to the desired norm.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        if "vector" not in df.columns:
            logging.warning("No 'vector' column found, skipping normalization.")
            return data

        # Normalize the vectors
        logging.info("Normalizing vectors...")
        df["vector"] = df["vector"].apply(self._normalize_vector)

        data["dataset"] = df
        logging.info("Normalization complete.")
        return data

    def _normalize_vector(self, vector):
        """
        Normalizes a single vector.
        """
        if isinstance(vector, (np.ndarray, list)):
            vector = np.array(vector)
            return self.normalizer.transform([vector])[0]  # Normalize and return the vector
        else:
            return vector  # In case it's not a valid vector
