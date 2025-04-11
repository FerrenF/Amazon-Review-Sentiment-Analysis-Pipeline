import pandas as pd
import numpy as np
import re
from core.step import Step

class UnflattenVectorColumnsStep(Step):
    name = "unflatten_vector_columns"

    def __init__(self, prefix="vec_"):
        self.prefix = prefix

    def run(self, data: dict) -> dict:
        """
        Reconstructs a single 'vector' column from individual flattened vector columns (e.g., vec_0, vec_1, ...).
        We do this in the vectorization step to allow it to be written to a parquet.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        # Find all vector columns
        vector_cols = sorted(
            [col for col in df.columns if re.match(rf"^{self.prefix}\d+$", col)],
            key=lambda x: int(x.replace(self.prefix, ""))
        )

        if not vector_cols:
            raise ValueError(f"No columns found matching pattern {self.prefix}<int>.")

        # Reconstruct 'vector' column
        df["vector"] = df[vector_cols].apply(lambda row: np.array(row.values, dtype=np.float32), axis=1)

        # Optionally drop the flattened columns
        df.drop(columns=vector_cols, inplace=True)

        data["dataset"] = df
        return data
