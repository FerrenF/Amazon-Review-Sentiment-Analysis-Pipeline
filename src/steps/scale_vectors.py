import logging
from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from core.step import Step
from scipy import sparse

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
        if method == 'minmax':
            self.scaler = MinMaxScaler() 
        elif method == 'standard': 
            self.scaler = StandardScaler()      
             
        self.target_column = target_column
        self.output_column = output_column if output_column is not None else target_column

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))
        data["stats"]["scaling"] = self.method

    def run(self, data: dict) -> dict:
        """
        Scales the specified column in the DataFrame by applying feature-wise scaling.
        Handles both dense lists of vectors and sparse matrices.
        """
    
        # Access the target data
        if isinstance(self.target_column, tuple):
            df = data[self.target_column[0]][self.target_column[1]]
        else:
            df = data[self.target_column]
    
        self.step_log(f"Scaling vectors in column '{self.target_column}' using {self.method} scaler...")
    
        # Convert to 2D array/matrix
        if sparse.issparse(df):
            vectors = df  # Keep sparse format
        else:
            vectors = np.array(df.tolist())  # Assume list of lists/arrays
    
        # Scale
        scaled_vectors = self.scaler.fit_transform(vectors)
    
        # Store result
        if isinstance(self.output_column, tuple):
            data[self.output_column[0]][self.output_column[1]] = scaled_vectors
        else:
            data[self.output_column] = scaled_vectors
    
        self.step_log(f"Scaling complete. Scaled vectors stored in column '{self.output_column}'.")
        return data
