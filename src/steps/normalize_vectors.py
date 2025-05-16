import datetime
import logging
from typing import Union, Tuple, Optional

import numpy as np
from scipy.sparse import issparse, csr_matrix
from sklearn.preprocessing import normalize

from core.step import Step


class VectorNormalizationStep(Step):
    name = "vector_normalization"

    def __init__(
        self,
        normalization_type = "l2",
        vector_key: Union[str, Tuple[str, str]] = "vector_data",
        output_key: Optional[Union[str, Tuple[str, str]]] = None,
        output_format: Optional[str] = None  # 'dense', 'sparse', or None to preserve input format
    ):

        """
        Normalizes a matrix of vectors ('l2' by default, see sklearn normalize for other types)

        :param vector_key: Where to read the matrix from (sparse or dense).
        :param output_key: Where to store the normalized matrix. If None, overwrites vector_key.
        :param output_format: 'dense', 'sparse', or None to keep the input format.
        """

        self.normalization_type = normalization_type
        self.vector_key = vector_key
        self.output_key = output_key or vector_key
        self.output_format = output_format.lower() if output_format else None
        if self.output_format and self.output_format not in {"dense", "sparse"}:
            raise ValueError("output_format must be one of: 'dense', 'sparse', or None")


    def _resolve_key(self, key: Union[str, Tuple[str, str]]) -> Tuple[Optional[str], str]:
        return (None, key) if isinstance(key, str) else key


    def _get_data_branch(self, data: dict, key: Union[str, Tuple[str, str]]):
        root, branch = self._resolve_key(key)
        return data[root][branch] if root else data[branch]


    def _set_data_branch(self, data: dict, key: Union[str, Tuple[str, str]], value):
        root, branch = self._resolve_key(key)
        if root:
            if root not in data:
                data[root] = {}
            data[root][branch] = value
        else:
            data[branch] = value

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.datetime.now()))
        data["stats"]["normalization"] = self.normalization_type

    def run(self, data: dict) -> dict:
        matrix = self._get_data_branch(data, self.vector_key)

        if matrix is None:
            raise ValueError(f"No vector data found at key: {self.vector_key}")

        self.step_log(f"Normalizing vectors: {self.vector_key} (type: {'sparse' if issparse(matrix) else 'dense'})")

        # Normalize in-place if dense, or create new matrix if sparse
        normed = normalize(matrix, norm=self.normalization_type, axis=1, copy=True)

        # Convert format if requested
        if self.output_format == "dense" and issparse(normed):
            normed = normed.toarray()
        elif self.output_format == "sparse" and not issparse(normed):
            normed = csr_matrix(normed)

        self._set_data_branch(data, self.output_key, normed)

        self.step_log(f"Stored normalized vectors in {self.output_key} (type: {'sparse' if issparse(normed) else 'dense'})")
        return data
