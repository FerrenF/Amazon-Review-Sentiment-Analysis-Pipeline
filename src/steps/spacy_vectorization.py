import logging
import subprocess
import sys
from datetime import datetime
from typing import Optional, Type, Tuple, Union

import spacy
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from core.step import Step


class SpacyVectorizationStep(Step):
    name = "spacy_vectorization"

    def __init__(
        self,
        model: str = "en_core_web_md",
        target_key: Union[str, Tuple[str, str]] = ("dataset", "text"),
        output_key: Union[str, Tuple[str, str]] = ("dataset", "vector_ref"),
        vector_key: Union[str, Tuple[str, str]] = "vector_data",
        output_format: str = "sparse",  # or 'dense'
        numeric_type: Type = np.float32,
        **model_kwargs
    ):
        """
        Vectorizes text using spaCy and stores the result as either a sparse or dense matrix.
        Each row in the dataset receives an index into the matrix.

        :param model: spaCy model to use.
        :param target_key: Where to read text data from.
        :param output_key: Where to store row indices pointing into the vector matrix.
        :param vector_key: Where to store the actual matrix (dense or sparse).
        :param output_format: 'sparse' or 'dense'.
        :param numeric_type: NumPy dtype for vector array.
        :param model_kwargs: Additional kwargs to pass to `spacy.load`.
        """
        self.model = model
        self.target_key = target_key
        self.output_key = output_key
        self.vector_key = vector_key
        self.output_format = output_format.lower()
        self.numeric_type = numeric_type
        self.model_kwargs = model_kwargs

        if self.output_format not in {"sparse", "dense"}:
            raise ValueError("`output_format` must be either 'sparse' or 'dense'.")

        try:
            self.nlp = spacy.load(model, **model_kwargs)
        except OSError:
            logging.warning(f"spaCy model '{model}' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            self.nlp = spacy.load(model, **model_kwargs)

        self.vector_size = self.nlp.vocab.vectors_length
        if self.vector_size == 0:
            raise ValueError(
                f"The spaCy model '{model}' does not provide word vectors. "
                f"Use 'en_core_web_md' or 'en_core_web_lg'."
            )

    def set_stats(self, data: dict):
        data["stats"]["time"] += (self.name, datetime.now())
        data["stats"]["vectorization"] = self.name

    def _resolve_key(self, key: Union[str, Tuple[str, str]]):
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

    def run(self, data: dict) -> dict:
        df = self._get_data_branch(data, self.target_key)
        if df is None:
            raise ValueError(f"Input text data not found at key: {self.target_key}")

        texts = df.fillna("").astype(str).tolist()
        logging.info(f"Vectorizing {len(texts)} texts using spaCy.")

        dense_vectors = np.zeros((len(texts), self.vector_size), dtype=self.numeric_type)
        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=32)):
            dense_vectors[i] = doc.vector if doc.has_vector else 0

        if self.output_format == "sparse":
            vector_matrix = csr_matrix(dense_vectors)
        else:
            vector_matrix = dense_vectors

        self._set_data_branch(data, self.vector_key, vector_matrix)
        self._set_data_branch(data, self.output_key, list(range(len(texts))))

        logging.info(f"Stored {self.output_format} matrix in {self.vector_key}, references in {self.output_key}")
        return data
