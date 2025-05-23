import logging
from datetime import datetime
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from core.step import Step


class TfidfVectorizationStep(Step):
    name = "tfidf_vectorization"

    def __init__(self,
                 max_features: Optional[int] = None,
                 stop_words: Optional[str] = "english",
                 target_key: Union[str, Tuple[str, str]] = ("dataset", "text"),
                 output_key: Union[str, Tuple[str, str]] = "X_bow",
                 sparse_output: bool = True,
                 **vectorizer_kwargs):
        """
        Initializes a TF-IDF vectorization step using sklearn's TfidfVectorizer.

        :param max_features: Optional limit on the number of features.
        :param stop_words: Stop words to filter (default: "english").
        :param target_key: Key or (dict_key, subkey) tuple where the text data resides.
        :param output_key: Key or (dict_key, subkey) tuple to store the output.
        :param sparse_output: Whether to store the output as a sparse matrix (default: True).
        :param vectorizer_kwargs: Additional keyword arguments passed to TfidfVectorizer.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            **vectorizer_kwargs
        )
        self.target_key = target_key
        self.output_key = output_key
        self.sparse_output = sparse_output

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))
        data["stats"]["vectorization"] = self.name

    def run(self, data: dict) -> dict:
        """
        Vectorizes the 'text' column of the dataset into a sparse or dense matrix using TF-IDF.
        Stores the result and retains the original DataFrame.
        """
        df = data[self.target_key] if not isinstance(self.target_key, tuple) else data[self.target_key[0]][self.target_key[1]]

        texts = df.fillna("").astype(str).tolist()

        logging.info("Vectorizing text using TF-IDF...")
        X_tfidf = self.vectorizer.fit_transform(texts)

        if not self.sparse_output:
            X_tfidf = X_tfidf.toarray()

        feature_names = self.vectorizer.get_feature_names_out()

        if not isinstance(self.output_key, tuple):
            data[self.output_key] = X_tfidf
        else:
            data[self.output_key[0]][self.output_key[1]] = X_tfidf

        data["vector_features"] = feature_names
        data["vectorizer"] = self.vectorizer
        logging.info(f"TF-IDF vectorization complete. Stored {'sparse' if self.sparse_output else 'dense'} matrix in {self.output_key}")

        return data
