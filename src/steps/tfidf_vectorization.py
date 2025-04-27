import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from core.step import Step


class TfidfVectorizationStep(Step):
    name = "tfidf_vectorization"

    def __init__(self, max_features: Optional[int] = None, stop_words: Optional[str] = "english", **vectorizer_kwargs):
        """
        Initializes a term frequency inverse document frequency (TF-IDF) vectorization step using sklearn's TfidfVectorizer.

        :param max_features: Optional limit on the number of features.
        :param stop_words: Stop words to filter (default: "english"). Uses sklearn's built-in stop word removal.
        :param vectorizer_kwargs: Additional TfidfVectorizer keyword arguments.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            **vectorizer_kwargs
        )

    def run(self, data: dict) -> dict:
        """
        Vectorizes the 'text' column of the dataset into a sparse matrix using TF-IDF.
        Stores it under 'X_tfidf' and retains the original DataFrame.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]
        if "text" not in df.columns:
            raise ValueError("'text' column not found in dataset.")

        texts = df["text"].fillna("").astype(str).tolist()

        logging.info("Vectorizing text using TF-IDF...")
        X_tfidf = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        dense_vectors = X_tfidf.toarray()

        df["vector"] = list(dense_vectors) # the dense vector is used for training
        data["X_tfidf"] = X_tfidf  # keep the sparse version too just in case
        data["vector_features"] = feature_names
        data["vectorizer"] = self.vectorizer
        logging.info("TF-IDF vectorization complete.")

        return data
