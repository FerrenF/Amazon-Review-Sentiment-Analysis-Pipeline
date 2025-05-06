import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from core.step import Step


class BagOfWordsVectorizationStep(Step):
    name = "bag_of_words_vectorization"

    def __init__(self, max_features: Optional[int] = None, stop_words: Optional[str] = "english", **vectorizer_kwargs):
        """
        Initializes a BoW vectorization step using sklearn's CountVectorizer.

        :param max_features: Optional limit on the number of features.
        :param stop_words: Optional stop words to filter (default: "english").
        :param vectorizer_kwargs: Additional CountVectorizer keyword arguments.
        """
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            **vectorizer_kwargs
        )

    def run(self, data: dict) -> dict:
        """
        Vectorizes the 'text' column of the dataset into a sparse matrix using bag-of-words..
        Stores it under 'X_bow' and retains the original DataFrame.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]
        if "text" not in df.columns:
            raise ValueError("'text' column not found in dataset.")

        texts = df["text"].fillna("").astype(str).tolist()

        logging.info("Vectorizing text using Bag-of-Words...")
        X_bow = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        #dense_vectors = X_bow.toarray()
        #df["vector"] = list(dense_vectors) Too much memory usage for dense vectors

        data["X_bow"] = X_bow  # keep the sparse version in case it's needed
        data["vector_features"] = feature_names
        data["vectorizer"] = self.vectorizer
        logging.info("Bag-of-Words vectorization complete.")

        return data
