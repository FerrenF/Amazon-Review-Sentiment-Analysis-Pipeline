import logging
from datetime import datetime
from typing import Optional, Union, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from core.step import Step


class BagOfWordsVectorizationStep(Step):
    name = "bag_of_words_vectorization"

    def __init__(self,
                 max_features: Optional[int] = None,
                 stop_words: Optional[str] = "english",
                 target_key: Union[str, Tuple[str, str]] = ("dataset", "text"),
                 output_key: Union[str, Tuple[str, str]] = "X_bow",
                 **vectorizer_kwargs):
        """
        Initializes a BoW vectorization step using sklearn's CountVectorizer.

        :param max_features: Optional limit on the number of features.
        :param stop_words: Optional stop words to filter (default: "english").
        :param target_key: Target key for which to vectorize.
        :param output_key: The target key to output vectors on.
        :param vectorizer_kwargs: Additional CountVectorizer keyword arguments.
        """
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            **vectorizer_kwargs
        )
        self.target_key = target_key
        self.output_key = output_key

    def set_stats(self, data:dict):
        data["stats"]["time"] += (self.name, datetime.now())
        data["stats"]["vectorization"] = self.name

    def run(self, data: dict) -> dict:

        """
        Vectorizes the <target_key> key of the object into a sparse matrix using bag-of-words.
        Stores it under 'X_bow' and retains the original DataFrame.
        """

        df = data[self.target_key] if not isinstance(self.target_key, tuple) else data[self.target_key[0]][self.target_key[1]]
        texts = df.fillna("").astype(str).tolist()

        logging.info("Vectorizing text using Bag-of-Words...")
        X_bow = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        #dense_vectors = X_bow.toarray()
        #df["vector"] = list(dense_vectors) Too much memory usage for dense vectors
        if not isinstance(self.output_key, tuple):
            data[self.output_key] = X_bow
        else:
            data[self.output_key[0]][self.output_key[1]] = X_bow

        data["vector_features"] = feature_names
        data["vectorizer"] = self.vectorizer
        logging.info(f"Bag-of-Words vectorization complete. Saved in {self.output_key}.")

        return data
