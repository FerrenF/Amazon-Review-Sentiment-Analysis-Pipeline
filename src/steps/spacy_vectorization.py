import logging
import subprocess
import sys
from typing import Optional, Any, Type

import spacy
import numpy as np
import pandas as pd
from core.step import Step


class SpacyVectorizationStep(Step):
    name = "spacy_vectorization"

    def __init__(self, model="en_core_web_md", numeric_type: Type = np.float32, **model_kwargs):
        """
               Initializes the BoW vectorization step using sklearn's CountVectorizer.

               :param model: The model to use (e.g. en_core_web_sm/en_core_web_md).
               :param numeric_type: The numpy numeric type to case vector components to.
               :param model_kwargs: Additional keyword arguments.
               """
        self.model = model
        self.numeric_type = numeric_type
        try:
            self.nlp = spacy.load(model, **model_kwargs)
        except OSError:
            print(f"spaCy model '{model}' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            self.nlp = spacy.load(model, **model_kwargs)

        self.vector_size = self.nlp.vocab.vectors_length
        if self.vector_size == 0:
            raise ValueError(
                f"The spaCy model '{model}' does not have word vectors. "
                f"Use 'en_core_web_md' or 'en_core_web_lg' for vectorization."
            )

    def run(self, data: dict) -> dict:
        """
        Uses spaCy to convert text into dense vector representations.
        Adds each vector component as a separate column to the DataFrame.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]
        texts = df["text"].fillna("").astype(str).tolist()

        logging.info("Vectorizing text using spaCy with nlp.pipe (batched)...")

        vectors = []
        for doc in self.nlp.pipe(texts, batch_size=32):
            vectors.append(doc.vector if doc.has_vector else np.zeros(self.vector_size, dtype=self.numeric_type))

        df["vector"] = vectors
        data["dataset"] = df
        logging.info("Spacy vectorization complete.")
        return data
