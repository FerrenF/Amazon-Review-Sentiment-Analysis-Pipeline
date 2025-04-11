import logging
import subprocess
import sys
from typing import Optional, Any

import spacy
from core.step import Step

class SpacyTokenizationStep(Step):
    name = "spacy_tokenization"

    def __init__(self, model: str = "en_core_web_sm", **model_kwargs):
        self.model = model
        try:
            self.nlp = spacy.load(model, **model_kwargs)
        except OSError:
            print(f"Spacy model {model} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            self.nlp = spacy.load(model, **model_kwargs)

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def tokenize(text: str) -> str:
            if not isinstance(text, str):
                return text
            doc = self.nlp(text)
            return " ".join([token.text for token in doc if not token.is_space])

        logging.info(f"Tokenizing text with spaCy model '{self.model}'...")
        df["text"] = df["text"].apply(tokenize)

        logging.info("Tokenization complete.")
        data["dataset"] = df
        return data

