import logging
import subprocess
import sys
from datetime import datetime
from typing import Optional, Any

import spacy
from core.step import Step

class SpacyTokenizationStep(Step):
    name = "spacy_tokenization"
    description = "Spacy tokenization"

    def __init__(self, model: str = "en_core_web_sm", use_lemmas=False, remove_stops=False, **model_kwargs):
        self.model = model
        self.use_lemmas = use_lemmas
        self.remove_stops = remove_stops
        try:
            self.nlp = spacy.load(model, **model_kwargs)
        except OSError:
            print(f"Spacy model {model} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            self.nlp = spacy.load(model, **model_kwargs)

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))
        data["stats"]["tokenization"] = self.name

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def tokenize(text: str) -> str:
            if not isinstance(text, str):
                return text
            doc = self.nlp(text)
            return " ".join([token.text for token in doc if not (token.is_space or (self.remove_stops and token.is_stop))])

        def tokenize_and_lemmatize(text):
            if not isinstance(text, str):
                return []
            doc = self.nlp(text)
            return " ".join([token.lemma_.lower() for token in doc if not token.is_punct and not (token.is_space or (self.remove_stops and token.is_stop))])

        self.step_log(f"Tokenizing text with '{self.model}'...")
        df["text"] = df["text"].apply(tokenize if not self.use_lemmas else tokenize_and_lemmatize)
        data["vectors"] = {} # Prep for next step
        data["dataset"] = df
        return data

