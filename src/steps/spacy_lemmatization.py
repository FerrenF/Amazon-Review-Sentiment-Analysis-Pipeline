import logging
import subprocess
import sys

import spacy
from core.step import Step


class SpacyLemmatizationStep(Step):
    name = "spacy_lemmatization"

    def __init__(self, model: str = "en_core_web_sm", pos_keep: list[str] = None, **model_kwargs):
        """
        :param model: The spaCy model to use, e.g., "en_core_web_sm"
        :param pos_keep: List of POS tags to keep (e.g., ['NOUN', 'VERB', 'ADJ']). If None, keeps all.
        """
        self.model = model
        self.pos_keep = pos_keep
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

        def lemmatize(text: str) -> str:
            if not isinstance(text, str):
                return text
            doc = self.nlp(text)
            if self.pos_keep:
                # Only keep tokens that match the selected POS tags
                return " ".join(
                    token.lemma_ for token in doc
                    if not token.is_space and token.pos_ in self.pos_keep
                )
            else:
                return " ".join(
                    token.lemma_ for token in doc if not token.is_space
                )

        logging.info(f"Lemmatizing text with spaCy model '{self.model}'...")
        if self.pos_keep:
            logging.info(f"Filtering lemmas by POS tags: {self.pos_keep}")
        else:
            logging.info("Keeping all POS tags.")

        df["text"] = df["text"].apply(lemmatize)

        logging.info("Lemmatization complete.")
        data["dataset"] = df
        return data
