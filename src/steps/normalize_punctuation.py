import re
from core.step import Step
import logging

class NormalizeOverpunctuationStep(Step):
    name = "overpunctuation_normalizer"

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        def normalize_punctuation(text: str) -> str:
            if not isinstance(text, str):
                return text
            # Normalize repeating punctuation: ... → ., !! → !, etc.
            text = re.sub(r"([!?.,])\1{1,}", r"\1", text)
            return text

        logging.info("Normalizing overpunctuation in 'title' and 'text' fields...")
        df = data["dataset"]
        df["title"] = df["title"].apply(normalize_punctuation)
        df["text"] = df["text"].apply(normalize_punctuation)

        data["dataset"] = df
        return data
