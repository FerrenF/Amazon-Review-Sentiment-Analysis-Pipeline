import re
import logging
from core.step import Step

class NormalizePunctuationStep(Step):
    name = "normalize_punctuation_step"

    def run(self, data: dict) -> dict:
        """
        Normalizes overpunctuation in the 'title' and 'text' fields of a dataset.
        Keeps at most one instance of repeated punctuation (e.g. '!!!' -> '!', '...' -> '.').
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def normalize_punctuation(text: str) -> str:
            if not isinstance(text, str):
                return text

            text = re.sub(r'([!?.,;:])\1+', r'\1', text)
            text = re.sub(r'(?<=\s)([!?.,;:])(\s\1)+(?=\s)', r'\1', text)

            return text

        logging.info("Normalizing overpunctuation in 'title' and 'text' columns...")
        df["title"] = df["title"].apply(normalize_punctuation)
        df["text"] = df["text"].apply(normalize_punctuation)

        data["dataset"] = df
        logging.info("Punctuation normalization complete.")
        return data
