import logging
import string
import re
import unicodedata

from src.core.step import Step


class CleanPunctuationStep(Step):
    name = "clean_punctuation"

    def run(self, data: dict) -> dict:
        """
        Removes punctuation characters from 'title' and 'text' fields in the dataset.
        Does not remove emojis and symbols.

        :param data: A dict that contains a 'dataset' key with a pandas DataFrame
        :return: the modified 'data' dict
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def clean_text(text: str) -> str:
            if not isinstance(text, str):
                return text

            # Normalize text to NFKD form to split ligatures and accented chars
            text = unicodedata.normalize("NFKD", text)

            # Remove all punctuation using Unicode regex
            return re.sub(r"[^\w\s]", "", text)

        logging.info("Cleaning punctuation from 'title' and 'text' columns...")
        df["title"] = df["title"].apply(clean_text)
        df["text"] = df["text"].apply(clean_text)

        logging.info("Punctuation cleaning complete.")
        return data
