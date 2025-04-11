import logging
import string
import re
import unicodedata

from core.step import Step


class LowercasingStep(Step):
    name = "lowercasing_step"

    def run(self, data: dict) -> dict:
        """
        Changes all uppercase letters to lower case.

        :param data: A dict that contains a 'dataset' key with a pandas DataFrame
        :return: the modified 'data' dict
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def lowercase_text(text: str) -> str:
            if not isinstance(text, str):
                return text
            return text.lower()

        logging.info("Applying lowercasing to dataset...")
        df["title"] = df["title"].apply(lowercase_text)
        df["text"] = df["text"].apply(lowercase_text)

        logging.info("Lowercasing complete.")
        return data
