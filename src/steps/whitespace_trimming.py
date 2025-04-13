import logging
import re
from core.step import Step


class WhitespaceTrimmingStep(Step):
    name = "whitespace_trimming_step"

    def run(self, data: dict) -> dict:
        """
        Trims excessive whitespace in the 'title' and 'text' columns.
        This removes leading/trailing whitespace and reduces multiple consecutive spaces to a single space.

        :param data: A dict that contains a 'dataset' key with a pandas DataFrame
        :return: the modified 'data' dict
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def trim_whitespace(text: str) -> str:
            if not isinstance(text, str):
                return text
            # Remove leading/trailing spaces and replace multiple spaces with a single space
            text = text.strip()
            text = re.sub(r'\s{2,}', ' ', text)
            return text

        logging.info("Trimming excessive whitespace in dataset...")
        df["title"] = df["title"].apply(trim_whitespace)
        df["text"] = df["text"].apply(trim_whitespace)

        logging.info("Whitespace trimming complete.")
        return data
