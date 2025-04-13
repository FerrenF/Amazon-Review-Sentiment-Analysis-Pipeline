import logging
import string
import re
import unicodedata

from core.step import Step


class ArtifactRemovalStep(Step):
    name = "artifact_removal_step"

    def run(self, data: dict) -> dict:
        """
        Attempts to remove miscellaneous artifacts from text, such as:
        - digits before and after a word with no space inbetween.

        :param data: A dict that contains a 'dataset' key with a pandas DataFrame
        :return: the modified 'data' dict
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def clean_text(text: str) -> str:
            if not isinstance(text, str):
                return text
            return re.sub(r'\d+(?=\w)|(?<=\w)\d+', '', text)

        logging.info("Applying artifact removal to dataset...")
        df["title"] = df["title"].apply(clean_text)
        df["text"] = df["text"].apply(clean_text)

        logging.info("Artifact removal complete.")
        return data
