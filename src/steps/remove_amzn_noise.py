import re
import logging
from core.step import Step

class RemoveAmznNoiseTokensStep(Step):
    name = "remove_noise_tokens"

    def run(self, data: dict) -> dict:
        """
        Removes noisy tokens that amazon reviews have like ASINs, VIDEOIDs, and attempts to find long hash strings.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def remove_noise_tokens(text):
            if not isinstance(text, str):
                return text

            # Remove VIDEOIDs
            text = re.sub(r'\bVIDEOID[\w\d]+\b', '', text)
            # Remove ASIN codes
            text = re.sub(r'\bASIN[\w\d]+\b', '', text)
            # Remove standalone long alphanumeric codes in general (16+ chars)
            # THIS COULD CATCH REAL WORDS, theoretically. Statistically speaking though, it's a good idea.
            text = re.sub(r'\b[a-fA-F0-9]{16,}\b', '', text)
            return text

        logging.info("Removing noise tokens: ASIN, VIDEOID, and long hashes from 'title' and 'text' columns...")
        df["title"] = df["title"].apply(remove_noise_tokens)
        df["text"] = df["text"].apply(remove_noise_tokens)

        data["dataset"] = df
        logging.info("Noise tokens removed from 'title' and 'text'.")
        return data
