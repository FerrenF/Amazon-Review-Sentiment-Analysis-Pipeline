import re
import logging
from datetime import datetime

from core.step import Step

class RemoveAmznNoiseTokensStep(Step):
    name = "remove_noise_tokens"

    def __init__(self, target_keys=None):
        """
            Removes some junk amazon uses to identify pictures and video IDs.

             :param target_keys: A string, tuple, or list of strings/tuples specifying where to find text to clean.
                    Example: "text", ("dataset", "text"), ["title", ("dataset", "text")]
        """
        self.target_keys = target_keys

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))

    def run(self, data: dict) -> dict:
        """
        Removes noisy tokens that amazon reviews have like ASINs, VIDEOIDs, and attempts to find long hash strings.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

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

        for key in self.target_keys:
            self.step_log(f"Removing noise tokens from {key}")
            if isinstance(key, tuple):
                data[key[0]][key[1]] = data[key[0]][key[1]].apply(remove_noise_tokens)
            elif isinstance(key, str):
                data[key[0]] = data[key[0]].apply(remove_noise_tokens)

        return data
