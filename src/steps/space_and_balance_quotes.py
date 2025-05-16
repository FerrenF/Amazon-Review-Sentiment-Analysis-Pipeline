import re
import logging
from datetime import datetime
from core.step import Step

class SpaceAndBalanceQuotesStep(Step):

    def __init__(self, target_keys=None):
        """
        Initializes a step that fixes spacing and balance issues around double quotes.

        :param target_keys: A string, a tuple, or a list of strings/tuples indicating which keys to clean.
        """
        self.name = "space_and_balance_quotes"
        self.description = "Fixes spacing and balance issues around quotes (e.g., adds spaces, collapses double quotes, balances lone trailing quotes)."

        if target_keys is None:
            raise ValueError("You must provide target_keys to specify what to clean.")
        if isinstance(target_keys, (str, tuple)):
            self.target_keys = [target_keys]
        elif isinstance(target_keys, list):
            self.target_keys = target_keys
        else:
            raise TypeError("target_keys must be a string, tuple, or list of strings/tuples.")

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))

    def run(self, data: dict) -> dict:
        def fix_quotes(text: str) -> str:
            if not isinstance(text, str):
                return text

            # Add space before and after a quote stuck to a word
            text = re.sub(r'(\w)"', r'\1 "', text)
            text = re.sub(r'"(\w)', r'" \1', text)

            # Collapse double quotes (e.g., `""word""` → `"word"`)
            text = re.sub(r'""+', '"', text)

            # Naively balance quotes by removing lone trailing quote
            if text.count('"') % 2 != 0:
                text = text.rstrip('"')

            return text.strip()

        self.step_log(f'{self.friendly_name()} : {self.description}')

        for key in self.target_keys:
            if isinstance(key, str):
                if key not in data:
                    raise KeyError(f"Key '{key}' not found in data.")
                self.step_log(f"Fixing quotes in data['{key}']")
                data[key] = data[key].apply(fix_quotes)

            elif isinstance(key, tuple) and len(key) == 2:
                outer, inner = key
                if outer not in data:
                    raise KeyError(f"Key '{outer}' not found in data.")
                if inner not in data[outer].columns:
                    raise KeyError(f"Column '{inner}' not found in data['{outer}']")
                self.step_log(f"Fixing quotes in data['{outer}']['{inner}']")
                data[outer][inner] = data[outer][inner].apply(fix_quotes)

            else:
                raise TypeError(f"Unsupported target key format: {key}")

        return data
