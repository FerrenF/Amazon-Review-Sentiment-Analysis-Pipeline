import logging
import re
from datetime import datetime

from core.step import Step

class WhitespaceTrimmingStep(Step):

    def __init__(self, target_keys=None):
        """
        Initializes the step that trims excessive whitespace in selected keys.

        :param target_keys: A string, a tuple, or a list of strings/tuples indicating which keys to clean.
        """
        self.name = "whitespace_trimming_step"
        self.description = "Trims leading/trailing whitespace and reduces multiple spaces to a single space."

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
        """
        Trims excessive whitespace in the specified fields.
        Removes leading/trailing whitespace and collapses multiple spaces to one.
        """

        def trim_whitespace(text: str) -> str:
            if not isinstance(text, str):
                return text
            text = text.strip()
            text = re.sub(r'\s{2,}', ' ', text)
            return text

        self.step_log(f'{self.friendly_name()} : {self.description}')

        for key in self.target_keys:
            if isinstance(key, str):
                if key not in data:
                    raise KeyError(f"Key '{key}' not found in data.")
                self.step_log(f"Trimming whitespace in data['{key}']")
                data[key] = data[key].apply(trim_whitespace)

            elif isinstance(key, tuple) and len(key) == 2:
                outer, inner = key
                if outer not in data:
                    raise KeyError(f"Key '{outer}' not found in data.")
                if inner not in data[outer].columns:
                    raise KeyError(f"Column '{inner}' not found in data['{outer}']")
                self.step_log(f"Trimming whitespace in data['{outer}']['{inner}']")
                data[outer][inner] = data[outer][inner].apply(trim_whitespace)

            else:
                raise TypeError(f"Unsupported target key format: {key}")

        return data
