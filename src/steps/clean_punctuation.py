import logging
import re
from datetime import datetime
import unicodedata
from html import unescape
from core.step import Step


class CleanPunctuationStep(Step):
    name = "clean_punctuation"
    description = "Cleans punctuation from text."

    def __init__(self, keep_punctuation=".,!?\"'", normalize_unicode=True, target_keys=None):
        """
        :param keep_punctuation: Punctuation characters to preserve.
        :param normalize_unicode: Whether to normalize Unicode to ASCII-friendly forms.
        :param target_keys: A string, tuple, or list of strings/tuples specifying where to find text to clean.
                            Example: "text", ("dataset", "text"), ["title", ("dataset", "text")]
        """
        self.keep_punctuation = keep_punctuation
        self.normalize_unicode = normalize_unicode

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
        def clean_text(text: str) -> str:
            if not isinstance(text, str):
                return text

            text = unescape(text)
            text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)

            if self.normalize_unicode:
                text = unicodedata.normalize("NFKC", text)

            escaped_keep = re.escape(self.keep_punctuation)

            # Remove apostrophes first without adding space
            text = text.replace("'", "")

            # Replace other punctuation with space
            text = re.sub(rf"[^\w\s{escaped_keep}]", " ", text)

            return re.sub(r"\s+", " ", text).strip()

        self.step_log(f'{self.friendly_name()} : {self.description}')
        for key in self.target_keys:
            if isinstance(key, str):
                if key not in data:
                    raise KeyError(f"Key '{key}' not found in data.")
                self.step_log(f"Cleaning data['{key}']")
                data[key] = data[key].apply(clean_text)

            elif isinstance(key, tuple) and len(key) == 2:
                outer, inner = key
                if outer not in data:
                    raise KeyError(f"Key '{outer}' not found in data.")
                if inner not in data[outer].columns:
                    raise KeyError(f"Column '{inner}' not found in data['{outer}']")
                self.step_log(f"Cleaning data['{outer}']['{inner}']")
                data[outer][inner] = data[outer][inner].apply(clean_text)

            else:
                raise TypeError(f"Unsupported target key format: {key}")

        return data
