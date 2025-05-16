import re
from datetime import datetime
from core.step import Step
import logging


class NormalizeOverPunctuationStep(Step):

    DEFAULT_SYMBOLS = "!?.:,;'" + "-"

    def __init__(self, target_keys=None, symbols=None, exclude_symbols=False):
        """
        Initializes a step that normalizes repeating punctuation such as '...', '???', '!!!!', or '----'.

        :param target_keys: A string, a tuple, or a list of strings/tuples indicating which keys to clean.
        :param symbols: A string of symbols to include or exclude, depending on exclude_symbols.
        :param exclude_symbols: If True, 'symbols' will be excluded from normalization.
        """
        self.name = "over_punctuation_normalization"
        self.description = "Normalizes repeating punctuation characters (e.g., '...', '???') into single instances."

        if target_keys is None:
            raise ValueError("You must provide target_keys to specify what to clean.")
        if isinstance(target_keys, (str, tuple)):
            self.target_keys = [target_keys]
        elif isinstance(target_keys, list):
            self.target_keys = target_keys
        else:
            raise TypeError("target_keys must be a string, tuple, or list of strings/tuples.")

        if symbols is None:
            self.symbols = self.DEFAULT_SYMBOLS
        else:
            self.symbols = symbols

        if exclude_symbols:
            self.symbols = ''.join(s for s in self.DEFAULT_SYMBOLS if s not in self.symbols)

        self._regex = re.compile(rf"([{re.escape(self.symbols)}])\1+")

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))

    def run(self, data: dict) -> dict:
        def normalize_over_punctuation(text: str) -> str:
            if not isinstance(text, str):
                return text
            return self._regex.sub(r"\1", text)

        self.step_log(f'{self.friendly_name()} : {self.description}')

        for key in self.target_keys:
            if isinstance(key, str):
                if key not in data:
                    raise KeyError(f"Key '{key}' not found in data.")
                self.step_log(f"Normalizing punctuation in data['{key}']")
                data[key] = data[key].apply(normalize_over_punctuation)

            elif isinstance(key, tuple) and len(key) == 2:
                outer, inner = key
                if outer not in data:
                    raise KeyError(f"Key '{outer}' not found in data.")
                if inner not in data[outer].columns:
                    raise KeyError(f"Column '{inner}' not found in data['{outer}']")
                self.step_log(f"Normalizing punctuation in data['{outer}']['{inner}']")
                data[outer][inner] = data[outer][inner].apply(normalize_over_punctuation)

            else:
                raise TypeError(f"Unsupported target key format: {key}")

        return data
