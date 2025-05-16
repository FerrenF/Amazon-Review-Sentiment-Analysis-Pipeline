from datetime import datetime
from core.step import Step
from spellchecker import SpellChecker
import re
import logging


class TokenMergeCorrectionStep(Step):

    def __init__(self, target_keys=None):
        self.name = "token_merge_correction_step"
        self.description = "Attempts to correct merged tokens in long words (e.g., 'thisisgreat' → 'this is great')"

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
        spell = SpellChecker()
        word_set = set(spell.word_frequency.words())

        def split_word(word):
            if word in word_set or len(word) <= 3:
                return [word]
            for i in range(len(word), 3, -1):
                first, rest = word[:i], word[i:]
                if first in word_set:
                    return [first] + split_word(rest)
            return [word]

        def fix_merged(text: str) -> str:
            if not isinstance(text, str):
                return text
            tokens = re.findall(r"\b\w{10,}\b", text)
            for token in tokens:
                split = split_word(token)
                if len(split) > 1 and all(part in word_set for part in split):
                    text = re.sub(rf"\b{re.escape(token)}\b", " ".join(split), text)
            return text

        self.step_log(f'{self.friendly_name()} : {self.description}')

        for key in self.target_keys:
            if isinstance(key, str):
                if key not in data:
                    raise KeyError(f"Key '{key}' not found in data.")
                logging.info(f"Fixing merged tokens in data['{key}']")
                data[key] = data[key].apply(fix_merged)

            elif isinstance(key, tuple) and len(key) == 2:
                outer, inner = key
                if outer not in data:
                    raise KeyError(f"Key '{outer}' not found in data.")
                if inner not in data[outer].columns:
                    raise KeyError(f"Column '{inner}' not found in data['{outer}']")
                logging.info(f"Fixing merged tokens in data['{outer}']['{inner}']")
                data[outer][inner] = data[outer][inner].apply(fix_merged)

            else:
                raise TypeError(f"Unsupported target key format: {key}")

        return data
