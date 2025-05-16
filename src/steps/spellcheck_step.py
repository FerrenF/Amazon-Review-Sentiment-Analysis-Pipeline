from datetime import datetime

from spellchecker import SpellChecker
import logging
import re
from core.step import Step

class SpellCheckStep(Step):
    name = "spellcheck_step"

    def __init__(self, max_distance = 2, target_keys = None):
        """
                  Removes some junk amazon uses to identify pictures and video IDs.

                   :param target_keys: A string, tuple, or list of strings/tuples specifying where to find text to clean.
                          Example: "text", ("dataset", "text"), ["title", ("dataset", "text")]
              """
        if target_keys is None:
            target_keys = [("dataset", "text"), ("dataset", "title")]
        self.max_distance = max_distance
        self.target_keys = target_keys

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))
        data["stats"]["spellcheck_distance"] = self.max_distance

    def run(self, data: dict) -> dict:
        """
            Corrects spelling in 'title' and 'text' fields using pyspellchecker.
            Preserves punctuation, contractions, and compound words with hyphens.
        """
        spell = SpellChecker(distance=self.max_distance)

        def correct_sentence(sentence: str) -> str:
            if not isinstance(sentence, str):
                return sentence

            # Pre-tokenize with support for apostrophes and hyphens within words
            tokens = re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", sentence, re.UNICODE)

            corrected = []
            for i, token in enumerate(tokens):
                if re.fullmatch(r"\w+(?:[-']\w+)*", token):
                    # Preserve proper nouns (capitalized, not at start of sentence)
                    if token[0].isupper() and i > 0:
                        corrected.append(token)
                    else:
                        corrected.append(spell.correction(token) or token)
                else:
                    corrected.append(token)

            # Reconstruct sentence with correct spacing, hopefully...
            text = ""
            for i, token in enumerate(corrected):
                if i > 0:
                    prev = corrected[i - 1]

                    # No space between quotes/apostrophes and their neighboring words
                    if token in {"'", '"'} or prev in {"'", '"'}:
                        pass  # no space
                    # Space after word followed by word (e.g., "this is")
                    elif re.fullmatch(r"\w+(?:[-']\w+)*", prev) and re.fullmatch(r"\w+(?:[-']\w+)*", token):
                        text += " "
                    # Space after punctuation if followed by word
                    elif prev in {'.', ',', '!', '?', ':', ';'}:
                        text += " "

                text += token

            return text.strip()

        self.step_log(f"Spellchecking: {self.target_keys}")

        for key in self.target_keys:
            if isinstance(key, tuple):
                data[key[0]][key[1]] =  data[key[0]][key[1]].apply(correct_sentence)
            if isinstance(key, str):
                data[key] = data[key].apply(correct_sentence)

        return data
