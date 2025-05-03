from spellchecker import SpellChecker
import logging
import re
from core.step import Step

class SpellCheckStep(Step):
    name = "spellcheck_step"

    def __init__(self, max_distance = 2):
        self.max_distance = max_distance

    def run(self, data: dict) -> dict:
        """
        Corrects spelling in 'title' and 'text' fields using pyspellchecker.
        Preserves punctuation, contractions, and compound words with hyphens.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]
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

        logging.info("Spellchecking 'title' and 'text' columns with punctuation preservation...")
        df["title"] = df["title"].apply(correct_sentence)
        df["text"] = df["text"].apply(correct_sentence)

        data["dataset"] = df
        logging.info("Spellcheck (with punctuation safe mode) complete.")
        return data
