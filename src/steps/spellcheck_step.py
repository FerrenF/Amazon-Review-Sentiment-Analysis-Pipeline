from spellchecker import SpellChecker
import logging
from src.core.step import Step

class SpellCheckStep(Step):
    name = "spellcheck_step"

    def run(self, data: dict) -> dict:
        """
        Corrects spelling in 'title' and 'text' fields using pyspellchecker.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]
        spell = SpellChecker(distance=1)

        def correct_sentence(sentence: str) -> str:
            if not isinstance(sentence, str):
                return sentence

            words = sentence.split()
            corrected = []

            for word in words:
                correction = spell.correction(word)
                # Fallback to original word if correction is None
                corrected.append(correction if correction is not None else word)

            return " ".join(corrected)

        logging.info("Running spellcheck on 'title' and 'text' columns...")
        df["title"] = df["title"].apply(correct_sentence)
        df["text"] = df["text"].apply(correct_sentence)

        data["dataset"] = df
        logging.info("Spellcheck complete.")
        return data
