from core.step import Step
from spellchecker import SpellChecker
import re
import logging


class TokenMergeCorrectionStep(Step):
    name = "token_merge_correction_step"

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        spell = SpellChecker()
        word_set = set(spell.word_frequency.words())

        def split_merged_words(text: str) -> str:
            if not isinstance(text, str):
                return text

            def split_word(word):
                if word in word_set or len(word) <= 3:
                    return [word]

                for i in range(len(word), 0, -1):
                    first, rest = word[:i], word[i:]
                    if first in word_set:
                        return [first] + split_word(rest)
                return [word]  # fallback

            def fix_merged(text):
                tokens = re.findall(r"\b\w{10,}\b", text)  # long words that are likely merged
                for token in tokens:
                    split = split_word(token)
                    if len(split) > 1:
                        text = text.replace(token, " ".join(split))
                return text

            return fix_merged(text)

        logging.info("Correcting merged tokens in 'title' and 'text' fields...")
        df = data["dataset"]
        df["title"] = df["title"].apply(split_merged_words)
        df["text"] = df["text"].apply(split_merged_words)

        data["dataset"] = df
        return data
