from datetime import datetime

from langdetect import detect_langs, DetectorFactory, LangDetectException
import logging
from core.step import Step

class FilterNonEnglishStep(Step):
    name = "filter_non_english"

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))

    def __init__(self, lang_code="en", min_prob=0.6,  target_keys = None):
        # Ensure consistent results
        DetectorFactory.seed = 0
        self.lang_code = lang_code
        self.min_prob = min_prob
        self.target_keys = target_keys

    def run(self, data: dict) -> dict:

        def is_english(text: str) -> bool:
            if not isinstance(text, str) or not text.strip():
                return False
            try:
                langs = detect_langs(text)
                for lang in langs:
                    if lang.lang == self.lang_code and lang.prob >= self.min_prob:
                        return True
                return False
            except LangDetectException:
                return False  # Treat as non-English if detection fails

        self.step_log(f"Filtering for '{self.lang_code}' with prob >= {self.min_prob}...")

        for key in self.target_keys:
            if isinstance(key, tuple):
                df = data[key[0]]
                col = key[1]
            else:
                df = data
                col = key

            # Create a mask of valid rows
            mask = df[col].apply(is_english)
            initial_len = len(df)
            df = df[mask]
            filtered_len = len(df)

            logging.info(f"Filtered {initial_len - filtered_len} rows from {key}. Remaining: {filtered_len}")

            # Save filtered DataFrame back
            if isinstance(key, tuple):
                data[key[0]] = df
            else:
                data = df  # this case is unusual, but covered

        return data

