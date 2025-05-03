from langdetect import detect_langs, DetectorFactory, LangDetectException
import logging
from core.step import Step

class FilterNonEnglishStep(Step):
    name = "filter_non_english"

    def __init__(self, lang_code="en", min_prob=0.6):
        # Ensure consistent results
        DetectorFactory.seed = 0
        self.lang_code = lang_code
        self.min_prob = min_prob

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

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

        logging.info(f"Filtering rows not detected as '{self.lang_code}' with prob >= {self.min_prob}...")
        initial_len = len(df)
        df = df[df["text"].apply(is_english) & df["title"].apply(is_english)]
        filtered_len = len(df)

        logging.info(f"Removed {initial_len - filtered_len} rows not meeting language criteria. Remaining: {filtered_len}")

        data["dataset"] = df
        return data
