from langdetect import detect, DetectorFactory, LangDetectException
import logging
from core.step import Step

class FilterNonEnglishStep(Step):
    name = "filter_non_english"

    def __init__(self, lang_code="en"):
        # Ensure consistent results
        DetectorFactory.seed = 0
        self.lang_code = lang_code

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]
        def is_english(text: str) -> bool:
            if not isinstance(text, str) or not text.strip():
                return False
            try:
                return detect(text) == self.lang_code
            except LangDetectException:
                return False  # Treat as non-English if fails

        logging.info(f"Filtering rows not detected as language '{self.lang_code}'...")
        initial_len = len(df)
        df = df[df["text"].apply(is_english) & df["title"].apply(is_english)]
        filtered_len = len(df)

        logging.info(f"Removed {initial_len - filtered_len} rows not in '{self.lang_code}'. Remaining: {filtered_len}")

        data["dataset"] = df
        return data