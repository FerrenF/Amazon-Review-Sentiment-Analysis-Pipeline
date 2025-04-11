import logging
import re
import unicodedata
from core.step import Step

class CleanPunctuationStep(Step):
    name = "clean_punctuation"

    def __init__(self, keep_punctuation=None, normalize_unicode=True):
        """
        :param keep_punctuation: Optional string of punctuation characters to keep (e.g., "!?.,").
        :param normalize_unicode: Whether to normalize unicode characters into ASCII (NFKD + encode-decode).
        """
        super().__init__()
        self.keep_punctuation = keep_punctuation or ""
        self.normalize_unicode = normalize_unicode

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def clean_text(text: str) -> str:
            if not isinstance(text, str):
                return text

            if self.normalize_unicode:
                # Normalize to NFKD and remove (unicode distinct) diacritics by encoding to ASCII
                text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")

            # Build regex to remove all punctuation except specified
            if self.keep_punctuation:
                escaped_keep = re.escape(self.keep_punctuation)
                pattern = rf"[^\w\s{escaped_keep}]"
            else:
                pattern = r"[^\w\s]"

            return re.sub(pattern, "", text)

        logging.info("Cleaning punctuation from 'title' and 'text' columns...")
        df["title"] = df["title"].apply(clean_text)
        df["text"] = df["text"].apply(clean_text)

        logging.info("Punctuation cleaning complete.")
        return data
