import logging
import re
import unicodedata
from html import unescape
from core.step import Step


class CleanPunctuationStep(Step):
    name = "clean_punctuation"

    def __init__(self, keep_punctuation=".,!?\"'", normalize_unicode=True):
        super().__init__()
        self.keep_punctuation = keep_punctuation
        self.normalize_unicode = normalize_unicode

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def clean_text(text: str) -> str:
            if not isinstance(text, str):
                return text

            # Unescape HTML (e.g., &quot;, <br />)
            text = unescape(text)
            text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)

            # Normalize unicode
            if self.normalize_unicode:
                text = unicodedata.normalize("NFKC", text)  # use NFKC to keep ASCII punctuation

            # Escape punctuation to keep
            escaped_keep = re.escape(self.keep_punctuation)
            
            #Remove apostrophe first without adding space
            text = text.replace("'", "")

            # Remove all other punctuation except keep list and add a space so that words arent merged
            text = re.sub(rf"[^\w\s{escaped_keep}]", " ", text)

            # Collapse multiple spaces
            text = re.sub(r"\s+", " ", text).strip()

            return text

        logging.info("Cleaning punctuation from 'title' and 'text' columns...")
        df["title"] = df["title"].apply(clean_text)
        df["text"] = df["text"].apply(clean_text)

        logging.info("Punctuation cleaning complete.")
        return data
