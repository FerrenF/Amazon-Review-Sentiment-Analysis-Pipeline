import logging
import spacy
from core.step import Step

class RemoveStopWordsStep(Step):
    name = "remove_stop_words"

    def __init__(self, model: str = "en_core_web_sm", disable=["parser", "ner"]):
        try:
            self.nlp = spacy.load(model, disable=disable)
        except OSError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            self.nlp = spacy.load(model, disable=disable)

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        def remove_stops(text):
            if not isinstance(text, str):
                return text
            doc = self.nlp(text)
            return " ".join([
                token.text for token in doc
                if not token.is_stop and not token.is_space
            ])

        logging.info("Removing stop words from 'title' and 'text' columns...")
        for col in ["title", "text"]:
            if col in df.columns:
                df[col] = df[col].apply(remove_stops)

        logging.info("Stop word removal complete.")
        data["dataset"] = df
        return data
