import logging
import contractions
from core.step import Step

class ExpandContractionsStep(Step):
    name = "expand_contractions"

    def run(self, data: dict) -> dict:
        """
        Expands contractions like "can't" → "cannot". Uses contractions library (requirement)
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        def expand(text):
            if not isinstance(text, str):
                return text
            return contractions.fix(text)

        logging.info("Expanding contractions in 'title' and 'text' columns...")
        df["title"] = df["title"].apply(expand)
        df["text"] = df["text"].apply(expand)

        data["dataset"] = df
        logging.info("Contractions expanded.")
        return data
