import logging
from datetime import datetime

import contractions
from core.step import Step

class ExpandContractionsStep(Step):


    def __init__(self):
        self.name = "expand_contractions"
        self.description = "Expands contractions (can't -> can not or cannot)."

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))

    def run(self, data: dict) -> dict:
        """
        Expands contractions like "can't" → "cannot". Uses contractions library (requirement)
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        self.step_log(f'{self.friendly_name()} : {self.description}')

        def expand(text):
            if not isinstance(text, str):
                return text
            return contractions.fix(text)

        df["title"] = df["title"].apply(expand)
        df["text"] = df["text"].apply(expand)

        data["dataset"] = df
        return data
