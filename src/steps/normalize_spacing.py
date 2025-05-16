import re
from datetime import datetime

from core.step import Step
import logging

class NormalizeSpacingStep(Step):
    name = "quote_apostrophe_spacing_fix"
    description = "Apostrophe's spacing fix"

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        def fix_spacing(text: str) -> str:
            if not isinstance(text, str):
                return text
            # Remove spaces after apostrophes or quotes
            text = re.sub(r"(['\"])\s+", r"\1", text)  # ' s → 's
            text = re.sub(r"\s+(['\"])", r"\1", text)  # s ' → s'
            return text

        self.step_log("Fixing spacing around quotes and apostrophes...")
        df = data["dataset"]
        df["title"] = df["title"].apply(fix_spacing)
        df["text"] = df["text"].apply(fix_spacing)

        data["dataset"] = df
        return data
