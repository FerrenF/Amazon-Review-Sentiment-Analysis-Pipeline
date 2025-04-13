import re
import logging
from core.step import Step


class HyphenChainNormalizerStep(Step):
    name = "hyphen_chain_normalizer_step"

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        def normalize_hyphens(text: str) -> str:
            if not isinstance(text, str):
                return text
            # Collapse sequences of 2+ hyphens into a single hyphen
            return re.sub(r"-{2,}", "-", text)

        logging.info("Normalizing hyphen chains in 'title' and 'text' fields...")
        df = data["dataset"]
        df["title"] = df["title"].apply(normalize_hyphens)
        df["text"] = df["text"].apply(normalize_hyphens)

        data["dataset"] = df
        return data
