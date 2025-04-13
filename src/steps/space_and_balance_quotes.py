import re
import logging
from core.step import Step

class SpaceAndBalanceQuotesStep(Step):
    name = "space_and_balance_quotes"

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        def fix_quotes(text: str) -> str:
            if not isinstance(text, str):
                return text

            # Add space before quote if it's stuck to a word
            text = re.sub(r'(\w)"', r'\1 "', text)
            text = re.sub(r'"(\w)', r'" \1', text)

            # Collapse double quotes (e.g., `""word""` → `"word"`)
            text = re.sub(r'""+', '"', text)

            # Optional: balance quotes — naive heuristic (don’t attempt deep fix here)
            if text.count('"') % 2 != 0:
                text = text.rstrip('"')  # Remove lone trailing quote
            return text.strip()

        logging.info("Fixing spacing around quotes in 'title' and 'text' fields...")
        df = data["dataset"]
        df["title"] = df["title"].apply(fix_quotes)
        df["text"] = df["text"].apply(fix_quotes)

        data["dataset"] = df
        return data
