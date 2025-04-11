import regex
import logging
from core.step import Step

class SymbolSeparationStep(Step):
    name = "separate_symbols"

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def extract_symbols(text):
            if not isinstance(text, str):
                return ""
            # Extract symbols and emojis only — avoid replacing with anything!
            matches = regex.findall(r'\p{So}|\p{Sk}|\p{Sc}|\p{Sm}|\p{Mn}|\p{Me}', text)
            return ''.join(matches)

        def extract_all_symbols(row):
            title_symbols = extract_symbols(row['title'])
            text_symbols = extract_symbols(row['text'])
            return title_symbols + text_symbols

        df["symbols"] = df.apply(extract_all_symbols, axis=1)
        df["symbols"] = df["symbols"].astype("string")

        df[["title", "text", "symbols"]].head(1000).to_csv("symbols_debug_sample.txt", index=False)
        logging.info("Symbols successfully extracted into the 'symbols' column.")
        data["dataset"] = df
        return data
