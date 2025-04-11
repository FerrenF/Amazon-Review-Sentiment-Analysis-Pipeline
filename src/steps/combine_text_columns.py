import logging
from core.step import Step

class CombineTextColumnsStep(Step):
    name = "combine_text_columns"

    def __init__(self, separator: str = ". "):
        """
        :param separator: The string to insert between title and text when combining.
        """
        self.separator = separator

    def run(self, data: dict) -> dict:
        """
        Combines 'title' and 'text' columns into a single 'combined' column using the provided separator.

        :param data: A dict containing a 'dataset' key with a pandas DataFrame.
        :return: The modified 'data' dict with a new 'combined' column.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def combine(title, text):
            title = title if isinstance(title, str) else ""
            text = text if isinstance(text, str) else ""
            if title and text:
                return f"{title}{self.separator}{text}"
            return title or text  # Only one may be present

        logging.info(f"Combining 'title' and 'text' into 'text' using separator: '{self.separator}'...")
        df["text"] = df.apply(lambda row: combine(row["title"], row["text"]), axis=1)

        logging.info("Text column combination complete.")
        data["dataset"] = df
        return data
