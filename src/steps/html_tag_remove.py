import re
import logging
from core.step import Step

class RemoveHTMLTagsStep(Step):
    name = "remove_html_tags"

    def run(self, data: dict) -> dict:
        """
        Removes HTML tags from 'title' and 'text' fields using regular expressions.
        """
        if "dataset" not in data:
            raise ValueError("No dataset found in data. Ensure LoadDatasetStep ran successfully.")

        df = data["dataset"]

        def remove_html_tags(text):
            if not isinstance(text, str):
                return text
            # Remove HTML tags using regex
            clean_text = re.sub(r'<.*?>', '', text)
            return clean_text

        logging.info("Removing HTML tags from 'title' and 'text' columns...")
        df["title"] = df["title"].apply(remove_html_tags)
        df["text"] = df["text"].apply(remove_html_tags)

        data["dataset"] = df
        logging.info("HTML tags removed from 'title' and 'text'.")
        return data