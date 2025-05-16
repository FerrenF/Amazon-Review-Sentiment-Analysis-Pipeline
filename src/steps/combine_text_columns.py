import logging
from datetime import datetime

from core.step import Step
import pandas as pd


class CombineTextColumnsStep(Step):
    name = "combine_text_columns"

    def __init__(self, separator: str = ". ", target_keys=None, output_key=("dataset", "text")):
        """
        :param separator: The string to insert between text segments when combining.
        :param target_keys: A list of keys (string or tuple) to retrieve Series from data.
        :param output_key: A string or tuple specifying where to store the combined result.
        """
        self.separator = separator
        self.description = "Combines text values from multiple keys using a separator."
        self.target_keys = target_keys or []
        self.output_key = output_key

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))

    def _resolve_key(self, data, key):
        if isinstance(key, tuple):
            ref = data
            for k in key:
                ref = ref[k]
            return ref
        return data[key]

    def _assign_key(self, data, key, value):
        if isinstance(key, tuple):
            ref = data
            for k in key[:-1]:
                ref = ref[k]
            ref[key[-1]] = value
        else:
            data[key] = value

    def run(self, data: dict) -> dict:
        """
        Combines text fields from multiple data keys into a single output location.
        """
        def sanitize(val):
            return val if isinstance(val, str) else ""

        def combine_all(parts):
            result = ""
            for part in parts:
                part = sanitize(part)
                if result and part:
                    if result.endswith(('!', '.', '?')):
                        result += " " + part
                    else:
                        result += self.separator + part
                elif part:
                    result = part
            return result

        # Retrieve all series to combine
        series_list = []
        for key in self.target_keys:
            value = self._resolve_key(data, key)
            if not isinstance(value, pd.Series):
                raise TypeError(f"Expected pandas Series at {key}, but got {type(value).__name__}")
            series_list.append(value)

        # Combine row-wise
        combined = pd.Series([
            combine_all([s.iloc[i] for s in series_list])
            for i in range(len(series_list[0]))
        ], index=series_list[0].index)

        self._assign_key(data, self.output_key, combined)

        return data
