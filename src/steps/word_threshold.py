import json
import os
import pathlib
import logging
from datetime import datetime

from core.step import Step

import pandas as pd
# We want to extend our Step class, and ensure that we have defined all the methods within Step so that our base
# pipeline can treat all of these steps the same way.


class ApplyWordThresholdStep(Step):
    name = "apply_threshold"
    
    def __init__(self, min_length: int, max_length: int):
        #Initializing the min and max attributes
        self.min_length = min_length
        self.max_length = max_length

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))
        data["stats"]["filtered_min_word_length"] = self.min_length
        data["stats"]["filtered_max_word_length"] = self.max_length

    def filter_length(self, text: str) -> bool:
        #Word count instead of text count
        length = len(text.split())
        return self.min_length <= length <= self.max_length


    def run(self, data: dict) -> dict:
        """
        This step performs cleaning on data by removing punctuation and filtering short/long text

        :param data: A dict, whose 'dataset' contains the data to be cleaned
        :return: the modified 'data' dict
        """
        
        df = data.get("dataset")
        if df is None:
            raise ValueError("No dataset found in 'data' parameter")
        
        self.step_log(f"Total rows at the beginning of cleaning: {len(df)}")
        
        #Iterate through dataframe and remove data points that dont meet the length filter
        filtered_df = df[df['text'].apply(self.filter_length)]
        
        rows_removed = len(df) - len(filtered_df)
        self.step_log(f"Cleaning complete, {rows_removed} rows removed and {len(filtered_df)} rows remaining.")

        data["dataset"] = filtered_df
        
        return data