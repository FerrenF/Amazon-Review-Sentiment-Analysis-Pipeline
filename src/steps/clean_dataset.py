import json
import os
import pathlib

from src.core.step import Step
from src.project_common import *

import pandas as pd
# We want to extend our Step class, and ensure that we have defined all the methods within Step so that our base
# pipeline can treat all of these steps the same way.


class CleanDatasetStep(Step):
    name = "clean_dataset"
    
    def __init__(self, min_length: int, max_length: int):
        #Initializing the min and max attributes
        self.min_length = min_length
        self.max_length = max_length

    
    def filter_length(self, text: str) -> bool:
        length = len(text)
        
        if length >= self.min_length and length <= self.max_length:
            return True
        else:
            return False


    def run(self, data: dict) -> dict:
        """
        This step performs cleaning on data by removing punctuation and filtering short/long text

        :param data: A dict, whose 'dataset' contains the data to be cleaned
        :return: the modified 'data' dict
        """
        
        df = data.get("dataset")
        if df is None:
            raise ValueError("No dataset found in 'data' parameter")
        
        logging.info(f"Total rows at the beginning of cleaning: {len(df)}")
        
        #Iterate through dataframe and remove data points that dont meet the length filter
        filtered_df = df[df['text'].apply(self.filter_length)]
        
        rows_removed = len(df) - len(filtered_df)
        logging.info(f"Cleaning complete, {rows_removed} rows removed and {len(filtered_df)} rows remaining.")
        
        #Lowercase text and title columns
        filtered_df['text'] = filtered_df['text'].str.lower()
        filtered_df['title'] = filtered_df['title'].str.lower()
        
        data["dataset"] = filtered_df
        
        return data
        
    
