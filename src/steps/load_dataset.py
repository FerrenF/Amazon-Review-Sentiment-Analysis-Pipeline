import json
import pathlib
from datetime import datetime

from core.step import Step
from common.project_common import *

import pandas as pd
# We want to extend our Step class, and ensure that we have defined all the methods within Step so that our base
# pipeline can treat all of these steps the same way.


class LoadDatasetStep(Step):
    name = "load_dataset"

    def set_stats(self, data: dict):
        data["stats"]["time"].append((self.name, datetime.now()))

    def run(self, data: dict) -> dict:
        """
        This step finds all jsonl files in the unprocessed data directory and then uses pandas to load this data into
        the passed in data dictionary before returning it.

        :param data: A dict, whose 'dataset' field will be set to the retrieved data
        :return: the modified 'data' dict
        """
        path_object = pathlib.Path(UNPROCESSED_DATA_DIR)
        if not path_object.exists():
            raise FileNotFoundError(f"Input path for unprocessed data at {UNPROCESSED_DATA_DIR} does not exist.")

        out_dataset = None
        # Iterate through objects in the directory.
        for child in path_object.iterdir():

            # If the child's extension matches .jsonl, then include it in our dataset
            if child.match("*" + UNPROCESSED_EXT):
                logging.info(f"Found data file: {str(child)}")

                # Our lines look like:
                #   {"label": 1,
                #   "rating": 5.0,
                #   "title": "",
                #   "text": "",
                #   "images": [],
                #   "asin": "",
                #   "parent_asin": "",
                #   "user_id": "",
                #   "timestamp": 1610961834416,
                #   "helpful_vote": 0,
                #   "verified_purchase": true}

                # We only care about label, rating, title and text. We will only read those.
                cols = ['label', 'rating', 'title', 'text']

                # More about dtypes at https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dtypes
                dtypes = {'label': 'Int8','rating': 'Float32','title': 'string','text': 'string' }

                rows = []
                # First, let's use json library to load the information we need into a list.

                with open(child, encoding='utf-8') as f:
                    for line in f:

                        try:
                            j_line = json.loads(line)
                        except json.JSONDecodeError as e:
                            logging.warning(f"Skipping invalid JSON line: {line} ({e})")
                            continue

                        rows.append([j_line['label'], j_line['rating'], j_line['title'], j_line['text']])

                self.step_log(f"{len(rows)} lines loaded from file. Adding to dataset.")

                # Make a new dataframe with appropriate types on the first occurance.
                if out_dataset is None:
                    out_dataset = pd.DataFrame(data=rows, columns=cols)
                    out_dataset.astype(dtypes)
                else:
                    out_dataset = pd.concat([out_dataset, pd.DataFrame(data=rows, columns=cols)])
                data["stats"]["beginning_count"] = len(out_dataset)

        data["dataset"] = out_dataset
        self.step_log(f"Total {out_dataset.shape[0]} entries in dataset.")
        return data
