import logging
from sklearn.model_selection import train_test_split
from core.step import Step


class TrainTestSplitStep(Step):
    name = "train_test_split_step"

    def __init__(
        self,
        test_size,
        random_state,
        perform_stratification=True,
        target_x="vector",  # configurable input key
        target_y="label",   # configurable target key
        text_key="text"     # optional key for splitting original text
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.perform_stratification = perform_stratification
        self.target_x = target_x
        self.target_y = target_y
        self.text_key = text_key

    def run(self, data: dict) -> dict:
        """
        Splits the dataset into train/test sets using configurable X and y targets from the data dict.
        Also tracks the text column if available.
        """
        logging.info(f"Splitting dataset into train/test sets using X='{self.target_x}' and y='{self.target_y}'...")

        X = data[self.target_x] if not isinstance(self.target_x, tuple) else data[self.target_x[0]][self.target_x[1]]
        y = data[self.target_y] if not isinstance(self.target_y, tuple) else data[self.target_y[0]][self.target_y[1]]


        if (not isinstance(self.text_key, tuple) and self.text_key in data) or (self.text_key[1] in data[self.text_key[0]]):
            text = data[self.text_key] if not isinstance(self.text_key, tuple) else data[self.text_key[0]][self.text_key[1]]
            (
                X_train, X_test,
                y_train, y_test,
                text_train, text_test
            ) = train_test_split(
                X, y, text,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if self.perform_stratification else None
            )
            data["text_train"] = text_train
            data["text_test"] = text_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if self.perform_stratification else None
            )

        data["X_train"] = X_train
        data["X_test"] = X_test
        data["y_train"] = y_train
        data["y_test"] = y_test

        logging.info(f"Train/test split complete. Train size: {len(y_train)}, Test size: {len(y_test)}")
        return data
