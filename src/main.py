import logging

from core import *
from steps import *
from utils import write_dataset, pickle_dataset


def stage_finished_callback(stage: Stage, _data: dict):
    path = write_dataset(stage, "data", _data)
    if path:
        logging.info(f"Checkpoint saved for stage '{stage.name}' at {path}")
    else:
        logging.warning(f"Failed to save checkpoint for stage '{stage.name}'")

def stage_finished_pickler_callback(stage: Stage, _data: dict):
   pickle_dataset(stage, "data", _data)

# The project pipeline is divided into 5 stages: Loading, Cleaning, Processing, Training, and Evaluation.

project_stages = [
    Stage("loading", [
        # During the loading stage, we import our unprocessed data and read it to be fed into the rest of the pipline.
        # Simple and easy.
        LoadCheckpointIfExists("processing", "data", is_pickle=True),
        CleanDatasetStep(),
        LoadDatasetStep()
    ],  on_complete=stage_finished_callback),
    Stage("cleaning", [
        # During the cleaning stage, we try to make the data we are going to feed into the rest of the pipeline more
        # consistent. Here, we do such tasks as removing punctuation, filtering out data that has less than X words,
        # or performing stemming.
        RemoveHTMLTagsStep(),
        SymbolSeparationStep(),
        CleanPunctuationStep("!?.,", True),
        ApplyWordThresholdStep(min_length = 3, max_length = 50),
        LowercasingStep(),
        WhitespaceTrimmingStep(),
        SpellCheckStep(),
        CombineTextColumnsStep()
    ],  on_complete=stage_finished_callback),
    Stage("processing", [
        # When processing our cleaned data, it is time to remove stopwords if needed, lemmatize, tokenize,
        # perform analysis of, and extract numeric features from the text.
        SpacyTokenizationStep(model="en_core_web_sm", disable=["parser", "ner"]),
        SpacyLemmatizationStep(model="en_core_web_sm", disable=["parser", "ner"]),
        SpacyVectorizationStep(model="en_core_web_md"),
        NormalizeVectorsStep(),
        BalanceLabelsStep(),

    ],  on_complete=stage_finished_pickler_callback),
    Stage("training", [
        # Here we finally split and then feed the cleaned and processed data into a model. The weights of the model are decided and
        # then returned and saved.
        TrainTestSplitStep(test_size=0.2, random_state=42),
        RandomForestRegressionStep()
    ]),
    Stage("evaluation", [
        # Here, we use our testing set to predict a set of labels for data that only we know the true value of.
        EvaluationStep(metrics=["mae", "mse", "r2"]),
        OutputPredictionsStep(save_to_file=True, filename="predictions.json")
    ])
]

data = dict()

project_pipeline = Pipeline(stages=project_stages)
project_pipeline.run(data)



