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


# Parameter grids for automated hyperparameter tuning.
logistic_param_grid = {
    'penalty': ['l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs', 'sag', 'saga', 'newton-cg'],
        'max_iter': [100, 200, 300]
}
svr_param_grid = {
    'C': [100, 200], #[0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.2],
    'gamma': ['scale'],#, 'auto', 0.01, 0.1],
    'kernel': ['poly']#['rbf', 'poly']
}
svc_param_grid = {
    'C': [200],  # [0.1, 1, 10, 100],
    'gamma': ['scale'],#, 'auto', 0.01, 0.1],
    'kernel': ['poly']#['rbf', 'poly']
}
rf_classifier_param_grid = {
    "n_estimators": [300, 400],
    "max_depth": [None, 25],
    "min_samples_split": [5],
    "min_samples_leaf": [1, 5]
}
knn_param_grid = {
            "n_neighbors": [3],
            "weights": ["distance"],
            "metric": ["manhattan"]
}

# The project pipeline is divided into 5 stages: Loading, Cleaning, Processing, Training, and Evaluation.
# It can be divided into more, but is kept this way for simplicity.
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
        ApplyWordThresholdStep(min_length = 3, max_length = 200),
        CleanPunctuationStep(keep_punctuation=".,!?\"'-", normalize_unicode=True),
        NormalizePunctuationStep(),
        HyphenChainNormalizerStep(),
        WhitespaceTrimmingStep(),
        SpellCheckStep(),
        SpaceAndBalanceQuotesStep(),
        TokenMergeCorrectionStep(),
        CombineTextColumnsStep()
    ],  on_complete=stage_finished_callback),
    Stage("processing", [
        # When processing our cleaned data, it is time to remove stopwords if needed, lemmatize, tokenize,
        # perform analysis of, and extract numeric features from the text.
        SpacyTokenizationStep(model="en_core_web_sm", disable=["parser", "ner"]),
        TfidfVectorizationStep(),
        #BagOfWordsVectorizationStep(),
        #SpacyVectorizationStep(model="en_core_web_md"),
        ScaleVectorsStep(),
        #NormalizeVectorsStep(),
        BalanceLabelsStep(sample_method="oversample"),

    ],  on_complete=stage_finished_pickler_callback),
    Stage("training", [
        # Here we finally split and then feed the cleaned and processed data into a model. The weights of the model are decided and
        # then returned and saved.
        TrainTestSplitStep(test_size=0.2, random_state=42),
        #GaussNaiveBayesClassificationStep(grid_search=True),
        #MultinomialNaiveBayesClassificationStep(grid_search=True),
        KNearestNeighborsClassificationStep(grid_search=True, param_grid=knn_param_grid),
        #RandomForestClassificationStep(grid_search=True, param_grid=rf_classifier_param_grid),
        #SupportVectorClassificationStep(grid_search=True, param_grid=svc_param_grid),

    ],  on_complete=stage_finished_pickler_callback),
    Stage("evaluation", [
        # Here, we use our testing set to predict a set of labels for data that only we know the true value of.
        ClassificationEvaluationStep(metrics=["f1", "accuracy"]),
        OutputPredictionsStep(save_to_file=True, filename="predictions.json")
    ])
]

logging.basicConfig(filename="pipeline_log.txt", filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', level=logging.INFO)

data = dict()

project_pipeline = Pipeline(stages=project_stages)
project_pipeline.run(data)



