import datetime
import logging
import pathlib

from common.project_common import OUTPUT_DATA_DIR, LOGGING_NAME, PIPELINE_NAME, PROCESSED_DATA_DIR
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

    logging.info("Saving keys: " + ",".join(_data.keys()))
    pickle_dataset(stage, "data", _data)


# Parameter grids for automated hyperparameter tuning.
logistic_param_grid = {
    'penalty': ['l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs', 'sag', 'saga', 'newton-cg'],
        'max_iter': [100, 200, 300]
}
logistic_best_param_7707 = {'C': [1], 'max_iter': [100], 'penalty': ['l2'], 'solver': ['saga']}
svr_param_grid = {
    'C': [200], #[0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1],
    'gamma': ['scale'],#, 'auto', 0.01, 0.1],
    'kernel': ['poly','rbf']
}
svc_param_grid = {
    'C': [10,200,300],
    'gamma': ['scale'],#, 'auto', 0.01, 0.1],
    'kernel': ['rbf']#, 'poly']
}
svc_best_7700 = {'C': [200], 'gamma': ['scale'], 'kernel': ['rbf']}
rf_classifier_param_grid = {
    "n_estimators": [250, 350, 600],
    "max_depth": [None, 25],
    "min_samples_split": [10, 15],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"]
}
lr_param_grid = {
            "penalty": ["l2"],
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear", "lbfgs", "sag", "saga", "newton-cg"],
            "max_iter": [100, 200, 300],
        }
lr_param_grid_best_7700 = {'C': [1], 'max_iter': [100], 'penalty': ['l2'], 'solver': ['saga']}
rf_classifier_best_7700 = {'max_depth': [25], 'max_features': ['sqrt'], 'min_samples_leaf': [1], 'min_samples_split': [15], 'n_estimators': [350]}
rf_classifier_best_20k = {'max_depth': [25], 'min_samples_leaf': [1], 'min_samples_split': [5], 'n_estimators': [400]}
knn_param_grid = {
            "n_neighbors": [2, 3, 5, 10],
            "weights": ["distance"],
            "metric": ["manhattan", "euclidean"],
}

# The project pipeline is divided into 5 stages: Loading, Cleaning, Processing, Training, and Evaluation.
# It can be divided into more, but is kept this way for simplicity.
project_stages = [
    Stage("loading", [

        # During the loading stage, we import our unprocessed data and read it to be fed into the rest of the pipline.
        # Simple and easy.

        # LoadCheckpointIfExists:
        # Load data from a previous run's checkpoint and then skip to the next appropriate stage in the pipeline.
        # LoadCheckpointIfExists("cleaning", "data", is_pickle=False),
         LoadCheckpointIfExists(stage_name="processing", checkpoint_dir=PROCESSED_DATA_DIR, checkpoint_basename="data_processing"),
        # CleanDatasetStep:
        # Delete data from the previous run if we aren't using checkpoints.

        CleanDatasetStep(),

        # LoadDatasetStep:
        # Actually load the data
        LoadDatasetStep()
    ],  on_complete=stage_finished_callback),
    Stage("cleaning", [

        # During the cleaning stage, we try to make the data we are going to feed into the rest of the pipeline more
        # consistent. Here, we do such tasks as removing punctuation, filtering out data that has less than X words,
        # or performing stemming.

        RemoveHTMLTagsStep(),
        SymbolSeparationStep(),
        ApplyWordThresholdStep(min_length = 3, max_length = 150),

        ExpandContractionsStep(),

        # Better for spaCy tokenization
        #CleanPunctuationStep(keep_punctuation=".,!?\"'-", normalize_unicode=True),

        # Better for BOW and TF-IDF
        CleanPunctuationStep(keep_punctuation="!?", normalize_unicode=True, target_keys=[("dataset","title"),("dataset","text")]),

        NormalizeOverPunctuationStep(target_keys=[ ("dataset", "text"), ("dataset", "title")], symbols="", exclude_symbols=True),

        RemoveAmznNoiseTokensStep(target_keys=[ ("dataset", "text"), ("dataset", "title")]),
        SpellCheckStep(target_keys=[ ("dataset", "text"), ("dataset", "title")], max_distance=1),
        FilterNonEnglishStep(target_keys=[ ("dataset", "text")], min_prob=0.6),

        SpaceAndBalanceQuotesStep(target_keys=[ ("dataset", "text"), ("dataset", "title")]),
        TokenMergeCorrectionStep(target_keys=[ ("dataset", "text"), ("dataset", "title")]),
        WhitespaceTrimmingStep(target_keys=[ ("dataset", "text"), ("dataset", "title")]),


        #CombineTextColumnsStep(target_keys=[ ("dataset", "text"), ("dataset", "title")], output_key=("dataset","text"), separator=" "),

    ],  on_complete=stage_finished_callback),
    Stage("processing", [
        # When processing our cleaned data, it is time to remove stopwords if needed, chain negations, lemmatize, tokenize,
        # perform analysis of, and extract numeric features from the text.
        SpacyTokenizationStep(model="en_core_web_sm", remove_stops=False, use_lemmas=True, disable=["parser", "ner"]),
        ChainWordQualifiersStep(model="en_core_web_sm", max_chain_length={
            "intensifier": '2',
            "qualifier": '2',
            "negator": '2',
        }, target_keys=[("dataset","text")], disable=["parser", "ner"]),
        RemoveStopWordsStep(),
        #TfidfVectorizationStep(target_key=("dataset","text"), output_key=("vectors", "sparse")),
        #BagOfWordsVectorizationStep(target_key=("dataset","text"), output_key=("vectors","sparse")),
        SpacyVectorizationStep(model="en_core_web_md", target_key=("dataset","text"), vector_key=("vectors", "sparse"), output_format="sparse", output_key=("dataset","vectors")),
        ScaleVectorsStep(target_column=("vectors","sparse"), output_column=("vectors","sparse")),
        #VectorNormalizationStep(output_format="sparse", vector_key=("vectors","sparse"), output_key=("vectors","normalized")),

    ],  on_complete=stage_finished_pickler_callback),
    Stage("training", [

        # Here we finally split, balance, and then feed the cleaned and processed data into a model.

        # First, we might want a 3 class model instead of a 5 class model. Here's our chance to fire a few techniques at that problem.
        # We can... Remap it:

            #RemapLabelsStep(mapping={2: 3, 4: 3}),

        # Or we can drop it entirely:

            DropLabelsStep(labels_to_drop=[2,4], target_key="label", vector_key=("vectors","sparse")),


        # Then we split the dataset up for holding out.

            TrainTestSplitStep(test_size=0.2, random_state=42, target_x=("vectors","sparse"), target_y=("dataset","label"), text_key=("dataset","text")),


        # We may still need to perform label balancing. The difference in oversampling versus undersampling is great.

            #BalanceLabelsStep(sample_method="oversample", targets=("X_train", "y_train"), oversample_cap=2500),
            BalanceLabelsStep(sample_method="undersample", targets=("X_train", "y_train")),

        #GaussNaiveBayesClassificationStep(grid_search=True),
        MultinomialNaiveBayesClassificationStep(grid_search=True),
        #KNearestNeighborsClassificationStep(grid_search=True, param_grid=knn_param_grid),
        #RandomForestClassificationStep(grid_search=True, param_grid=rf_classifier_best_7700),
        #LogisticRegressionStep(grid_search=True, param_grid=logistic_best_param_7707)
        #SupportVectorClassificationStep(grid_search=True, param_grid=svc_best_7700),

    ],  on_complete=stage_finished_pickler_callback),
    Stage("evaluation", [
        # Here, we use our testing set to predict a set of labels for data that only we know the true value of.
        ClassificationEvaluationStep(metrics=["f1", "accuracy", "precision", "recall"]),
        OutputPredictionsStep(save_to_file=True, filename=OUTPUT_DATA_DIR.joinpath(PIPELINE_NAME +datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")+"predictions.json")),
        OutputStatisticsStep(output_dir=OUTPUT_DATA_DIR, filename="statistics.jsonl")
    ])
]

out_path = OUTPUT_DATA_DIR.joinpath(PIPELINE_NAME +datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")+ LOGGING_NAME)

logging.basicConfig(filename=out_path, filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
data = dict()

project_pipeline = Pipeline(stages=project_stages)
project_pipeline.run(data)



