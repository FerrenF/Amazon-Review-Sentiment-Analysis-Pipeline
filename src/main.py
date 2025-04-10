
from src.core.pipeline import Pipeline
from src.core.stage import Stage
from src.steps.load_dataset import LoadDatasetStep
from src.utils.write_dataset import write_dataset


def loading_stage_finished_callback(stage: Stage, _data: dict):
    write_dataset(stage, "data", _data)


# The project pipeline is divided into 5 stages: Loading, Cleaning, Processing, Training, and Evaluation.

project_stages = [
    Stage("loading", [
        # During the loading stage, we import our unprocessed data and read it to be fed into the rest of the pipline.
        # Simple and easy.

        LoadDatasetStep()
    ],  on_complete=loading_stage_finished_callback),
    Stage("cleaning", [
        # During the cleaning stage, we try to make the data we are going to feed into the rest of the pipeline more
        # consistent. Here, we do such tasks as removing punctuation, filtering out data that has less than X words,
        # or performing stemming.

    ]),
    Stage("processing", [
        # When processing our cleaned data, it is time to tokenize, perform analysis of, and split the data we have into
        # a training and testing set.

    ]),
    Stage("training", [
        # Here we finally feed the cleaned and processed data into a model. The weights of the model are decided and
        # then returned and saved.

    ]),
    Stage("evaluation", [
        # Here, we use our testing set to predict a set of labels for data that only we know the true value of. The
        # results
    ])
]

data = dict()

project_pipeline = Pipeline(stages=project_stages)
project_pipeline.run(data)



