
from src.core.pipeline import Pipeline
from src.core.stage import Stage
from src.steps.load_dataset import LoadDatasetStep
from src.steps.apply_word_threshold import ApplyWordThresholdStep
from src.utils.write_dataset import write_dataset


def loading_stage_finished_callback(stage: Stage, _data: dict):
    write_dataset(stage, "data", _data)
def cleaning_stage_finished_callback(stage: Stage, _data: dict):
    write_dataset(stage, "data", _data)



project_stages = [
    Stage("loading", [
        LoadDatasetStep()
    ],  on_complete=loading_stage_finished_callback),
    Stage("cleaning", [
        ApplyWordThresholdStep(min_length = 2, max_length = 50)

    ], on_complete=cleaning_stage_finished_callback),
    Stage("processing", [

    ]),
    Stage("training", [

    ]),
    Stage("evaluation", [

    ])
]

data = dict()

project_pipeline = Pipeline(stages=project_stages)
project_pipeline.run(data)



