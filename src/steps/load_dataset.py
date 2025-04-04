from src.core.step import Step


class LoadDatasetStep(Step):
    name = "load_dataset"

    def run(self, data):
        # Simulated dataset load
        data["dataset"] = [1, 2, 3, 4, 5]
        return data
