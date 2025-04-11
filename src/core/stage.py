import logging
from typing import Callable, Optional
from .step import Step


class Stage:
    def __init__(
        self,
        name: str,
        steps: list[Step],
        on_complete: Optional[Callable[['Stage', dict], None]] = None
    ):
        """
        :param name: Name of the stage
        :param steps: List of Step instances
        :param on_complete: Optional callback function called after all steps are run.
                            It receives the final `data` dict as an argument.
        """
        self.name = name
        self.steps = steps
        self.on_complete = on_complete


    def is_skipping(self, data: dict) -> bool:
        skip_to = data.get("skip_to_stage", None)

        # If skipping is in place and the current stage is not the target, skip it
        if skip_to is not None:
            if self.name != skip_to:
                data["skip_ended"] = False
                return True
            else:
                data["skip_to_stage"] = None
                data["skip_ended"] = True
                return True
        else:
            data["skip_ended"] = True
            return False

    def run(self, data: dict) -> dict:


        if self.is_skipping(data):
            logging.info(f"Skipping stage '{self.name}' due to checkpoint load.")
            return data

        print(f"=== Running stage: {self.name} ===")


        for step in self.steps:
            if self.is_skipping(data):
                return data
            print(f"> Running step: {step.__class__.__name__}")
            data = step.run(data)

        if self.on_complete:
            self.on_complete(self, data)


        return data
