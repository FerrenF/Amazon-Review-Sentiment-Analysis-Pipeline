import logging
from typing import Callable, Optional
from .step import Step


class Stage:
    output_path: str

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

    def stage_log(self, message, messg_type="info"):
        if messg_type == "info":
            logging.info(f"# STAGE {self.name}: '{message}'.")
        elif messg_type == "warning":
            logging.warning(message)
        elif messg_type == "error":
            logging.error(message)

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
            self.stage_log(f"SKIP '{self.name}' due to checkpoint load.")
            return data

        self.stage_log(f"BEGIN")

        for step in self.steps:
            if self.is_skipping(data):
                return data
            self.stage_log(f"STEP BEGIN: {step.__class__.__name__}")
            step.set_stats(data)
            data = step.run(data)
            self.stage_log(f"STEP COMPLETE: {step.__class__.__name__}")

        if self.on_complete:
            self.on_complete(self, data)


        return data
