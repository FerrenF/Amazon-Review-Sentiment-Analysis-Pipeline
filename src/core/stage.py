import src.core.stage
from src.core.step import Step
from typing import Callable, Optional


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

    def run(self, data: dict) -> dict:
        print(f"=== Running stage: {self.name} ===")
        for step in self.steps:
            print(f"> Running step: {step.__class__.__name__}")
            data = step.run(data)

        if self.on_complete:
            self.on_complete(self, data)

        return data
