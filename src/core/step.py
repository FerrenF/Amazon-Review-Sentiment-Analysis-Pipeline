from abc import ABC, abstractmethod


class Step(ABC):
    name: str

    @abstractmethod
    def run(self, data: dict) -> dict:
        """Run the step with current pipeline data. Return potentially modified pipeline data."""
        pass
