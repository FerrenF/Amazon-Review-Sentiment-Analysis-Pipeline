import logging
from abc import ABC, abstractmethod
from datetime import datetime


class Step(ABC):
    name: str
    output_path: str

    @abstractmethod
    def run(self, data: dict) -> dict:
        """Run the step with current pipeline data. Return potentially modified pipeline data."""
        pass

    @abstractmethod
    def set_stats(self, data: dict):
        """Set step specific statistics for time tracking and analysis."""
        pass

    def friendly_name(self):
        return self.name.replace("_", " ").title()

    def step_log(self, message, mess_type="info"):
        if mess_type == "info":
            logging.info(f"# STEP {self.name}: '{message}'.")
        elif mess_type == "warning":
            logging.warning(message)
        elif mess_type == "error":
            logging.error(message)
