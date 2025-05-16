import logging

from common.project_common import PIPELINE_NAME
from .stage import Stage


class Pipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, data: dict = None):
        logging.info(f"=== PIPELINE BEGIN: {PIPELINE_NAME} ===")
        data = data or {}
        data["stats"] = dict()
        for stage in self.stages:
            data = stage.run(data)
        return data
