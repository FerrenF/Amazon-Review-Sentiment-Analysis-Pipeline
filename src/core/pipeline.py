from .stage import Stage


class Pipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, data: dict = None):
        data = data or {}
        for stage in self.stages:
            data = stage.run(data)
        return data
