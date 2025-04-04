class Stage:
    def __init__(self, name: str, steps: list[Step]):
        self.name = name
        self.steps = steps

    def run(self, data: dict) -> dict:
        print(f"=== Running stage: {self.name} ===")
        for step in self.steps:
            print(f"> Running step: {step.__class__.__name__}")
            data = step.run(data)
        return data

