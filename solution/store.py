
class Store:
    def __init__(self):
        self.records = []
        self.checkpoints = {}

    def add_record(self, label: int, features: list[float]):
        self.records.append((label, features))

    def get_records(self):
        return self.records

    def save_checkpoint(self, key: str, weights: list[float], bias: float):
        self.checkpoints[key] = (list(weights), bias)

    def load_checkpoint(self, key: str):
        if key in self.checkpoints:
            w, b = self.checkpoints[key]
            return list(w), b
        return None
