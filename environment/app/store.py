
class Store:
    def __init__(self):
        self.records = []
        self.checkpoints = {}

    def add_record(self, label: int, features: list[float]):
        self.records.append((label, features))

    def get_records(self):
        return self.records

    def save_checkpoint(self, key: str, weights: list[float], bias: float):
        pass

    def load_checkpoint(self, key: str):
        return None
