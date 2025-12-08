import pandas as pd
from .interfaces import IDataLoader

class CSVLoader(IDataLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at: {self.file_path}")
