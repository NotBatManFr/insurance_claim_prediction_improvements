from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple

class IDataLoader(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

class IPreprocessor(ABC):
    @abstractmethod
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        pass

class IModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass
