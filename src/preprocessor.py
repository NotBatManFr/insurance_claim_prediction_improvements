import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from .interfaces import IPreprocessor
from typing import Tuple
from .config import AppConfig

class Preprocessor(IPreprocessor):
    """
    Preprocessor that uses FeatureConfig for column treatment.
    
    Args:
        feature_config: Configuration defining which columns get which preprocessing
    """
    
    def __init__(self, app_config: AppConfig) -> None:
        self.feature_config = app_config.features
        self.app_config = app_config

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Convert raw dataframe into model-ready feature matrix X and target y.

        Behavior:
        - Drops identifier columns if present.
        - Converts Yes/No columns to booleans (missing -> False).
        - Converts numeric columns to numeric dtype (coercing errors to NaN).
        - Ordinal-encodes columns from config.
        - One-hot encodes categorical columns.
        - Ensures the target column exists and returns (X, y).

        Args:
            df: Raw input pandas DataFrame

        Returns:
            Tuple of (X: pd.DataFrame, y: pd.Series[int])

        Raises:
            ValueError: if required target column is missing.
        """
        data = df.copy()

        # 1. Drop identifiers
        data.drop(columns = self.feature_config.id_columns, errors='ignore', inplace=True)

        # 2. Boolean Conversion
        for col in self.feature_config.boolean_columns:
            if col in data.columns:
                data[col] = data[col].map({'Yes': True, 'No': False}).fillna(False)

        # 3. Float Conversion 
        # Coerce to numeric and ensure float dtype for consistency
        for col in self.feature_config.float_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').astype(float)

        # 4. Ordinal Encoding
        for col, categories in self.feature_config.ordinal_columns.items():
            if col in data.columns:
                data[col] = data[col].astype(str)
                enc = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1)
                data[col] = enc.fit_transform(data[[col]])
                data.drop(columns=["ncap_rating"], inplace=True)

        # 5. One-Hot Encoding
        if self.feature_config.categorical_columns:
            data = pd.get_dummies(data, columns=[c for c in self.feature_config.categorical_columns if c in data.columns], drop_first=True)

        # 6. Separate Target Variable
        target_col = self.app_config.target_column
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' missing from dataset")
            
        X = data.drop(columns=[target_col])
        y = data[target_col].astype(int)

        return X, y
