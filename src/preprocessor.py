import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from .interfaces import IPreprocessor
from typing import Tuple
from .config import AppConfig

class InsurancePreprocessor(IPreprocessor):
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Convert raw dataframe into model-ready feature matrix X and target y.

        Behavior:
        - Drops identifier columns such as ``policy_id`` if present.
        - Converts "Yes"/"No" boolean-like columns to booleans (missing -> False).
        - Converts numeric-like columns to numeric dtype (coercing errors to NaN).
        - Ordinal-encodes ``ncap_rating`` into ``NCAP_Rating`` **if present** (no error if missing).
        - One-hot encodes several categorical columns.
        - Ensures the target column ``is_claim`` exists and returns (X, y).

        Args:
            df: Raw input pandas DataFrame

        Returns:
            Tuple of (X: pd.DataFrame, y: pd.Series[int])

        Raises:
            ValueError: if required target column ``is_claim`` is missing.
        """
        data = df.copy()

        # 1. Drop identifiers
        data.drop(columns = self.config.drop_cols, errors='ignore', inplace=True)

        # 2. Boolean Conversion
        for col in self.config.bool_cols:
            if col in data.columns:
                data[col] = data[col].map({'Yes': True, 'No': False}).fillna(False)

        # 3. Float Conversion 
        # Coerce to numeric and ensure float dtype for consistency
        for col in self.config.num_cols:
            if col in data.columns
                data[col] = pd.to_numeric(data[col], errors='coerce').astype(float)

        # 4. Ordinal Encoding
        for col, categories in self.config.ordinal_cols.items():
            if col in data.columns:
                data[col] = data[col].astype(str)
                enc = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=0)
                data[col] = enc.fit_transform(data[[col]])
                data.drop(columns=["ncap_rating"], inplace=True) <-------------

        # 5. One-Hot Encoding
        if self.config.categorical_cols:
            data = pd.get_dummies(data, columns=[c for c in self.config.categorical_cols if c in data.columns], drop_first=True)

        # 6. Separate Target Variable
        target_col = self.config.target_column
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' missing from dataset")
            
        X = data.drop(columns=[target_col])
        y = data[target_col].astype(int)

        return X, y
