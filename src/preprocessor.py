import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from .interfaces import IPreprocessor
from typing import Tuple

class InsurancePreprocessor(IPreprocessor):
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        data = df.copy()

        # 1. Drop identifiers
        if "policy_id" in data.columns:
            data.drop(columns=["policy_id"], inplace=True)

        # 2. Boolean Conversion
        boolean_cols = [
            "is_parking_camera", "is_tpms", "is_adjustable_steering", "is_esc",
            "is_parking_sensors", "is_front_fog_lights", "is_rear_window_wiper",
            "is_rear_window_washer", "is_rear_window_defogger", "is_brake_assist",
            "is_power_door_locks", "is_power_steering", "is_central_locking",
            "is_driver_seat_height_adjustable", "is_day_night_rear_view_mirror",
            "is_ecw", "is_speed_alert"
        ]
        
        existing_bool_cols = [col for col in boolean_cols if col in data.columns]
        for col in existing_bool_cols:
             data[col] = data[col].map({'Yes': True, 'No': False}).fillna(False)

        # 3. Float Conversion
        float_cols = ["length", "width", "height", "gross_weight", "airbags", "population_density"]
        for col in float_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # 4. Ordinal Encoding for NCAP Rating
        if "ncap_rating" in data.columns:
            data["ncap_rating"] = data["ncap_rating"].astype(str)
            enc = OrdinalEncoder(categories=[["0", "1", "2", "3", "4", "5"]])
            data["NCAP_Rating"] = enc.fit_transform(data[["ncap_rating"]])
            data.drop(columns=["ncap_rating"], inplace=True)

        # 5. One-Hot Encoding
        categorical_cols = [
            "transmission_type", "cylinder", "gear_box", "rear_brakes_type", 
            "steering_type", "fuel_type", "make", "segment", "model", 
            "engine_type", "max_torque", "max_power", "area_cluster", "displacement"
        ]
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # 6. Separate Target Variable
        target_col = "is_claim"
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' missing from dataset")
            
        X = data.drop(columns=[target_col])
        y = data[target_col].astype(int)

        return X, y
