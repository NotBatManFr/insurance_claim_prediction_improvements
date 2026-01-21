from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class FeatureConfig:
    """Defines the expected structure of the input data."""
    # Columns to drop (IDs, noise)
    drop_cols: List[str] = field(default_factory=lambda: ["policy_id"])
    
    # Boolean columns (Yes/No -> True/False)
    bool_cols: List[str] = field(default_factory=lambda: [
        "is_parking_camera", "is_tpms", "is_adjustable_steering", "is_esc",
        "is_parking_sensors", "is_front_fog_lights", "is_rear_window_wiper", 
        "is_rear_window_washer", "is_rear_window_defogger", "is_brake_assist", 
        "is_power_door_locks", "is_power_steering", "is_central_locking", 
        "is_driver_seat_height_adjustable", "is_day_night_rear_view_mirror", 
        "is_ecw", "is_speed_alert"
    ])

    # Numeric columns (Forced to float)
    num_cols: List[str] = field(default_factory=lambda: [
        "length", "width", "height", "gross_weight", "airbags", "population_density"
    ])

    # Categorical columns (One-Hot Encoding)
    categorical_cols: List[str] = field(default_factory=lambda: [
        "transmission_type", "cylinder", "gear_box", "rear_brakes_type", 
        "steering_type", "fuel_type", "make", "segment", "model", 
        "engine_type", "max_torque", "max_power", "area_cluster", "displacement"
    ])

    # Ordinal columns (Mapped order)
    ordinal_cols: Dict[str, List[str]] = field(default_factory=lambda: {
        "ncap_rating": ["0", "1", "2", "3", "4", "5"]
    })

@dataclass
class AppConfig:
    data_path: str = "data/train_data.csv"
    target_column: str = "is_claim"
    test_size: float = 0.2
    random_state: int = 11
    schema: FeatureConfig = field(default_factory=FeatureConfig)
