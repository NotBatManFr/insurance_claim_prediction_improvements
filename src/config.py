from dataclasses import dataclass

@dataclass
class AppConfig:
    data_path: str = "data/train_data.csv"
    target_column: str = "is_claim"
    test_size: float = 0.2
    random_state: int = 11
