# Insurance Claim Prediction Pipeline

## 📌 Project Overview
This project implements a machine learning pipeline to predict whether an insurance policyholder will file a claim. It compares the performance of three classification algorithms: **Logistic Regression**, **Decision Tree**, and **Random Forest**.

The codebase has been refactored from a monolithic script into a modular architecture following **SOLID principles** and industry best practices. This ensures scalability, maintainability, and accurate model evaluation by preventing data leakage.

## 🚀 Key Features
- **Modular Architecture**: Separate components for data loading, preprocessing, and modeling.
- **Leakage Prevention**: Strict separation of training and testing data before oversampling or scaling.
- **Imbalance Handling**: Uses `RandomOverSampler` (SMOTE-like approach) strictly on the training set to address class imbalance.
- **Automated Setup**: Includes shell scripts for environment setup and execution.
- **Extensible**: Easily add new models (e.g., XGBoost, SVM) without modifying core logic.

## 📂 Project Structure
```text
insurance_prediction/
├── data/                  # Place your 'train_data.csv' here
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Centralized configuration (paths, params)
│   ├── interfaces.py      # Abstract Base Classes (Contracts)
│   ├── data_loader.py     # Data ingestion logic
│   ├── preprocessor.py    # Cleaning, encoding, and feature engineering
│   ├── models.py          # Model adapters and evaluation metrics
│   ├── pipeline.py        # Orchestrator handling the workflow
│   └── main.py            # Entry point
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── setup.sh               # One-click installation script
└── run.sh                 # Execution script



- **Repo:**: `insurance_claim_prediction` — a small pipeline to preprocess insurance data, train several sklearn models, and report performance.

**Requirements**
- **Python:**: 3.8+ recommended.
- **Dependencies:**: listed in `requirements.txt`.

**Quick Setup**
- **Create venv:**: Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

- **Install deps:**: Install requirements.

```bash
pip install -r requirements.txt
```

**Data**
- **Default dataset path:**: `data/train_data.csv` (configured in `src/config.py` as `AppConfig.data_path`).
- **Quick small dataset:**: `data/truncated_train_data.csv` is available for faster experimentation.
- **Important column:**: The pipeline expects a target column named `is_claim`.

If your dataset is in a different location, either update `data_path` in `src/config.py` or replace the file at `data/train_data.csv`.

**Run Training & Evaluation**
- The main entrypoint runs preprocessing, trains three models, evaluates them, and prints a summary.

Recommended (module) invocation:
```bash
python -m src.main
```

Or run the script directly:
```bash
python src/main.py
```

You should see console output steps (loading, preprocessing, training) and a final DataFrame with metrics for each model.

**Quick Tricks**
- **Use the truncated dataset**: edit `src/config.py` and set `data_path = "data/truncated_train_data.csv"` then re-run.
- **Change test split or random seed**: edit `test_size` and `random_state` in `src/config.py` (these are fields on `AppConfig`).
- **Add/remove models**: open `src/main.py` and modify the `pipeline.add_model(...)` lines. Models are wrapped with `SklearnModelAdapter`.

**Project Structure (key files)**
- `src/main.py`: orchestrates configuration, pipeline and model registration.
- `src/pipeline.py`: data split, imbalance handling, scaling, training loop and evaluation.
- `src/preprocessor.py`: dataset cleaning, encoding and returns `X, y`.
- `src/models.py`: model adapter and metric evaluation helpers.
- `data/`: put your `train_data.csv` (and optional `truncated_train_data.csv`) here.

**Troubleshooting**
- If you see `FileNotFoundError: Dataset not found at: ...`, ensure the file exists at the configured `data_path`.
- If you see `Target column 'is_claim' missing`, verify your CSV contains the `is_claim` column (case-sensitive).
- For fast iteration, use `data/truncated_train_data.csv`.

**Next steps & Suggestions**
- Persist trained models to disk (e.g., `joblib`) in `src/models.py` or in a new trainer script.
- Add CLI flags or environment-variable overrides for `data_path` and hyperparameters.

If you'd like, I can also:
- add a CLI wrapper to pass `--data` and `--output` options,
- commit this `README.md`, or
- add a small example `Makefile` or `run.sh` snippet to standardize commands.
