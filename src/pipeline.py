import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from .config import AppConfig
from .interfaces import IDataLoader, IPreprocessor
from .models import ModelEvaluator

class PipelineOrchestrator:
    def __init__(self, config: AppConfig, loader: IDataLoader, preprocessor: IPreprocessor):
        self.config = config
        self.loader = loader
        self.preprocessor = preprocessor
        self.models = {}

    def add_model(self, name: str, model_instance):
        self.models[name] = model_instance

    def run(self):
        print("1. Loading Data...")
        raw_df = self.loader.load()
        
        print("2. Preprocessing...")
        X, y = self.preprocessor.process(raw_df)

        print("3. Splitting Data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state, 
            stratify=y
        )

        print("4. Handling Imbalance (Training Set Only)...")
        oversampler = RandomOverSampler(sampling_strategy='minority', random_state=self.config.random_state)
        X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)

        print("5. Scaling Features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)

        results = []
        for name, model in self.models.items():
            print(f"6. Training & Evaluating: {name}...")
            model.train(X_train_scaled, y_train_res)
            y_pred = model.predict(X_test_scaled)
            metrics = ModelEvaluator.evaluate(y_test, y_pred, name)
            results.append(metrics)

        return pd.DataFrame(results).set_index("Model")
