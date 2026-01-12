import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from .config import AppConfig
from .interfaces import IDataLoader, IPreprocessor, IVisualizer
from .models import ModelEvaluator
from typing import Optional

class PipelineOrchestrator:
    def __init__(self, config: AppConfig, loader: IDataLoader, 
                 preprocessor: IPreprocessor, visualizer: Optional[IVisualizer] = None):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config: Application configuration
            loader: Data loader instance
            preprocessor: Preprocessor instance
            visualizer: Optional visualizer instance for generating plots
        """
        self.config = config
        self.loader = loader
        self.preprocessor = preprocessor
        self.visualizer = visualizer
        self.models = {}
        
        # Store data for visualization
        self._X_test = None
        self._y_test = None
        self._y_train_before = None
        self._y_train_after = None
        self._feature_names = None

    def add_model(self, name: str, model_instance):
        self.models[name] = model_instance

    def run(self, enable_visualization: bool = True):
        """
        Execute the full ML pipeline with optional visualization.
        
        Args:
            enable_visualization: Whether to generate visualization plots
            
        Returns:
            DataFrame with model performance metrics
        """
        print("1. Loading Data...")
        raw_df = self.loader.load()
        
        print("2. Preprocessing...")
        X, y = self.preprocessor.process(raw_df)
        self._feature_names = X.columns.tolist()

        print("3. Splitting Data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state, 
            stratify=y
        )
        
        # Store for visualization
        self._y_train_before = y_train.copy()
        self._X_test = X_test
        self._y_test = y_test

        print("4. Handling Imbalance (Training Set Only)...")
        oversampler = RandomOverSampler(sampling_strategy='minority', 
                                       random_state=self.config.random_state)
        X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)
        
        # Store for visualization
        self._y_train_after = y_train_res.copy()

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

        results_df = pd.DataFrame(results).set_index("Model")
        
        # Generate visualizations if enabled and visualizer provided
        if enable_visualization and self.visualizer is not None:
            self.visualizer.generate_all_plots(
                results=results_df,
                models_dict=self.models,
                X_test=X_test_scaled,
                y_test=y_test,
                y_train_before=self._y_train_before,
                y_train_after=self._y_train_after,
                feature_names=self._feature_names
            )
        elif enable_visualization and self.visualizer is None:
            print("\n⚠ Visualization requested but no visualizer provided. Skipping plots.")
        
        return results_df