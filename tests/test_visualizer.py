import unittest
import tempfile
import os
import numpy as np
import pandas as pd
from src.visualizer import MatplotlibVisualizer

class StubInner:
    # Intentionally no predict_proba or feature_importances_
    pass

class StubModel:
    def __init__(self):
        self.model = StubInner()
    def predict(self, X):
        # Return zeros with proper length
        if hasattr(X, 'shape'):
            return np.zeros(X.shape[0], dtype=int)
        return np.zeros(len(X), dtype=int)

class TestMatplotlibVisualizer(unittest.TestCase):
    def test_generate_all_plots_handles_models_without_proba_or_feature_importance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MatplotlibVisualizer(output_dir=tmpdir)

            # Minimal results dataframe
            results = pd.DataFrame({'Accuracy': [0.5]}, index=['stub']).T

            # One stub model without predict_proba or feature_importances_
            models = {'StubModel': StubModel()}

            # Small test data
            X_test = np.zeros((3, 1))
            y_test = pd.Series([0, 0, 0])
            y_train_before = pd.Series([0, 0])
            y_train_after = pd.Series([0, 0])
            feature_names = ['f1']

            paths = viz.generate_all_plots(
                results=results,
                models_dict=models,
                X_test=X_test,
                y_test=y_test,
                y_train_before=y_train_before,
                y_train_after=y_train_after,
                feature_names=feature_names
            )

            # Expected keys that must be present
            self.assertIn('performance', paths)
            self.assertIn('confusion_matrices', paths)
            self.assertIn('class_distribution', paths)

            # Should NOT include ROC curves (no predict_proba) or feature importance
            self.assertNotIn('roc_curves', paths)
            self.assertFalse(any(k.startswith('feature_importance_') for k in paths.keys()))

            # Ensure files are returned as strings and (attempted) written
            for k, p in paths.items():
                self.assertIsInstance(p, str)
                # file should be inside tmpdir
                self.assertTrue(os.path.commonpath([tmpdir, p]) == tmpdir)
