import unittest
from unittest.mock import MagicMock
import numpy as np
from src.models import SklearnModelAdapter, ModelEvaluator

class TestSklearnModelAdapter(unittest.TestCase):
    def test_train_and_predict(self):
        # Mock the underlying sklearn model
        mock_sklearn = MagicMock()
        mock_sklearn.predict.return_value = np.array([1, 0])
        
        adapter = SklearnModelAdapter(mock_sklearn)
        X_train = np.array([[1], [2]])
        y_train = np.array([1, 0])
        
        # Test Train
        adapter.train(X_train, y_train)
        mock_sklearn.fit.assert_called_once_with(X_train, y_train)
        
        # Test Predict
        res = adapter.predict(X_train)
        mock_sklearn.predict.assert_called_once_with(X_train)
        np.testing.assert_array_equal(res, np.array([1, 0]))

class TestModelEvaluator(unittest.TestCase):
    def test_evaluate_metrics(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 0] # 1 TP, 2 TN, 1 FN, 0 FP
        
        metrics = ModelEvaluator.evaluate(y_true, y_pred, "TestModel")
        
        self.assertEqual(metrics['Model'], "TestModel")
        self.assertEqual(metrics['Accuracy'], 0.75)
        # Precision: TP / (TP + FP) = 1 / 1 = 1.0
        self.assertEqual(metrics['Precision'], 1.0)
        # Recall: TP / (TP + FN) = 1 / 2 = 0.5
        self.assertEqual(metrics['Recall'], 0.5)
