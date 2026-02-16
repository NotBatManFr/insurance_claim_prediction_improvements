import unittest
import math
from unittest.mock import MagicMock
import numpy as np
from src.models import SklearnModelAdapter, ModelEvaluator

class TestSklearnModelAdapter(unittest.TestCase):
    def test_train_and_predict(self):
        # Mocking underlying sklearn model
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
        # 1 TP, 2 TN, 1 FN, 0 FP
        self.y_true = [1, 0, 1, 0]
        self.y_pred = [1, 0, 0, 0] 

        metrics = ModelEvaluator.evaluate(self.y_true, self.y_pred, "TestModel")
        
        self.assertEqual(metrics['Model'], "TestModel")
        self.assertEqual(metrics['Accuracy'], 0.75)
        self.assertEqual(metrics['Precision'], 1.0)
        self.assertEqual(metrics['Recall'], 0.5)
        self.assertAlmostEqual(metrics['F1_Score'], 0.666666, places=5)
        self.assertEqual(metrics['FNR'], 0.5)
    
    def test_evaluate_with_y_score(self):
        self.y_true = [0, 1, 0, 1]
        self.y_pred = [0, 1, 0, 1] 
        self.y_score = [0.1, 0.9, 0.2, 0.8]

        metrics = ModelEvaluator.evaluate(self.y_true, self.y_pred, "ROCModel", self.y_score)
        self.assertEqual(metrics['ROC_AUC'], 1.0)

    def test_evaluate_exception_handling(self):
        metrics = ModelEvaluator.evaluate(None, None, "FailModel")
        self.assertEqual(metrics['Model'], "FailModel")
        self.assertTrue(math.isnan(metrics['Accuracy']))
        self.assertTrue(math.isnan(metrics['ROC_AUC']))