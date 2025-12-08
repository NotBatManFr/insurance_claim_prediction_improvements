import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.config import AppConfig
from src.pipeline import PipelineOrchestrator

class TestPipeline(unittest.TestCase):
    @patch('src.pipeline.RandomOverSampler')
    @patch('src.pipeline.StandardScaler')
    @patch('src.pipeline.train_test_split')
    def test_run_pipeline(self, mock_split, mock_scaler, mock_ros):
        # Setup Mocks
        config = AppConfig(test_size=0.2, random_state=42)
        mock_loader = MagicMock()
        mock_preprocessor = MagicMock()
        mock_model = MagicMock()
        
        # Create dummy data
        mock_loader.load.return_value = pd.DataFrame({'a': [1, 2]})
        mock_preprocessor.process.return_value = (pd.DataFrame({'a': [1, 2]}), pd.Series([0, 1]))
        
        # Mock split return values (X_train, X_test, y_train, y_test)
        mock_split.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        
        # Mock ROS and Scaler returns
        mock_ros_instance = mock_ros.return_value
        mock_ros_instance.fit_resample.return_value = (MagicMock(), MagicMock())
        
        mock_scaler_instance = mock_scaler.return_value
        mock_scaler_instance.fit_transform.return_value = MagicMock()
        mock_scaler_instance.transform.return_value = MagicMock()

        # Initialize Pipeline
        pipeline = PipelineOrchestrator(config, mock_loader, mock_preprocessor)
        pipeline.add_model("TestModel", mock_model)
        
        # Run
        results = pipeline.run()
        
        # Assertions
        # 1. Verify Loader called
        mock_loader.load.assert_called_once()
        
        # 2. Verify Preprocessor called
        mock_preprocessor.process.assert_called_once()
        
        # 3. Verify Model trained and predicted
        mock_model.train.assert_called_once()
        mock_model.predict.assert_called_once()
        
        # 4. Verify output is DataFrame
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("TestModel", results.index)

