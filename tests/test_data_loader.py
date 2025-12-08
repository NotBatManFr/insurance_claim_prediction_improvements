import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data_loader import CSVLoader

class TestCSVLoader(unittest.TestCase):
    @patch('pandas.read_csv')
    def test_load_success(self, mock_read_csv):
        # Setup
        mock_df = pd.DataFrame({'col1': [1, 2]})
        mock_read_csv.return_value = mock_df
        loader = CSVLoader('dummy_path.csv')

        # Action
        df = loader.load()

        # Assert
        pd.testing.assert_frame_equal(df, mock_df)
        mock_read_csv.assert_called_once_with('dummy_path.csv')

    @patch('pandas.read_csv')
    def test_load_file_not_found(self, mock_read_csv):
        # Setup
        mock_read_csv.side_effect = FileNotFoundError
        loader = CSVLoader('invalid_path.csv')

        # Assert
        with self.assertRaises(FileNotFoundError):
            loader.load()
