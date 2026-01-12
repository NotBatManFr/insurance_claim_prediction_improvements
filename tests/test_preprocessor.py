import unittest
import pandas as pd
import numpy as np
from src.preprocessor import InsurancePreprocessor

class TestInsurancePreprocessor(unittest.TestCase):
    def setUp(self):
        self.processor = InsurancePreprocessor()
        self.raw_data = pd.DataFrame({
            'policy_id': ['123', '456'],
            'is_parking_camera': ['Yes', 'No'],
            'is_claim': [1, 0],
            'length': ['4000', '4200'],
            'ncap_rating': [3, 5],
            'transmission_type': ['Manual', 'Automatic']
        })

    def test_process_logic(self):
        # Action
        X, y = self.processor.process(self.raw_data)

        # Assertions
        
        # 1. Check policy_id dropped
        self.assertNotIn('policy_id', X.columns)
        
        # 2. Check boolean conversion
        self.assertTrue(X['is_parking_camera'].iloc[0])  # Yes -> True
        self.assertFalse(X['is_parking_camera'].iloc[1]) # No -> False
        
        # 3. Check float conversion
        self.assertEqual(X['length'].dtype, float)
        
        # 4. Check One-Hot Encoding (transmission_type should generate columns)
        self.assertNotIn('transmission_type', X.columns)
        
        # 5. Check Target separation
        self.assertEqual(len(y), 2)
        self.assertTrue(isinstance(y, pd.Series))

    def test_missing_target_raises_error(self):
        bad_data = self.raw_data.drop(columns=['is_claim'])
        with self.assertRaises(ValueError):
            self.processor.process(bad_data)
            
    def test_process_handles_missing_ncap_rating_gracefully(self):
        raw_data = pd.DataFrame({
            'policy_id': ['a','b'],
            'is_parking_camera': ['Yes', 'No'],
            'is_claim': [1, 0],
            'length': [4000, 4200],
            'transmission_type': ['Manual', 'Automatic']
        })

        # Should not raise and NCAP_Rating should not be present in X
        X, y = self.processor.process(raw_data)
        self.assertNotIn('NCAP_Rating', X.columns)
        self.assertEqual(len(y), 2)

    def test_process_raises_when_target_missing(self):
        raw_data = pd.DataFrame({
            'policy_id': ['a','b'],
            'is_parking_camera': ['Yes', 'No'],
            'length': [4000, 4200]
        })

        with self.assertRaises(ValueError):
            self.processor.process(raw_data)
