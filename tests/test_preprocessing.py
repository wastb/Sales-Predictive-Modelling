import pandas as pd
import numpy as np
import unittest
import sys
sys.path.append('../scripts')

from io import StringIO
from preprocessing import change_datatypes, replace_missing_values 

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        """Set up a test DataFrame with missing values and incorrect data types."""
        self.df = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "CompetitionOpenSinceYear": [2010, None],  # Include NaN for testing
            "Promo2SinceYear": [2012, None],
            "CompetitionOpenSinceMonth": [3, None],  
            "Promo2": [1, 0],  # Determines replacement logic
            "Promo2SinceWeek": [5, None],
            "PromoInterval": [None, None],
            "CompetitionDistance": [1000, None],
            "Open": [1, None]  # Missing value to be replaced by mode
        })

    def test_change_datatypes(self):
        """Test if data types are correctly changed."""
        change_datatypes(self.df)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.df['Date']))
        self.assertTrue(pd.api.types.is_integer_dtype(self.df['CompetitionOpenSinceYear']))
        self.assertTrue(pd.api.types.is_integer_dtype(self.df['Promo2SinceYear']))
        self.assertTrue(pd.api.types.is_integer_dtype(self.df['CompetitionOpenSinceMonth']))

    def test_replace_missing_values(self):
        """Test if missing values are correctly handled."""
        replace_missing_values(self.df)

        # Check if missing values are replaced as expected
        self.assertEqual(self.df['Promo2SinceWeek'].iloc[1], 0)  # Should be 0 due to Promo2==0
        self.assertEqual(self.df['Promo2SinceYear'].iloc[1], 1900)  # Should be 1900 due to Promo2==0
        self.assertEqual(self.df['PromoInterval'].iloc[1], 'UNKNOWN')  # Should be 'UNKNOWN' due to Promo2==0
        self.assertEqual(self.df['CompetitionDistance'].iloc[1], 0)  # Should be 0 as it was NaN
        self.assertEqual(self.df['CompetitionOpenSinceMonth'].iloc[1], 0)  # Should be 0 as it was NaN
        self.assertEqual(self.df['CompetitionOpenSinceYear'].iloc[1], 1900)  # Should be 1900 as it was NaN
        self.assertIsNotNone(self.df['Open'].iloc[1])  # Should be filled with mode

if __name__ == '__main__':
    unittest.main()



