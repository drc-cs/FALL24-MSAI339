from basic_statistics import (get_float64_column_names, get_missing_value_indices, drop_missing_values, 
                            calculate_covariance_numpy, calculate_pearson_correlation_scipy, calculate_pearson_correlation_numpy, calculate_spearman_correlation_scipy, 
                            perform_independent_t_test, check_variance_homogeneity, check_normality)
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch

class TestBasicStatistics(unittest.TestCase):
    def test_get_float64_column_names(self):
        df = pd.DataFrame({
            'float_col1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'float_col2': [1.0, 2.0, 3.0, 4.0, 5.0],
            'non_float_col': ['a', 'b', 'c', 'd', 'e']
        })
        float_columns = get_float64_column_names(df)
        self.assertListEqual(float_columns, ['float_col1', 'float_col2'])
    
    def test_get_missing_value_indices(self):
        df = pd.DataFrame({
            'float_col1': [1.0, np.nan, 3.0, 4.0, np.nan],
            'float_col2': [1.0, 2.0, 3.0, 4.0, np.nan],
            'non_float_col': ['a', 'b', 'c', 'd', np.nan]
        })
        missing_indices = get_missing_value_indices(df, 'float_col1')
        self.assertListEqual(missing_indices, [1, 4])
    
    def test_drop_missing_values(self):
        df = pd.DataFrame({
            'float_col1': [1.0, np.nan, 3.0, 4.0, np.nan],
            'float_col2': [1.0, 2.0, 3.0, 4.0, np.nan],
            'non_float_col': ['a', 'b', 'c', 'd', np.nan]
        })
        df = drop_missing_values(df)
        self.assertEqual(df.shape[0], 3)

    @patch('numpy.cov')
    def test_calculate_covariance_numpy_no_cov(self, patch):
        x = np.array([1, 2, 3, 3, 5])
        y = np.array([1, 2, 3, 4, 5])
        covariance = calculate_covariance_numpy(x, y)
        self.assertEqual(covariance, 1.8)
        assert not patch.called

    @patch('numpy.corrcoef')
    def test_calculate_pearson_correlation_numpy(self, patch):
        x = np.array([1, 2, 3, 3, 5])
        y = np.array([1, 2, 3, 4, 5])
        correlation = calculate_pearson_correlation_numpy(x, y)
        self.assertEqual(round(correlation, 4), 0.9594)
        assert not patch.called

    def test_calculate_pearson_correlation_scipy(self):
        x = np.array([1, 2, 3, 3, 5])
        y = np.array([1, 2, 3, 4, 5])
        correlation = calculate_pearson_correlation_scipy(x, y)
        self.assertEqual(round(correlation, 4), 0.9594)
    
    def test_calculate_spearman_correlation_scipy(self):
        x = np.array([1, 2, 3, 3, 5])
        y = np.array([1, 2, 3, 4, 5])
        correlation = calculate_spearman_correlation_scipy(x, y)
        self.assertEqual(round(correlation, 3), 0.975)
    
    def test_perform_independent_t_test(self):
        x = np.array([1, 2, 3, 3, 5])
        y = np.array([1, 2, 3, 4, 5])
        _, p_value = perform_independent_t_test(x, y)
        self.assertEqual(round(p_value, 3), 0.842)

    def test_check_variance_homogeneity(self):
        x = np.array([1, 2, 3, 3, 5])
        y = np.array([1, 2, 3, 4, 5])
        _, equal_variance = check_variance_homogeneity(x, y)
        self.assertAlmostEqual(equal_variance, 0.74, 2)

    def test_check_normality(self):
        x = np.array([1, 2, 3, 3, 5])
        _, normality = check_normality(x)
        self.assertAlmostEqual(normality, 0.78, 2)