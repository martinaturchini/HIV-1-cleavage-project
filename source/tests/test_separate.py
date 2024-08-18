import unittest
import numpy as np
from separate import separate

class TestSeparateFunction(unittest.TestCase):
    
    def test_mixed_values(self):
        Y_pred = np.array([0.3, 0.7, 0.4, 0.6])
        expected = np.array([0, 1, 0, 1])
        result = separate(Y_pred)
        np.testing.assert_array_equal(result, expected)

    def test_all_below_threshold(self):
        Y_pred = np.array([0.2, 0.1, 0.4, 0.3])
        expected = np.array([0, 0, 0, 0])
        result = separate(Y_pred)
        np.testing.assert_array_equal(result, expected)

    def test_all_above_threshold(self):
        Y_pred = np.array([0.6, 0.8, 0.9, 0.7])
        expected = np.array([1, 1, 1, 1])
        result = separate(Y_pred)
        np.testing.assert_array_equal(result, expected)

    def test_edge_case_at_threshold(self):
        Y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        expected = np.array([0, 0, 0, 0])
        result = separate(Y_pred)
        np.testing.assert_array_equal(result, expected)

    def test_empty_input(self):
        Y_pred = np.array([])
        expected = np.array([])
        result = separate(Y_pred)
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()

