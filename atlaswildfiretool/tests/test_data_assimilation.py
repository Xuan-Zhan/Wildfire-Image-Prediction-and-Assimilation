import unittest
import numpy as np
import matplotlib.pyplot as plt
from atlaswildfiretool.assimilation.data_assimilation import DataAssimilation


class TestMSEMethod(unittest.TestCase):
    def setUp(self):
        # This method will be run before each test
        class DummyClass:
            def mse(self, y_obs, y_pred):
                return np.mean((y_obs - y_pred) ** 2)

        self.dummy = DummyClass()

    def test_mse(self):
        # Define observed and predicted values
        y_obs = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])

        # Calculate expected MSE manually
        expected_mse = np.mean((y_obs - y_pred) ** 2)

        # Call the method and get the result
        result = self.dummy.mse(y_obs, y_pred)

        # Check if the result matches the expected MSE
        self.assertAlmostEqual(result, expected_mse, places=7)
