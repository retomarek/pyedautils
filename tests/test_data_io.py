import unittest
import os
import pandas as pd
import numpy as np
import gzip
import pickle
from pyedautils.data_io import save_data, load_data

class TestDataIO(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = "test_data"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        # Remove the temporary directory and its contents after tests
        for file_name in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file_name)
            os.remove(file_path)
        os.rmdir(self.test_dir)

    def test_save_and_load_json(self):
        # Test data
        data = {"key1": "value1", "key2": "value2"}
        file_path = os.path.join(self.test_dir, "test.json")

        # Test saving and loading JSON file
        save_data(data, file_path)
        loaded_data = load_data(file_path)

        self.assertEqual(data, loaded_data)

    def test_save_and_load_csv(self):
        # Test data: time series data with datetime index
        dates = pd.date_range(start="2022-01-01", periods=10, freq="D")
        values = np.round(np.random.randn(10), 3)
        df = pd.DataFrame({"Value": values}, index=dates)
        file_path = os.path.join(self.test_dir, "test.csv")

        # Test saving and loading CSV file
        save_data(df, file_path, index=True)
        loaded_data = load_data(file_path)
        loaded_data.index = pd.to_datetime(loaded_data.iloc[:, 0])
        loaded_data.drop(loaded_data.columns[0], axis=1, inplace=True)

        self.assertTrue(isinstance(loaded_data, pd.DataFrame))
        self.assertTrue(df.equals(loaded_data))

    def test_save_and_load_pickle(self):
        # Test data: time series data with datetime index
        dates = pd.date_range(start="2022-01-01", periods=10, freq="D")
        values = np.random.randn(10)
        df = pd.DataFrame({"Value": values}, index=dates)
        file_path = os.path.join(self.test_dir, "test.pkl")

        # Test saving and loading pickle file
        save_data(df, file_path)
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        self.assertTrue(isinstance(loaded_data, pd.DataFrame))
        self.assertTrue(df.equals(loaded_data))

    def test_save_and_load_compressed_pickle(self):
        # Test data: time series data with datetime index
        dates = pd.date_range(start="2022-01-01", periods=10, freq="D")
        values = np.random.randn(10)
        df = pd.DataFrame({"Value": values}, index=dates)
        file_path = os.path.join(self.test_dir, "test.pklz")

        # Test saving and loading compressed pickle file
        save_data(df, file_path)
        with gzip.open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        self.assertTrue(isinstance(loaded_data, pd.DataFrame))
        self.assertTrue(df.equals(loaded_data))

if __name__ == "__main__":
    unittest.main() # pragma: no cover
