import unittest
import os
import sys
import io
import shutil
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
        shutil.rmtree(self.test_dir)

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
        
    def test_save_and_load_new_directory(self):
        # Test data: time series data with datetime index
        dates = pd.date_range(start="2022-01-01", periods=10, freq="D")
        values = np.round(np.random.randn(10), 3)
        df = pd.DataFrame({"Value": values}, index=dates)
        file_path = os.path.join(self.test_dir, "newdir/test.csv")

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
        loaded_data = load_data(file_path)

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
        loaded_data = load_data(file_path)

        self.assertTrue(isinstance(loaded_data, pd.DataFrame))
        self.assertTrue(df.equals(loaded_data))
    
    def test_save_unsupported_format(self):
        # Test data: time series data with datetime index
        dates = pd.date_range(start="2022-01-01", periods=10, freq="D")
        values = np.random.randn(10)
        df = pd.DataFrame({"Value": values}, index=dates)
        file_path = os.path.join(self.test_dir, "test.abc")
        
        # Redirect stdout to a StringIO object to capture the output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        save_data(df, file_path)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Get the captured output
        console_output = captured_output.getvalue().strip()
        
        self.assertEqual(console_output, "Saving data to: test_data\\test.abc\nError: Unsupported file format")
        
    def test_load_unsupported_format(self):
        # Test data: time series data with datetime index
        file_path = os.path.join(self.test_dir, "test.def")
        
        # Redirect stdout to a StringIO object to capture the output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        df = load_data(file_path)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Get the captured output
        console_output = captured_output.getvalue().strip()
        
        self.assertEqual(console_output, "Loading data from: test_data\\test.def\nError: Unsupported file format")
        
        
if __name__ == "__main__":
    unittest.main() # pragma: no cover
