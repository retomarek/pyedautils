import os
import unittest
from pyedautils.data_io import load_data
from pyedautils.plots import plot_daily_profiles_overview

class TestPlots(unittest.TestCase):

    def test_plot_daily_profiles_overview(self):
        # Get the path to the current module
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print("current_dir path:", current_dir)  # Debug print
        # Construct the path to the file relative to the package folder
        #file_path = "../pyedautils/data/ele_meter.csv"
        file_path = os.path.join(current_dir,'data', 'ele_meter.csv')
        print("File path:", file_path)  # Debug print

        # Call the function under test
        df = load_data(file_path)
        df['value'] = df['value'].diff()
        df = df.dropna()
        
        fig = plot_daily_profiles_overview(df)

        # Assertions
        self.assertTrue(fig._data[111]["y"][23] == 42.75)
        self.assertTrue(fig._data[111]["x"][23] == 23)

if __name__ == '__main__':
    unittest.main()
