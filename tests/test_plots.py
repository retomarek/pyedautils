import unittest
import pandas as pd
from pyedautils.plots import plot_daily_profiles_overview

class TestPlots(unittest.TestCase):

    def test_plot_daily_profiles_overview(self):
        url = "https://raw.githubusercontent.com/retomarek/pyedautils/main/pyedautils/data/ele_meter.csv"

        # Call the function under test
        df = pd.read_csv(url, engine="python", sep=None)
        df['value'] = df['value'].diff()
        df = df.dropna()
        
        fig = plot_daily_profiles_overview(df)

        # Assertions
        self.assertTrue(fig._data[111]["y"][23] == 42.75)
        self.assertTrue(fig._data[111]["x"][23] == 23)

if __name__ == '__main__':
    unittest.main()
