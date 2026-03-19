import unittest
import os
import numpy as np
import pandas as pd
from unittest.mock import patch

from pyedautils.energy_signature import compute_pes, PESResult


def _fast_get_season(date, **kwargs):
    """Fast mock for get_season that skips ephem calculations."""
    if isinstance(date, pd.Series):
        return date.apply(lambda x: _fast_get_season(x))
    month = date.month
    if month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    elif month in (9, 10, 11):
        return "Fall"
    else:
        return "Winter"


def _load_pes_data():
    """Load the bundled PES sample CSV."""
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', 'pyedautils', 'data',
        'bldg_engy_sig_pes.csv',
    )
    return pd.read_csv(csv_path)


@patch('pyedautils.energy_signature.get_season', side_effect=_fast_get_season)
class TestComputePES(unittest.TestCase):
    """Tests for the PES computation algorithm."""

    @classmethod
    def setUpClass(cls):
        cls.df = _load_pes_data()

    def test_returns_pes_result(self, _mock):
        result = compute_pes(self.df)
        self.assertIsInstance(result, PESResult)

    def test_result_fields(self, _mock):
        result = compute_pes(self.df)
        self.assertIsNotNone(result.tb)
        self.assertIsNotNone(result.q_tot)
        self.assertIsNotNone(result.p_dhwc)
        self.assertIsNotNone(result.p_dhw)
        self.assertIsNotNone(result.p_ihg)

    def test_tb_in_range(self, _mock):
        result = compute_pes(self.df)
        self.assertGreaterEqual(result.tb, 10.0)
        self.assertLessEqual(result.tb, 20.0)

    def test_q_tot_positive(self, _mock):
        result = compute_pes(self.df)
        self.assertGreater(result.q_tot, 0)

    def test_p_dhwc_positive(self, _mock):
        result = compute_pes(self.df)
        self.assertGreater(result.p_dhwc, 0)

    def test_p_dhw_positive(self, _mock):
        result = compute_pes(self.df)
        self.assertGreater(result.p_dhw, 0)

    def test_backward_compat_aliases(self, _mock):
        result = compute_pes(self.df)
        self.assertEqual(result.p_stby, result.p_dhwc)
        self.assertEqual(result.p_hw, result.p_dhw)

    def test_p_ihg_echoed(self, _mock):
        result = compute_pes(self.df, p_ihg=3.5)
        self.assertEqual(result.p_ihg, 3.5)

    def test_convergence_with_ihg(self, _mock):
        result = compute_pes(self.df, p_ihg=4.8)
        self.assertIsInstance(result, PESResult)
        self.assertGreater(result.q_tot, 0)


@patch('pyedautils.energy_signature.get_season', side_effect=_fast_get_season)
class TestComputePESEdgeCases(unittest.TestCase):
    """Edge case tests for compute_pes."""

    def test_insufficient_warm_weeks(self, _mock):
        """Data with only winter months should fail."""
        hours = pd.date_range('2020-01-01', '2020-03-31 23:00', freq='h')
        n = len(hours)
        np.random.seed(42)
        df = pd.DataFrame({
            'timestamp': hours,
            'outside_temp': np.random.normal(-5, 3, n),
            'power': np.random.uniform(5, 20, n),
            'room_temp': np.random.normal(21, 0.5, n),
        })
        with self.assertRaises(ValueError):
            compute_pes(df)

    def test_no_cold_days(self, _mock):
        """Data with only hot temps should fail to find cold days."""
        hours = pd.date_range('2020-01-01', '2020-12-31 23:00', freq='h')
        n = len(hours)
        np.random.seed(42)
        df = pd.DataFrame({
            'timestamp': hours,
            'outside_temp': np.random.uniform(25, 35, n),
            'power': np.random.uniform(2, 5, n),
            'room_temp': np.random.normal(21, 0.5, n),
        })
        with self.assertRaises(ValueError):
            compute_pes(df)


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
