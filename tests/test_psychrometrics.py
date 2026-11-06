"""Public psychrometrics module (re-exports of pyedautils._mollier)."""

import unittest

import pyedautils._mollier as _mollier
import pyedautils.psychrometrics as psychrometrics
from pyedautils.psychrometrics import pressure_from_altitude, state


class TestPublicPsychrometrics(unittest.TestCase):
    def test_all_names_are_the_mollier_objects(self):
        for name in psychrometrics.__all__:
            self.assertIs(getattr(psychrometrics, name), getattr(_mollier, name), name)

    def test_no_private_names_exported(self):
        self.assertFalse([n for n in psychrometrics.__all__ if n.startswith("_")])

    def test_pressure_from_altitude(self):
        self.assertAlmostEqual(pressure_from_altitude(0), 101325.0)
        self.assertAlmostEqual(pressure_from_altitude(450), 96035, delta=1)
        self.assertLess(pressure_from_altitude(1500), pressure_from_altitude(450))

    def test_state_altitude_matches(self):
        self.assertAlmostEqual(state(t=21, phi=0.5, altitude=450).p,
                               pressure_from_altitude(450))


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
