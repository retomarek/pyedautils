"""Warning bands (± offsets) and the optional abs-humidity limit of the h,x plots."""

import inspect
import json
import re
import unittest

from pyedautils import comfort
from pyedautils.plots import plot_mollier_hx, plot_mollier_hx_fast


def _js_range(html, name):
    """Value of ``let <name> = [...]`` in the plot_mollier_hx HTML."""
    m = re.search(r"let " + name + r" = (\[[^\]]*\]);", html)
    assert m, name
    return json.loads(m.group(1))


def _bands(html, which):
    return {k: _js_range(html, which + k) for k in ("T", "Phi", "X")}


def _fast_calls(html):
    """(comfort, orange, red) passed to setComfort / setBands."""
    cz = json.loads(re.search(r"c\.setComfort\((.*?)\);", html).group(1))
    o, r = json.loads("[" + re.search(r"c\.setBands\((.*?)\);", html).group(1) + "]")
    return cz, o, r


class TestComfortWarningBands(unittest.TestCase):

    def test_defaults_match_compass_stage_limits(self):
        sig = inspect.signature(comfort.comfort_compass_distribution).parameters
        bands = comfort.comfort_warning_bands()
        self.assertEqual(
            (bands["orange"]["temperature"], bands["red"]["temperature"]),
            sig["temp_stage_limits"].default)
        self.assertEqual(
            (bands["orange"]["rel_humidity"] * 100, bands["red"]["rel_humidity"] * 100),
            sig["hum_stage_limits"].default)
        self.assertEqual(
            (bands["orange"]["abs_humidity"], bands["red"]["abs_humidity"]),
            sig["hum_abs_stage_limits"].default)

    def test_same_signature_defaults(self):
        sig_d = inspect.signature(comfort.comfort_compass_distribution).parameters
        sig_b = inspect.signature(comfort.comfort_warning_bands).parameters
        for name in sig_b:
            self.assertEqual(sig_b[name].default, sig_d[name].default, name)

    def test_red_encloses_orange(self):
        b = comfort.comfort_warning_bands(temp_stage_limits=(3, 2),
                                          hum_stage_limits=(5, 10))
        self.assertEqual(b["red"]["temperature"], 3.0)
        self.assertAlmostEqual(b["red"]["rel_humidity"], 0.10)


class TestMollierBandOffsets(unittest.TestCase):

    CZ = {"temperature": (20, 26), "rel_humidity": (0.30, 0.65),
          "abs_humidity": (0, 0.0115)}

    def test_bands_off_by_default(self):
        html = plot_mollier_hx()
        self.assertIn("hasOrange = false", html)
        self.assertIn("hasRed = false", html)

    def test_absolute_ranges_unchanged(self):
        html = plot_mollier_hx(comfort_zone_orange={
            "temperature": (19, 27), "rel_humidity": (0.25, 0.70)})
        self.assertEqual(_bands(html, "orange"),
                         {"T": [19, 27], "Phi": [0.25, 0.70], "X": [0, 0.030]})

    def test_offsets_around_comfort_zone(self):
        html = plot_mollier_hx(
            comfort_zone=self.CZ,
            comfort_zone_orange={"temperature": 1, "rel_humidity": 0.05,
                                 "abs_humidity": 0.001},
            comfort_zone_red={"temperature": 2.5, "rel_humidity": 0.10,
                              "abs_humidity": 0.0025})
        self.assertIn("hasOrange = true", html)
        self.assertEqual(_bands(html, "orange"),
                         {"T": [19, 27], "Phi": [0.25, 0.70], "X": [0, 0.0125]})
        self.assertEqual(_bands(html, "red"),
                         {"T": [17.5, 28.5], "Phi": [0.20, 0.75], "X": [0, 0.014]})

    def test_true_uses_compass_defaults(self):
        explicit = comfort.comfort_warning_bands()
        a = plot_mollier_hx(comfort_zone=self.CZ, comfort_zone_orange=True,
                            comfort_zone_red=True)
        b = plot_mollier_hx(comfort_zone=self.CZ,
                            comfort_zone_orange=explicit["orange"],
                            comfort_zone_red=explicit["red"])
        for w in ("orange", "red"):
            self.assertEqual(_bands(a, w), _bands(b, w))

    def test_partial_offsets_fill_defaults(self):
        html = plot_mollier_hx(comfort_zone=self.CZ,
                               comfort_zone_orange={"temperature": 2})
        self.assertEqual(_bands(html, "orange"),
                         {"T": [18, 28], "Phi": [0.25, 0.70], "X": [0, 0.0125]})

    def test_mixed_offset_and_absolute(self):
        html = plot_mollier_hx(comfort_zone=self.CZ, comfort_zone_orange={
            "temperature": 1, "rel_humidity": (0.2, 0.8), "abs_humidity": None})
        self.assertEqual(_bands(html, "orange"),
                         {"T": [19, 27], "Phi": [0.2, 0.8], "X": [0, 1.0]})

    def test_clamped(self):
        html = plot_mollier_hx(
            comfort_zone={"temperature": (20, 26), "rel_humidity": (0.05, 0.98),
                          "abs_humidity": (0.0005, 0.0115)},
            comfort_zone_orange={"rel_humidity": 0.1, "abs_humidity": 0.002})
        b = _bands(html, "orange")
        self.assertEqual(b["Phi"], [0.0, 1.0])
        self.assertEqual(b["X"][0], 0.0)

    def test_red_widened_to_enclose_orange(self):
        html = plot_mollier_hx(comfort_zone=self.CZ,
                               comfort_zone_orange={"temperature": 3},
                               comfort_zone_red={"temperature": 1})
        self.assertEqual(_bands(html, "red")["T"], [17, 29])

    def test_absolute_red_not_widened(self):
        html = plot_mollier_hx(comfort_zone=self.CZ,
                               comfort_zone_orange={"temperature": 3},
                               comfort_zone_red={"temperature": (19, 27)})
        self.assertEqual(_bands(html, "red")["T"], [19, 27])

    def test_label_and_colour_with_offsets(self):
        html = plot_mollier_hx(comfort_zone_orange={
            "temperature": 1, "label": "Stage 1", "color": "#123456"})
        self.assertIn('"Stage 1"', html)
        self.assertIn('"#123456"', html)

    def test_offsets_without_comfort_zone_raise(self):
        with self.assertRaises(ValueError):
            plot_mollier_hx(comfort_zone=False,
                            comfort_zone_orange={"temperature": 1})
        # absolute ranges still work without a comfort zone
        html = plot_mollier_hx(comfort_zone=False, comfort_zone_orange={
            "temperature": (19, 27), "rel_humidity": (0.25, 0.70)})
        self.assertIn("hasOrange = true", html)

    def test_fast_offsets(self):
        html = plot_mollier_hx_fast(
            comfort_zone=self.CZ,
            comfort_zone_orange={"temperature": 1, "rel_humidity": 0.05,
                                 "abs_humidity": 0.001},
            comfort_zone_red=True)
        _, o, r = _fast_calls(html)
        self.assertEqual((o["t"], o["phi"], o["x"]),
                         ([19, 27], [0.25, 0.70], [0, 0.0125]))
        self.assertEqual(o["color"], "#E67E22")
        self.assertEqual((r["t"], r["phi"], r["x"]),
                         ([17.5, 28.5], [0.20, 0.75], [0, 0.014]))
        self.assertEqual(r["color"], "#C0392B")

    def test_fast_absolute_unchanged(self):
        html = plot_mollier_hx_fast(comfort_zone_orange={
            "temperature": (19, 27), "rel_humidity": (0.25, 0.70)})
        cz, o, r = _fast_calls(html)
        self.assertIsNone(cz)
        self.assertIsNone(r)
        self.assertEqual((o["t"], o["phi"], o["x"]),
                         ([19, 27], [0.25, 0.70], [0, 0.030]))


class TestMollierNoAbsHumidityLimit(unittest.TestCase):

    def test_default_cap_kept(self):
        html = plot_mollier_hx()
        self.assertEqual(_js_range(html, "rangeX"), [0, 0.0115])
        html = plot_mollier_hx(comfort_zone={"temperature": (18, 24)})
        self.assertEqual(_js_range(html, "rangeX"), [0, 0.0115])

    def test_none_means_no_upper_limit(self):
        html = plot_mollier_hx(comfort_zone={
            "temperature": (20, 26), "rel_humidity": (0.30, 0.65),
            "abs_humidity": None})
        self.assertEqual(_js_range(html, "rangeX"), [0, 1.0])
        self.assertNotIn("null", re.search(r"let rangeX = .*;", html).group(0))

    def test_none_upper_bound(self):
        html = plot_mollier_hx(comfort_zone={"abs_humidity": (0.002, None)})
        self.assertEqual(_js_range(html, "rangeX"), [0.002, 1.0])

    def test_offset_band_without_abs_limit_stays_open(self):
        html = plot_mollier_hx(comfort_zone={"abs_humidity": None},
                               comfort_zone_orange=True, comfort_zone_red=True)
        self.assertEqual(_bands(html, "orange")["X"], [0, 1.0])
        self.assertEqual(_bands(html, "red")["X"], [0, 1.0])

    def test_band_abs_none(self):
        html = plot_mollier_hx(comfort_zone_orange={
            "temperature": (19, 27), "abs_humidity": None})
        self.assertEqual(_bands(html, "orange")["X"], [0, 1.0])

    def test_fast_none(self):
        html = plot_mollier_hx_fast(comfort_zone={"abs_humidity": None},
                                    comfort_zone_orange=True)
        cz, o, _ = _fast_calls(html)
        self.assertEqual(cz["x"], [0, 1.0])
        self.assertEqual(o["x"], [0, 1.0])


if __name__ == "__main__":
    unittest.main()
