"""Tests for the D3 v7 bundle, outside air / ventilation, edge arrows,
clipping and the user-definable captions / texts of the h,x diagrams."""

import json
import re
import unittest
from importlib import resources

import numpy as np
import pandas as pd

from pyedautils._mollier import P_STD, state
from pyedautils.comfort import VENTILATION_THRESHOLD, ventilation_effect
from pyedautils.plots import (
    MOLLIER_DEFAULT_CAPTIONS,
    MOLLIER_DEFAULT_TEXTS,
    plot_mollier_hx,
    plot_mollier_hx_fast,
)
from pyedautils.plots._constants import _SEASON_LABELS_DE

D3_DIR = resources.files("pyedautils") / "data" / "d3_mollierhx"
JS_FILES = ("mollierFunctions.js", "coordinateGenerator.js", "drawComfort.js",
            "mollierChart.js")
BOTH = (plot_mollier_hx, plot_mollier_hx_fast)


def _js_source(name):
    return (D3_DIR / name).read_text(encoding="utf-8")


def _strip_bundle(html):
    """The generated HTML without the bundled d3-mollierhx sources (they
    contain the English defaults as JS constants)."""
    for name in JS_FILES:
        html = html.replace(_js_source(name), "")
    return html


def _js_object(src, name):
    """Parse ``const NAME = { key: "value", ... };`` of a bundled JS file."""
    body = re.search(r"const " + name + r" = \{(.*?)\n\};", src, re.S).group(1)
    return dict(re.findall(r'(\w+):\s*"((?:[^"\\]|\\.)*)"', body))


def _data(n=48, start="2024-01-10"):
    idx = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({"timestamp": idx,
                         "humidity": np.linspace(40, 55, n),
                         "temperature": np.linspace(20, 22, n)})


def _let(html, name):
    """JSON value of ``let <name> = ...;`` in the plot_mollier_hx HTML."""
    m = re.search(r"\n  let " + name + r" = (.*?);\n", html)
    return json.loads(m.group(1))


def _chart_opts(html):
    """Options passed to ``new MollierChart`` in the fast HTML."""
    m = re.search(r"new MollierChart\('#mollier_\w+',(.*?)\);", html)
    return json.loads(m.group(1).replace("<\\/", "</"))


class TestBundleAndD3v7(unittest.TestCase):

    def test_d3_v7_from_cdn(self):
        for fn in BOTH:
            html = fn(data=_data(), outdoor=(4, 85))
            self.assertIn("https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js", html)
            for old in ("d3.v5", "d3js.org", "d3.event", "d3.mouse("):
                self.assertNotIn(old, html, f"{fn.__name__}: {old}")

    def test_inline_handlers_use_v7_signature(self):
        html = _strip_bundle(plot_mollier_hx(data=_data(), show_frequency=True,
                                             outdoor=(4, 85)))
        self.assertNotRegex(html, r'\.on\("mouse\w+", function\s*\(d\)')
        self.assertIn("function(event, d)", html)
        self.assertIn("function (event, d)", html)   # frequency-line hover

    def test_bundled_sources_are_inlined_unchanged(self):
        hx = plot_mollier_hx()
        for name in JS_FILES[:3]:
            self.assertIn(_js_source(name), hx)
        fast = plot_mollier_hx_fast()
        for name in JS_FILES:
            self.assertIn(_js_source(name), fast)

    def test_python_defaults_match_js_defaults(self):
        caps = _js_object(_js_source("coordinateGenerator.js"), "HX_DEFAULT_CAPTIONS")
        overlay = _js_object(_js_source("drawComfort.js"), "HX_DEFAULT_TEXTS")
        chart = _js_object(_js_source("mollierChart.js"), "MOLLIER_CHART_DEFAULT_TEXTS")
        self.assertEqual(set(caps), set(MOLLIER_DEFAULT_CAPTIONS))
        for key, value in caps.items():
            if key != "x_axis_title":   # pyedautils keeps its x_axis_title default
                self.assertEqual(MOLLIER_DEFAULT_CAPTIONS[key], value, key)
        for key, value in {**overlay, **chart}.items():
            self.assertEqual(MOLLIER_DEFAULT_TEXTS[key], value, key)


class TestOutdoorAndVentilation(unittest.TestCase):

    def test_no_outdoor_by_default(self):
        html = plot_mollier_hx(data=_data())
        self.assertIsNone(_let(html, "outdoorPoint"))
        self.assertNotIn("c.setOutdoor(", plot_mollier_hx_fast(data=_data()))

    def test_outdoor_point_and_defaults(self):
        html = plot_mollier_hx(data=_data(), outdoor=(4, 85))
        self.assertEqual(_let(html, "outdoorPoint"), {"t": 4.0, "rh": 85.0})
        self.assertEqual(_let(html, "outdoorOpts"), {
            "color": "#2563eb", "ventilationLine": True, "note": True,
            "threshold": 0.0003})
        self.assertIn("hxDrawOutdoor(top, ctx, outdoorPoint, roomState", html)

    def test_line_and_note_switches(self):
        html = plot_mollier_hx(data=_data(), outdoor=(4, 85), ventilation_line=False,
                               ventilation_note=False, ventilation_threshold=0.001,
                               outdoor_color="#0a0")
        self.assertEqual(_let(html, "outdoorOpts"), {
            "color": "#0a0", "ventilationLine": False, "note": False,
            "threshold": 0.001})

    def test_room_point_is_latest_row(self):
        html = plot_mollier_hx(data=_data(), outdoor=(4, 85))
        room = _let(html, "roomPoint")
        self.assertEqual(room["title"], "Latest")
        self.assertEqual(room["ts"], "2024-01-11 23:00")
        self.assertEqual(room["color"], "black")
        self.assertIn("x", room["state"])

    def test_current_replaces_latest(self):
        html = plot_mollier_hx(data=_data(), current=(21, 50), outdoor=(4, 85))
        self.assertIsNone(_let(html, "currentRecord"))
        room = _let(html, "roomPoint")
        self.assertEqual(room["state"], {"t": 21.0, "rh": 50.0})
        self.assertEqual(room["title"], "Now")
        self.assertEqual(room["legend"], "Room")

    def test_current_without_data(self):
        html = plot_mollier_hx(current=(21, 50))
        self.assertEqual(_let(html, "roomPoint")["state"], {"t": 21.0, "rh": 50.0})

    def test_no_room_point_without_data_or_current(self):
        html = plot_mollier_hx(outdoor=(4, 85))
        self.assertIsNone(_let(html, "roomPoint"))
        html = plot_mollier_hx(data=_data(), highlight_latest=False, outdoor=(4, 85))
        self.assertIsNone(_let(html, "roomPoint"))

    def test_points_drawn_unclipped_on_top(self):
        html = plot_mollier_hx(data=_data(), outdoor=(-30, 80))
        self.assertIn('svg.append("g").attr("class", "hx-top")', html)
        self.assertIn("hxDrawStatePoint(top, ctx, roomPoint.state", html)

    def test_invalid_points_raise(self):
        for fn in BOTH:
            for bad in ((4,), "warm", (4, None), (4, 120), (float("nan"), 50)):
                with self.assertRaises(ValueError, msg=f"{fn.__name__} {bad!r}"):
                    fn(outdoor=bad)
            with self.assertRaises(ValueError):
                fn(current=(21, -5))

    def test_fast_outdoor_and_current(self):
        html = plot_mollier_hx_fast(current=(21, 50), outdoor=(4, 85),
                                    ventilation_line=False, ventilation_note=False,
                                    ventilation_threshold=0.0005)
        self.assertIn('c.setCurrent({"t": 21.0, "rh": 50.0},{color:"black"});', html)
        self.assertIn('c.setOutdoor({"t": 4.0, "rh": 85.0},{"color": "#2563eb", '
                      '"ventilationLine": false, "note": false, "threshold": 0.0005});',
                      html)
        # the room point is set before the outside air (line starts there)
        self.assertLess(html.index("c.setCurrent("), html.index("c.setOutdoor("))

    def test_fast_highlights_latest_row(self):
        html = plot_mollier_hx_fast(data=_data())
        self.assertIn('c.setCurrent({"t": 22.0, "rh": 55.0},{color:"black"});', html)
        self.assertEqual(_chart_opts(html)["texts"]["now"], "Latest: 2024-01-11 23:00")
        html = plot_mollier_hx_fast(data=_data(), highlight_color=None)
        self.assertIn("{color:c.seasonColors[seasonIndex(toDate(", html)
        html = plot_mollier_hx_fast(data=_data(), highlight_latest=False)
        self.assertNotIn("c.setCurrent(", html)

    def test_fast_validates_frequency_unit(self):
        with self.assertRaises(ValueError):
            plot_mollier_hx_fast(frequency_unit="weeks")


class TestClipping(unittest.TestCase):

    def test_data_layers_clipped(self):
        html = plot_mollier_hx(data=_data(), comfort_zone_orange=True,
                               comfort_zone_red=True, show_frequency=True)
        self.assertRegex(html, r'hxAddClipPath\(svg, "clip_\w+", width, height\)')
        self.assertIn('.attr("clip-path", clipUrl)', html)
        # comfort zone, bands, scatter and frequency lines go into `plot`
        for snippet in ('plot.append("path").attr("class", "hx-comfort")',
                        'plot.append("path").attr("class", cls)',
                        'plot.append("g").attr("class", "hx-scatter")',
                        'let freqLines = plot.append("g")'):
            self.assertIn(snippet, html)

    def test_open_abs_humidity_bands(self):
        html = plot_mollier_hx(
            comfort_zone={"temperature": (20, 26), "rel_humidity": (0.3, 0.65),
                          "abs_humidity": None},
            comfort_zone_orange=True, comfort_zone_red=True)
        self.assertEqual(_let(html, "rangeX"), [0, 1.0])
        self.assertIn("hxAddClipPath(svg", html)


class TestCaptionsAndTexts(unittest.TestCase):

    def test_default_captions(self):
        caps = _let(plot_mollier_hx(), "hxOpts")["captions"]
        self.assertEqual(caps, MOLLIER_DEFAULT_CAPTIONS)
        self.assertEqual(_chart_opts(plot_mollier_hx_fast())["captions"],
                         MOLLIER_DEFAULT_CAPTIONS)

    def test_captions_merge_and_hide(self):
        for fn in BOTH:
            html = fn(captions={"density": "", "temperature": "Temperatur ϑ [°C]"})
            caps = (_let(html, "hxOpts") if fn is plot_mollier_hx
                    else _chart_opts(html))["captions"]
            self.assertEqual(caps["density"], "")
            self.assertEqual(caps["temperature"], "Temperatur ϑ [°C]")
            self.assertEqual(caps["enthalpy"], "Enthalpy [kJ/kg]")

    def test_x_axis_title_maps_to_caption(self):
        caps = _let(plot_mollier_hx(x_axis_title="x-Achse"), "hxOpts")["captions"]
        self.assertEqual(caps["x_axis_title"], "x-Achse")
        caps = _let(plot_mollier_hx(x_axis_title="x-Achse",
                                    captions={"x_axis_title": "Wassergehalt"}),
                    "hxOpts")["captions"]
        self.assertEqual(caps["x_axis_title"], "Wassergehalt")
        opts = _chart_opts(plot_mollier_hx_fast(x_axis_title="x-Achse"))
        self.assertEqual(opts["captions"]["x_axis_title"], "x-Achse")
        self.assertEqual(opts["lineOpts"]["xAxisTitle"], "x-Achse")

    def test_unknown_keys_raise(self):
        for fn in BOTH:
            with self.assertRaisesRegex(ValueError, "captions"):
                fn(captions={"temprature": "T"})
            with self.assertRaisesRegex(ValueError, "texts"):
                fn(texts={"outside": "Aussenluft"})

    def test_comfort_label_and_season_labels_map_to_texts(self):
        html = plot_mollier_hx(data=_data(), comfort_label="Komfortzone",
                               season_labels=_SEASON_LABELS_DE)
        self.assertEqual(_let(html, "comfortLabel"), "Komfortzone")
        texts = _let(html, "T")
        self.assertEqual(texts["comfort_zone"], "Komfortzone")
        self.assertEqual(texts["season_spring"], "Frühling")
        self.assertIn("Frühling", _let(html, "colorMap"))
        # an explicit texts key wins over the older argument
        html = plot_mollier_hx(comfort_label="A", texts={"comfort_zone": "B"})
        self.assertEqual(_let(html, "comfortLabel"), "B")

    def test_altitude_and_pressure_lines(self):
        html = plot_mollier_hx(pressure=96000, altitude=450)
        self.assertEqual(_let(html, "infoLines"), ["450 m a.s.l.", "960 hPa"])
        html = plot_mollier_hx(pressure=96000, altitude=450,
                               texts={"altitude": "{alt} m ü. M.", "pressure": ""})
        self.assertEqual(_let(html, "infoLines"), ["450 m ü. M."])
        opts = _chart_opts(plot_mollier_hx_fast(pressure=96000, altitude=450))
        self.assertEqual(opts["texts"]["altitude"], "450 m a.s.l.")
        opts = _chart_opts(plot_mollier_hx_fast(pressure=96000))
        self.assertEqual(opts["texts"]["altitude"], "{alt} m a.s.l.")   # filled in JS

    def test_texts_cannot_close_the_script(self):
        html = plot_mollier_hx(texts={"outdoor": "</script><b>x"})
        self.assertNotIn("</script><b>x", html)
        self.assertIn("<\\/script><b>x", html)

    def test_german_overrides_leave_no_english_default(self):
        captions = {
            "temperature": "Temperatur ϑ [°C]", "rel_humidity": "rel. Feuchte φ [%]",
            "density": "Dichte ρ [kg/m³]", "enthalpy": "Enthalpie h [kJ/kg]",
            "x_axis": "x [g/kg]", "x_axis_title": "Wassergehalt x [g/kg]",
        }
        texts = {
            "outdoor": "Aussenluft", "room": "Raum",
            "ventilation_dries": "Lüften trocknet",
            "ventilation_humidifies": "Lüften befeuchtet",
            "ventilation_neutral": "Lüften: keine Änderung",
            "state_value": "{t} °C · {rh} % r. F.",
            "now": "Jetzt", "tip_x": "x", "tip_t": "T", "tip_phi": "φ",
            "tip_h": "h", "tip_td": "Taupunkt",
            "freq_hours": "{n} h/Jahr ausserhalb",
            "freq_days": "{n} Tage/Jahr ausserhalb",
            "altitude": "{alt} m ü. M.", "pressure": "{hpa} hPa",
            "latest": "Letzter Wert", "comfort_zone": "Komfortzone",
            "season_winter": "Winter", "season_spring": "Frühling",
            "season_summer": "Sommer", "season_fall": "Herbst",
        }
        self.assertEqual(set(captions), set(MOLLIER_DEFAULT_CAPTIONS))
        self.assertEqual(set(texts), set(MOLLIER_DEFAULT_TEXTS))
        english = {**{k: v for k, v in MOLLIER_DEFAULT_CAPTIONS.items()
                      if captions[k] != v},
                   **{k: v for k, v in MOLLIER_DEFAULT_TEXTS.items()
                      if texts[k] != v}}
        for fn in BOTH:
            html = _strip_bundle(fn(
                data=_data(), outdoor=(4, 85), show_frequency=True,
                pressure=96000, captions=captions, texts=texts))
            # Every string literal of the page (JS comments and CSS removed,
            # object keys skipped) — that is where visible text comes from.
            code = re.sub(r"//[^\n]*|style=\"[^\"]*\"", "", html)
            literals = "\n".join(re.findall(r'"((?:[^"\\]|\\.)*)"(?!\s*:)', code))
            for key, value in english.items():
                self.assertNotIn(json.dumps(value, ensure_ascii=False), html,
                                 f"{fn.__name__}: {key}")
                for word in re.findall(r"[A-Za-z][A-Za-z.]{3,}", value):
                    if word == "rel.":   # same in German
                        continue
                    self.assertNotRegex(
                        literals, r"(?<![A-Za-z_])" + re.escape(word) + r"(?![A-Za-z_])",
                        f"{fn.__name__}: {key} {word!r}")
            for value in ("Aussenluft", "Lüften trocknet", "Temperatur ϑ [°C]"):
                self.assertIn(value, html)


class TestVentilationEffect(unittest.TestCase):

    def test_winter_airing_dries(self):
        # 0 °C / 80 % holds far less water than 21 °C / 50 %
        self.assertEqual(ventilation_effect((21, 50), (0, 80)), "dries")

    def test_summer_cellar_gets_moister(self):
        self.assertEqual(ventilation_effect((16, 70), (26, 60)), "humidifies")

    def test_neutral_within_threshold(self):
        self.assertEqual(ventilation_effect((21, 50), (21, 51)), "neutral")
        self.assertEqual(ventilation_effect((21, 50), (21, 51), threshold=0.0),
                         "humidifies")
        self.assertEqual(VENTILATION_THRESHOLD, 0.0003)

    def test_threshold_is_exclusive(self):
        room, out = state(t=21, x=0.0080, p=P_STD), state(t=10, x=0.0083, p=P_STD)
        self.assertEqual(ventilation_effect(room, out, threshold=0.0003 + 1e-12),
                         "neutral")
        self.assertEqual(ventilation_effect(room, out, threshold=0.0002), "humidifies")

    def test_accepts_states_and_pressure(self):
        room = state(t=21, phi=0.5, p=P_STD)
        out = state(t=0, phi=0.8, p=P_STD)
        self.assertEqual(ventilation_effect(room, out), "dries")
        # same (t, rh) holds more water at lower pressure
        self.assertEqual(ventilation_effect((20, 50), (20, 50), pressure=80000),
                         "neutral")

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            ventilation_effect((21,), (0, 80))


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
