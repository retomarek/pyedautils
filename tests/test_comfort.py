# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pyedautils import comfort
from pyedautils.plots.comfort import (
    _COMPASS_NAMES_DE,
    _COMPASS_STAGE_NAMES_DE,
    plot_comfort_compass,
    plot_comfort_donuts,
    plot_comfort_sia180,
    plot_overheating_bar,
    plot_overheating_timeseries,
    plot_temp_humidity_timeseries,
)


class TestSia180Curves(unittest.TestCase):
    def test_max_temp_plateaus_and_slope(self):
        """Upper boundary: 24.5 below 12 °C, 26.5 above 17.5 °C, linear between."""
        got = comfort.sia180_max_temp([-5, 12, 17.5, 40])
        self.assertTrue(np.allclose(got, [24.5, 24.5, 26.5, 26.5]))
        # midpoint 14.75 °C -> halfway between 24.5 and 26.5
        self.assertAlmostEqual(float(comfort.sia180_max_temp(14.75)), 25.5)

    def test_min_temp_plateaus_and_slope(self):
        """Lower boundary: 20.5 below 19 °C, 22.0 above 23.5 °C, linear between."""
        got = comfort.sia180_min_temp([-5, 19, 23.5, 40])
        self.assertTrue(np.allclose(got, [20.5, 20.5, 22.0, 22.0]))

    def test_scalar_input(self):
        self.assertEqual(float(comfort.sia180_max_temp(5.0)), 24.5)


class TestSummerSemester(unittest.TestCase):
    def test_boundaries_inclusive(self):
        ts = pd.to_datetime([
            "2024-04-15", "2024-04-16", "2024-07-01",
            "2024-10-15", "2024-10-16", "2024-01-01",
        ])
        mask = comfort.is_summer_semester_sia180(ts).tolist()
        self.assertEqual(mask, [False, True, True, True, False, False])


class TestAlignAndOverheating(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2024-07-01", periods=72, freq="h")
        self.room = pd.DataFrame({"timestamp": idx, "value": np.linspace(22, 30, 72)})
        self.outdoor = pd.DataFrame({"timestamp": idx, "value": np.linspace(15, 25, 72)})

    def test_align_hourly_columns(self):
        al = comfort.align_hourly(self.room, self.outdoor)
        self.assertListEqual(list(al.columns), ["t_room", "t_oa", "t_oa_48h"])
        self.assertFalse(al.empty)

    def test_align_hourly_empty(self):
        al = comfort.align_hourly(pd.DataFrame(), self.outdoor)
        self.assertTrue(al.empty)

    def test_align_hourly_microsecond_resolution(self):
        """Regression: datetime64[us] inputs (e.g. from Parquet) must still join.

        pandas >= 2.0 keeps non-nanosecond units; mismatched units silently
        break concat alignment, yielding an almost-empty result.
        """
        room = self.room.copy()
        outdoor = self.outdoor.copy()
        room["timestamp"] = room["timestamp"].astype("datetime64[us]")
        outdoor["timestamp"] = outdoor["timestamp"].astype("datetime64[us]")
        al = comfort.align_hourly(room, outdoor)
        # full overlap -> dozens of aligned hours, not a handful
        self.assertGreater(len(al), 50)

    def test_overheating_hours_adaptive(self):
        al = comfort.align_hourly(self.room, self.outdoor)
        total, thr = comfort.overheating_hours(al, summer_only=True)
        self.assertGreater(total, 0)
        self.assertEqual(len(thr), len(thr.dropna()))

    def test_overheating_hours_fixed(self):
        al = comfort.align_hourly(self.room, self.outdoor)
        total, thr = comfort.overheating_hours(al, method="fixed", summer_only=False)
        self.assertTrue((thr == comfort.FIXED_OVERHEATING_THRESHOLD).all())

    def test_comfort_kpis_keys(self):
        al = comfort.align_hourly(self.room, self.outdoor)
        kpis = comfort.comfort_kpis(al)
        for k in ("n_hours", "h_cold", "h_warm", "h_ok", "pct_ok",
                  "overheating_h", "sia180_compliant", "minergie_compliant"):
            self.assertIn(k, kpis)

    def test_overheating_per_month(self):
        al = comfort.align_hourly(self.room, self.outdoor)
        _, thr = comfort.overheating_hours(al, summer_only=False)
        monthly = comfort.overheating_per_month(al, thr)
        self.assertListEqual(list(monthly.columns), ["month", "hours"])

    def test_overheating_per_month_empty(self):
        empty = pd.DataFrame(columns=["t_room", "t_oa", "t_oa_48h"])
        monthly = comfort.overheating_per_month(empty, pd.Series(dtype=float))
        self.assertTrue(monthly.empty)

    def test_business_hours_only(self):
        al = comfort.align_hourly(self.room, self.outdoor)
        total, thr = comfort.overheating_hours(
            al, summer_only=True, business_hours_only=True)
        self.assertTrue(set(thr.index.hour) <= set(range(7, 22)))

    def test_comfort_kpis_summer_only(self):
        al = comfort.align_hourly(self.room, self.outdoor)
        kpis = comfort.comfort_kpis(al, summer_only=True)
        self.assertGreater(kpis["n_hours"], 0)

    def test_comfort_kpis_empty_window(self):
        # July data, but a (hypothetical) winter-only filter empties it
        idx = pd.date_range("2024-01-01", periods=72, freq="h")
        room = pd.DataFrame({"timestamp": idx, "value": np.linspace(20, 22, 72)})
        out = pd.DataFrame({"timestamp": idx, "value": np.linspace(0, 5, 72)})
        al = comfort.align_hourly(room, out)
        kpis = comfort.comfort_kpis(al, summer_only=True)  # winter -> empty
        self.assertEqual(kpis["n_hours"], 0)
        self.assertFalse(kpis["sia180_compliant"])

    def test_overheating_hours_empty(self):
        idx = pd.date_range("2024-01-01", periods=72, freq="h")
        room = pd.DataFrame({"timestamp": idx, "value": np.linspace(20, 22, 72)})
        out = pd.DataFrame({"timestamp": idx, "value": np.linspace(0, 5, 72)})
        al = comfort.align_hourly(room, out)
        total, thr = comfort.overheating_hours(al, summer_only=True)  # winter
        self.assertEqual(total, 0.0)
        self.assertTrue(thr.empty)


class TestComfortPlots(unittest.TestCase):
    def test_sia180_plot_uses_curve_functions(self):
        """The refactored plot must draw the exact adaptive boundary y-values."""
        idx = pd.date_range("2024-01-01", periods=200, freq="h")
        outdoor = pd.DataFrame({"timestamp": idx, "value": np.linspace(-5, 30, 200)})
        room = pd.DataFrame({"timestamp": idx, "value": np.linspace(20, 27, 200)})
        fig = plot_comfort_sia180(outdoor, room)
        self.assertIsInstance(fig, go.Figure)
        lower = next(t for t in fig.data if t.name == "Lower limit SIA 180")
        self.assertTrue(np.allclose(
            comfort.sia180_min_temp(lower.x), lower.y))
        upper = next(t for t in fig.data if t.name == "Upper limit active cooling")
        self.assertTrue(np.allclose(
            comfort.sia180_max_temp(upper.x), upper.y))

    def test_donuts(self):
        d = pd.DataFrame({"temperature": [18, 22, 28, 21],
                          "humidity": [20, 50, 70, 45]})
        fig = plot_comfort_donuts(d)
        self.assertEqual(len(fig.data), 2)
        # temperature donut: 1 cold, 2 in range, 1 warm
        self.assertEqual(list(fig.data[0].values), [1, 2, 1])

    def test_donuts_bullet_legend(self):
        d = pd.DataFrame({"temperature": [18, 22, 28], "humidity": [20, 50, 70]})
        fig = plot_comfort_donuts(d, count_label="days")
        # native legend is off; legend is rendered as bullet annotations
        self.assertFalse(fig.layout.showlegend)
        bullets = [a for a in fig.layout.annotations if "●" in a.text]
        # one centred multi-line annotation per donut, each with 3 bullets
        self.assertEqual(len(bullets), 2)
        for a in bullets:
            self.assertEqual(a.text.count("●"), 3)
            self.assertEqual(a.text.count("<br>"), 2)   # 3 stacked lines
            self.assertEqual(a.xanchor, "center")        # centred block
            self.assertEqual(a.align, "left")            # bullets aligned
            self.assertIn("days", a.text)
        self.assertTrue(any("In range" in a.text for a in bullets))

    def test_donuts_english_labels(self):
        d = pd.DataFrame({"temperature": [18, 22, 28], "humidity": [20, 50, 70]})
        fig = plot_comfort_donuts(d)
        self.assertEqual(list(fig.data[0].labels), ["Too cold", "In range", "Too warm"])
        self.assertEqual(list(fig.data[1].labels), ["Too dry", "In range", "Too humid"])

    def test_donuts_hover_shows_percent_and_count(self):
        d = pd.DataFrame({"temperature": [18, 22, 28], "humidity": [20, 50, 70]})
        fig = plot_comfort_donuts(d, count_label="days")
        ht = fig.data[0].hovertemplate
        self.assertIn("%{percent}", ht)
        self.assertIn("days", ht)
        cd = list(np.ravel(fig.data[0].customdata))
        self.assertEqual(cd, ["Too cold", "In range", "Too warm"])

    def test_donuts_empty_data(self):
        # all-NaN -> zero totals (no percent text) and "—" center stats
        d = pd.DataFrame({"temperature": [np.nan, np.nan],
                          "humidity": [np.nan, np.nan]})
        fig = plot_comfort_donuts(d)
        self.assertTrue(all(txt == "" for txt in fig.data[0].text))
        texts = " ".join(a.text for a in fig.layout.annotations)
        self.assertIn("—", texts)

    def test_donuts_center_stats(self):
        d = pd.DataFrame({"temperature": [18, 22, 28], "humidity": [20, 50, 70]})
        texts = " ".join(a.text for a in
                         plot_comfort_donuts(d, show_center_stats=True).layout.annotations)
        # average value with unit appears only in the centre stats
        self.assertIn("°C", texts)
        texts2 = " ".join(a.text for a in
                          plot_comfort_donuts(d, show_center_stats=False).layout.annotations)
        self.assertNotIn("°C", texts2)

    def test_overheating_bar_monthly(self):
        monthly = pd.DataFrame({"month": [6, 7, 8], "hours": [12, 40, 25]})
        fig = plot_overheating_bar(monthly)
        self.assertEqual(len(fig.data), 1)
        # all 12 months on the x-axis, Jan–Dec
        self.assertEqual(list(fig.data[0].x), ["Jan", "Feb", "Mar", "Apr", "May",
                                               "Jun", "Jul", "Aug", "Sep", "Oct",
                                               "Nov", "Dec"])
        # July value present, missing months zero-filled
        self.assertEqual(fig.data[0].y[6], 40)   # Jul
        self.assertEqual(fig.data[0].y[0], 0)    # Jan

    def test_overheating_bar_empty(self):
        empty = pd.DataFrame(columns=["month", "hours"])
        fig = plot_overheating_bar(empty)
        self.assertEqual(list(fig.data[0].y), [0] * 12)

    def test_overheating_timeseries(self):
        idx = pd.date_range("2024-07-01", periods=72, freq="h")
        room = pd.DataFrame({"timestamp": idx, "value": np.linspace(22, 30, 72)})
        out = pd.DataFrame({"timestamp": idx, "value": np.linspace(15, 25, 72)})
        aligned = comfort.align_hourly(room, out)
        fig = plot_overheating_timeseries(aligned, method="adaptive", summer_only=True)
        names = [tr.name for tr in fig.data]
        self.assertIn("Room temperature", names)
        self.assertIn("Overheating", names)
        # threshold line carries the SIA 180 label in adaptive mode
        self.assertTrue(any("SIA 180" in n for n in names))

    def test_overheating_timeseries_fixed(self):
        idx = pd.date_range("2024-07-01", periods=72, freq="h")
        room = pd.DataFrame({"timestamp": idx, "value": np.linspace(22, 30, 72)})
        out = pd.DataFrame({"timestamp": idx, "value": np.linspace(15, 25, 72)})
        aligned = comfort.align_hourly(room, out)
        fig = plot_overheating_timeseries(aligned, method="fixed", summer_only=False)
        self.assertTrue(any("26.5" in (tr.name or "") for tr in fig.data))


class TestTempHumidityTimeseries(unittest.TestCase):
    def _df(self):
        idx = pd.date_range("2024-01-01", periods=72, freq="h")
        return pd.DataFrame({
            "timestamp": idx,
            "temperature": np.linspace(10, 25, 72),
            "humidity": np.linspace(35, 75, 72),
        })

    def test_lines_bands_and_legend(self):
        fig = plot_temp_humidity_timeseries(
            self._df(), temp_band=(18, 22), hum_band=(40, 60),
            temp_band_orange=(17, 23), temp_band_red=(16, 24),
            hum_band_orange=(35, 65), hum_band_red=(30, 70),
            temp_title="Temperature", hum_title="abs. Humidity")
        self.assertIsInstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        for n in ("Temperature", "Humidity", "Comfort band",
                  "Moderate", "Severe"):
            self.assertIn(n, names)
        # per row: 1 green rect + 2 green edge lines + 2 orange + 2 red = 7
        self.assertEqual(len(fig.layout.shapes), 14)
        # optional per-subplot titles are rendered as annotations
        ann = [a.text for a in fig.layout.annotations]
        self.assertIn("Temperature", ann)
        self.assertIn("abs. Humidity", ann)
        # hover uses T / phi and the h,x-diagram date format
        temp = next(t for t in fig.data if t.name == "Temperature")
        self.assertIn("T = ", temp.hovertemplate)
        self.assertIn("%Y-%m-%d", temp.hovertemplate)

    def test_no_bands_daily_only(self):
        # 72 hourly rows -> 3 daily means; bands disabled -> no legend swatches
        df = self._df().assign(temperature=20.0, humidity=50.0)
        fig = plot_temp_humidity_timeseries(
            df, show_hourly=False,
            temp_band=None, hum_band=None,
            temp_band_orange=None, hum_band_orange=None,
            temp_band_red=None, hum_band_red=None)
        line = next(t for t in fig.data if t.name == "Temperature")
        self.assertEqual(len(line.x), 3)
        self.assertNotIn("Comfort band", [t.name for t in fig.data])
        self.assertEqual(len(fig.layout.shapes), 0)

    def test_hourly_only_no_daily(self):
        fig = plot_temp_humidity_timeseries(self._df(), show_daily_mean=False)
        # only the faint hourly markers, no daily-mean line
        self.assertTrue(any(t.mode == "markers" for t in fig.data))
        self.assertFalse(any(t.name == "Temperature" for t in fig.data))

    def test_single_column_and_empty(self):
        # humidity column absent -> second subplot stays empty (guard hit)
        df = self._df()[["timestamp", "temperature"]]
        fig = plot_temp_humidity_timeseries(df, show_hourly=False)
        self.assertNotIn("Humidity", [t.name for t in fig.data])
        # empty frame -> no daily resample, still returns a figure
        empty = self._df().iloc[0:0]
        self.assertIsInstance(
            plot_temp_humidity_timeseries(empty), go.Figure)


class TestComfortCompass(unittest.TestCase):
    def test_categories(self):
        cats = comfort.comfort_compass_categories()
        self.assertEqual(len(cats), 25)        # "ok" + 8 directions x 3 stages
        self.assertEqual(cats[0], "ok")
        self.assertIn("w_l", cats)
        self.assertIn("f_s", cats)

    def test_distribution_states_and_stages(self):
        # one row per day so daily means equal the given values
        idx = pd.date_range("2024-01-01", periods=6, freq="D")
        df = pd.DataFrame({
            #          ok  warm-mild warm-mod warm-sev cold-mod warm+humid
            "temperature": [22, 24.5, 26.0, 27.5, 18.0, 26.0],
            "humidity":    [40, 40.0, 40.0, 40.0, 40.0, 56.0],
        }, index=idx)
        dist = comfort.comfort_compass_distribution(df, (20, 24), (30, 50))
        self.assertEqual(sum(dist.values()), 6)             # counts = days
        self.assertEqual(dist["ok"], 1)
        self.assertEqual(dist["w_l"], 1)   # +0.5 K  -> mild
        self.assertEqual(dist["w_d"], 1)   # +2.0 K  -> moderate
        self.assertEqual(dist["w_s"], 1)   # +3.5 K  -> severe
        self.assertEqual(dist["c_d"], 1)   # -2.0 K  -> moderate cold
        self.assertEqual(dist["wf_d"], 1)  # warm+humid, worse axis -> moderate

    def test_distribution_aggregate_false_empty_and_error(self):
        # aggregate_daily=False counts the rows as given
        df = pd.DataFrame({"temperature": [22, 22, 28], "humidity": [40, 40, 40]})
        dist = comfort.comfort_compass_distribution(
            df, (20, 24), (30, 50), aggregate_daily=False)
        self.assertEqual(dist["ok"], 2)
        self.assertEqual(dist["w_s"], 1)   # +4 K -> severe
        # daily aggregation requires a DatetimeIndex
        with self.assertRaises(ValueError):
            comfort.comfort_compass_distribution(df, (20, 24), (30, 50))
        # all-NaN -> all zeros
        e = pd.DataFrame({"temperature": [np.nan], "humidity": [np.nan]},
                         index=pd.date_range("2024-01-01", periods=1, freq="D"))
        de = comfort.comfort_compass_distribution(e, (20, 24), (30, 50))
        self.assertEqual(sum(de.values()), 0)

    def test_distribution_absolute_humidity(self):
        # 30 °C / 51 % rH ~ 13.6 g/kg: within the relative band (30–65 %) but
        # above the absolute cap (12 g/kg) -> warm+humid once hum_abs_band is set.
        df = pd.DataFrame({"temperature": [30.0], "humidity": [51.0]})
        rel = comfort.comfort_compass_distribution(
            df, (20, 26), (30, 65), aggregate_daily=False)
        self.assertEqual(rel["w_s"], 1)            # too warm only (relative)
        relabs = comfort.comfort_compass_distribution(
            df, (20, 26), (30, 65), hum_abs_band=(0.0, 0.012),
            aggregate_daily=False)
        self.assertEqual(relabs["wf_s"], 1)        # warm + humid (rel OR abs)
        self.assertEqual(relabs["w_s"], 0)

    def test_plot_basics_and_zoom_off(self):
        fig = plot_comfort_compass({"ok": 50, "f_l": 30, "w_d": 20}, title="Room A")
        self.assertIsInstance(fig, go.Figure)
        self.assertFalse(fig.layout.dragmode)                 # zoom disabled
        self.assertTrue(any(t.type == "barpolar" for t in fig.data))

    def test_plot_legend_counts_and_toggle(self):
        fig = plot_comfort_compass({"ok": 50, "f_l": 30, "w_d": 20},
                                   count_label="days")
        bullets = [a for a in fig.layout.annotations if "●" in a.text]
        self.assertEqual(len(bullets), 1)
        self.assertIn("days", bullets[0].text)
        self.assertIn("in range", bullets[0].text)            # lower-case default
        # legend is optional
        off = plot_comfort_compass({"ok": 50, "f_l": 50}, show_legend=False)
        self.assertFalse(any("●" in a.text for a in off.layout.annotations))

    def test_plot_overrides_german(self):
        fig = plot_comfort_compass(
            {"ok": 50, "f_l": 30, "w_d": 20}, count_label="Tage",
            names=_COMPASS_NAMES_DE, stage_names=_COMPASS_STAGE_NAMES_DE)
        bullet = next(a for a in fig.layout.annotations if "●" in a.text)
        self.assertIn("Tage", bullet.text)
        self.assertIn("im Zielband", bullet.text)
        hov = "".join("".join(t.hovertext) for t in fig.data
                      if getattr(t, "hovertext", None))
        self.assertIn("mittel", hov)                          # German stage name

    def test_plot_stats_text(self):
        # with the legend on, the stats block is appended below the legend
        # rows (same annotation, so it stays attached and left-aligned)
        fig = plot_comfort_compass({"ok": 50, "f_l": 30, "w_d": 20},
                                   stats_text="T min 18 / max 24 °C")
        bullet = next(a for a in fig.layout.annotations if "●" in a.text)
        self.assertIn("T min 18 / max 24 °C", bullet.text)
        # without the legend it becomes its own left-aligned annotation
        solo = plot_comfort_compass({"ok": 50, "f_l": 50}, show_legend=False,
                                    stats_text="T min 18 / max 24 °C")
        anns = [a for a in solo.layout.annotations
                if "T min 18 / max 24 °C" in a.text]
        self.assertEqual(len(anns), 1)
        self.assertEqual(anns[0].xanchor, "left")
        # off by default
        plain = plot_comfort_compass({"ok": 50, "f_l": 50})
        self.assertFalse(any("T min" in a.text
                             for a in plain.layout.annotations))

    def test_plot_count_label_singular(self):
        # a count of exactly 1 uses the singular word in legend and hover
        fig = plot_comfort_compass({"ok": 1, "f_l": 30},
                                   count_label="Tage",
                                   count_label_singular="Tag")
        bullet = next(a for a in fig.layout.annotations if "●" in a.text)
        self.assertIn("<b>1</b> Tag ", bullet.text)
        self.assertIn("<b>30</b> Tage ", bullet.text)
        hov = "".join("".join(t.hovertext) for t in fig.data
                      if getattr(t, "hovertext", None))
        self.assertIn("1 Tag (", hov)
        # without the singular word the plural label is always used
        plain = plot_comfort_compass({"ok": 1, "f_l": 30}, count_label="Tage")
        b2 = next(a for a in plain.layout.annotations if "●" in a.text)
        self.assertIn("<b>1</b> Tage ", b2.text)
        # title subtitle also switches for a total of exactly 1
        one = plot_comfort_compass({"ok": 1}, title="R", count_label="Tage",
                                   count_label_singular="Tag")
        self.assertIn("1 Tag<", one.layout.title.text)

    def test_severity_lightness_planes(self):
        # Severity must be readable from the shade alone: within every
        # direction mild is the lightest and severe the darkest colour, and
        # each stage sits on one shared lightness plane across all directions.
        def luma(hexcol):
            h = hexcol.lstrip("#")
            r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        from pyedautils.plots.comfort import (_COMPASS_STAGE_COLORS,
                                              _COMPASS_STAGE_GREYS)
        for k, ramp in _COMPASS_STAGE_COLORS.items():
            self.assertGreater(luma(ramp["l"]), luma(ramp["d"]), k)
            self.assertGreater(luma(ramp["d"]), luma(ramp["s"]), k)
        # every "mild" is lighter than every "severe", regardless of direction
        min_mild = min(luma(r["l"]) for r in _COMPASS_STAGE_COLORS.values())
        max_severe = max(luma(r["s"]) for r in _COMPASS_STAGE_COLORS.values())
        self.assertGreater(min_mild, max_severe)
        self.assertGreater(luma(_COMPASS_STAGE_GREYS["l"]),
                           luma(_COMPASS_STAGE_GREYS["s"]))

    def test_plot_severity_key_in_legend(self):
        fig = plot_comfort_compass({"ok": 50, "f_l": 30, "w_d": 20})
        legend = next(a for a in fig.layout.annotations if "●" in a.text)
        for stage in ("mild", "moderate", "severe"):
            self.assertIn(stage, legend.text)
        # deviation rows carry the mild/moderate/severe split of their share
        self.assertIn("(30% → 30/0/0%)", legend.text)   # f: all mild
        self.assertIn("(20% → 0/20/0%)", legend.text)   # w: all moderate
        self.assertNotIn("50% →", legend.text)          # "ok" row has no split
        # no severity key when there is no deviation at all
        clean = plot_comfort_compass({"ok": 50})
        legend = next(a for a in clean.layout.annotations if "●" in a.text)
        self.assertNotIn("severe", legend.text)

    def test_plot_direction_labels_hug_glyph(self):
        def label_radius(fig):
            for t in fig.data:
                if getattr(t, "mode", None) == "text" and len(t.r or []) == 8:
                    return t.r[0]
        # 100 % in range -> labels sit just outside the reference circle ...
        near = label_radius(plot_comfort_compass({"ok": 100}))
        # ... one dominant arm -> labels move out to the longest arm
        far = label_radius(plot_comfort_compass({"ok": 5, "w_s": 95}))
        self.assertLess(near, 1.5)
        self.assertGreater(far, 2.0)
        self.assertLess(near, far)

    def test_plot_fixed_scale_keeps_circle_size(self):
        def r_range(fig):
            return tuple(fig.layout.polar.radialaxis.range)
        # fixed_scale (default): radial range is identical regardless of the
        # distribution, so the dashed reference circle stays the same size and
        # charts remain comparable.
        small = r_range(plot_comfort_compass({"ok": 100}))
        big = r_range(plot_comfort_compass({"ok": 5, "w_s": 95}))
        self.assertEqual(small, big)
        # fixed_scale=False: range zooms to each glyph, so a small glyph gets a
        # tighter range than a long-armed one (circle differs between charts).
        small_dyn = r_range(plot_comfort_compass({"ok": 100}, fixed_scale=False))
        big_dyn = r_range(plot_comfort_compass({"ok": 5, "w_s": 95},
                                               fixed_scale=False))
        self.assertLess(small_dyn[1], big_dyn[1])

    def test_plot_centre_pct_threshold(self):
        def centre_pct(fig):
            for t in fig.data:
                if getattr(t, "mode", None) == "text" and list(t.r or []) == [0]:
                    return t.text[0]
            return None
        self.assertEqual(centre_pct(plot_comfort_compass({"ok": 40, "f_l": 60})),
                         "40%")
        # below 15 % the centre value is dropped (disc too small)
        self.assertIsNone(centre_pct(plot_comfort_compass({"ok": 5, "f_l": 95})))


if __name__ == "__main__":
    unittest.main()
