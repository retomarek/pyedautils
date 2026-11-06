"""Thermal comfort and psychrometric chart plots."""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go

from pyedautils.comfort import (
    ADAPTIVE_MAX_T_OA_HI,
    ADAPTIVE_MAX_T_OA_LO,
    ADAPTIVE_MIN_T_OA_HI,
    ADAPTIVE_MIN_T_OA_LO,
    FIXED_OVERHEATING_THRESHOLD,
    overheating_hours,
    sia180_max_temp,
    sia180_min_temp,
)
from pyedautils.plots._constants import DEFAULT_SEASON_COLORS


def plot_comfort_sia180(
    data_outdoor: pd.DataFrame,
    data_room: pd.DataFrame,
    title: str = "Thermal Comfort according to SIA 180:2014",
    colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    SIA 180:2014 thermal comfort plot.

    Scatter of room temperature vs 48-hour rolling mean outdoor
    temperature, colored by season, with SIA 180 comfort boundaries.

    Args:
        data_outdoor: DataFrame ``[timestamp, value]`` with outdoor temp.
        data_room: DataFrame ``[timestamp, value]`` with room temp.
        title: Plot title.
        colors: Season color overrides.

    Returns:
        go.Figure

    See Also:
        Why temperature and humidity belong together, and how to save
        energy without losing comfort:
        https://retomarek.github.io/pyedautils/guides/mollier_hx.html#hx-comfort
    """
    from pyedautils.data_prep.season import get_season

    c = {**DEFAULT_SEASON_COLORS, **(colors or {})}

    # Outdoor: hourly mean, fill gaps, 48h rolling mean
    df_oa = data_outdoor.copy()
    df_oa.columns = ["timestamp", "value"]
    df_oa["timestamp"] = pd.to_datetime(df_oa["timestamp"])
    df_oa["hour"] = df_oa["timestamp"].dt.floor("h")
    df_oa = df_oa.groupby("hour")["value"].mean().reset_index()
    df_oa.columns = ["timestamp", "temp_oa"]
    full = pd.date_range(df_oa["timestamp"].min(),
                         df_oa["timestamp"].max(), freq="h")
    df_oa = df_oa.set_index("timestamp").reindex(full).interpolate()
    df_oa["temp_oa_48h"] = df_oa["temp_oa"].rolling(48, min_periods=1).mean()
    df_oa = df_oa.dropna(subset=["temp_oa_48h"]).reset_index()
    df_oa.columns = ["timestamp", "temp_oa", "temp_oa_48h"]

    # Room: hourly mean
    df_r = data_room.copy()
    df_r.columns = ["timestamp", "value"]
    df_r["timestamp"] = pd.to_datetime(df_r["timestamp"])
    df_r["hour"] = df_r["timestamp"].dt.floor("h")
    df_r = df_r.groupby("hour")["value"].mean().reset_index()
    df_r.columns = ["timestamp", "temp_r"]

    # Merge
    data = df_oa[["timestamp", "temp_oa_48h"]].merge(
        df_r, on="timestamp", how="inner"
    ).dropna()
    data["season"] = get_season(data["timestamp"])

    # Axis ranges
    min_x = min(0, data["temp_oa_48h"].min())
    max_x = max(28, data["temp_oa_48h"].max())
    min_y = min(21, data["temp_r"].min()) - 1
    max_y = max(32, data["temp_r"].max()) + 1

    fig = go.Figure()

    # SIA 180 boundaries (adaptive curves from pyedautils.comfort —
    # single source of truth, see sia180_min_temp / sia180_max_temp).
    # Lower limit (heating setpoint): constant below 19 °C, linear to
    # 23.5 °C, constant above.
    lower_x = [min_x, ADAPTIVE_MIN_T_OA_LO, ADAPTIVE_MIN_T_OA_HI, max_x]
    fig.add_trace(go.Scatter(
        x=lower_x,
        y=sia180_min_temp(lower_x),
        mode="lines", name="Lower limit SIA 180",
        line=dict(color="#440154", width=2),
    ))
    # Upper limit active cooling: constant below 12 °C, linear to
    # 17.5 °C, constant above.
    upper_x = [min_x, ADAPTIVE_MAX_T_OA_LO, ADAPTIVE_MAX_T_OA_HI, max_x]
    fig.add_trace(go.Scatter(
        x=upper_x,
        y=sia180_max_temp(upper_x),
        mode="lines", name="Upper limit active cooling",
        line=dict(color="#1E9B8A", width=2),
    ))
    # Upper limit passive cooling
    fig.add_trace(go.Scatter(
        x=[min_x, 10, max_x],
        y=[25, 25, 0.33 * max_x + 21.8],
        mode="lines", name="Upper limit passive cooling",
        line=dict(color="#FDE725", width=2),
    ))

    # Scatter by season
    for season in ["Spring", "Summer", "Fall", "Winter"]:
        s = data[data["season"] == season]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["temp_oa_48h"], y=s["temp_r"],
            mode="markers", name=season,
            marker=dict(color=c.get(season, "#999"), size=5, opacity=0.3),
            hovertemplate=(
                "T_room: %{y:.1f} °C<br>"
                "T_oa (48h): %{x:.1f} °C<br>"
                "Date: %{customdata}<br>"
                f"Season: {season}<extra></extra>"
            ),
            customdata=s["timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
        ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        xaxis=dict(
            title="Moving avg outdoor temp (48h) [°C]",
            range=[min_x, max_x], dtick=2,
        ),
        yaxis=dict(
            title="Room Temperature [°C]",
            range=[min_y, max_y], dtick=1,
        ),
    )
    return fig


def plot_comfort_temp_humidity(
    data: pd.DataFrame,
    title: str = "Temperature vs Humidity Comfort Plot",
    colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Scatter of daily mean temperature vs humidity with comfort zones.

    Shows two comfort zone polygons: "comfortable" (green) and
    "still comfortable" (orange), based on common building standards.

    Args:
        data: DataFrame with ``timestamp``, ``temperature`` and ``humidity``
            columns (read by name, so the order does not matter). Humidity in
            %rH, temperature in °C.
        title: Plot title.
        colors: Season color overrides.

    Returns:
        go.Figure

    See Also:
        The same data as an h,x diagram (:func:`plot_mollier_hx`), and how
        to read comfort zone, mould risk and energy saving from it:
        https://retomarek.github.io/pyedautils/guides/mollier_hx.html#hx-reading-the-diagram
    """
    from pyedautils.data_prep.season import get_season

    c = {**DEFAULT_SEASON_COLORS, **(colors or {})}

    # Read by column name (like plot_comfort_donuts / comfort_compass_*) so the
    # column order does not matter; a timestamp + temperature + humidity column
    # are required.
    df = data.copy()
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df["timestamp"] = pd.to_datetime(df[ts_col])
    df["day"] = df["timestamp"].dt.date

    daily = df.groupby("day").agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
    ).reset_index()
    daily["timestamp"] = pd.to_datetime(daily["day"])
    daily["season"] = get_season(daily["timestamp"])

    fig = go.Figure()

    # Comfort zones (polygons)
    # "Still comfortable"
    still_t = [20, 17, 16, 17, 21.5, 25, 27, 25.5, 20]
    still_h = [20, 40, 75, 85, 80, 60, 30, 20, 20]
    fig.add_trace(go.Scatter(
        x=still_t, y=still_h,
        mode="lines",
        fill="toself", fillcolor="rgba(255,165,0,0.25)",
        line=dict(color="orange"),
        name="Still comfortable",
    ))

    # "Comfortable"
    comf_t = [19, 17.5, 22.5, 24, 19]
    comf_h = [38, 74, 65, 35, 38]
    fig.add_trace(go.Scatter(
        x=comf_t, y=comf_h,
        mode="lines",
        fill="toself", fillcolor="rgba(154,205,50,0.4)",
        line=dict(color="yellowgreen"),
        name="Comfortable",
    ))

    # Scatter by season
    for season in ["Spring", "Summer", "Fall", "Winter"]:
        s = daily[daily["season"] == season]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["temperature"], y=s["humidity"],
            mode="markers", name=season,
            marker=dict(color=c.get(season, "#999"), size=6, opacity=0.5),
            hovertemplate=(
                "Temp: %{x:.1f} °C<br>"
                "Hum: %{y:.1f} %rH<br>"
                "Date: %{customdata}<br>"
                f"Season: {season}<extra></extra>"
            ),
            customdata=s["day"].astype(str),
        ))

    min_x = min(14, daily["temperature"].min())
    max_x = max(28, daily["temperature"].max())

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        xaxis=dict(title="Temperature [°C]",
                   range=[min_x, max_x], dtick=2),
        yaxis=dict(title="Humidity [%rH]",
                   range=[0, 100], dtick=20),
    )
    return fig


#: D3 build loaded by the generated pages (the bundled d3-mollierhx sources
#: target D3 v7).
_D3_URL = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"


def _load_d3_js():
    """Load the bundled d3-mollierhx sources used by :func:`plot_mollier_hx`."""
    from importlib import resources as _res
    d3_dir = _res.files("pyedautils") / "data" / "d3_mollierhx"
    return {
        "mollier_functions": (d3_dir / "mollierFunctions.js").read_text(encoding="utf-8"),
        "coordinate_generator": (d3_dir / "coordinateGenerator.js").read_text(encoding="utf-8"),
        "draw_comfort": (d3_dir / "drawComfort.js").read_text(encoding="utf-8"),
    }


def _get_season_fast(dt):
    """Fast season detection without ephem (for D3 data prep)."""
    m = dt.month
    if m in (12, 1, 2):
        return "Winter"
    elif m in (3, 4, 5):
        return "Spring"
    elif m in (6, 7, 8):
        return "Summer"
    return "Fall"


#: Captions of the h,x-diagram grid (English defaults) for the ``captions``
#: argument of :func:`plot_mollier_hx` / :func:`plot_mollier_hx_fast`. Copy
#: and translate as needed; ``""`` hides a caption. ``x_axis_title`` defaults
#: to the ``x_axis_title`` argument.
MOLLIER_DEFAULT_CAPTIONS: Dict[str, str] = {
    "temperature": "Temperature ϑ [°C]",     # left axis, rotated
    "rel_humidity": "rel. humidity φ [%]",   # right edge, top
    "density": "Density ρ [kg/m³]",          # right edge, middle
    "enthalpy": "Enthalpy [kJ/kg]",          # right edge, bottom
    "x_axis": "x [g/kg]",                    # right margin, end of the x axis
    "x_axis_title": "absolute water content x [g/kg]",  # below the x axis
}

#: All other visible texts of :func:`plot_mollier_hx` /
#: :func:`plot_mollier_hx_fast` (English defaults) for their ``texts``
#: argument. Placeholders in braces are filled in: ``{t}`` °C, ``{rh}`` %,
#: ``{x}`` g/kg, ``{n}`` count, ``{alt}`` m, ``{hpa}`` hPa. ``""`` hides the
#: altitude / pressure line and the ventilation note.
MOLLIER_DEFAULT_TEXTS: Dict[str, str] = {
    # outside air and ventilation (d3-mollierhx HX_DEFAULT_TEXTS)
    "outdoor": "Outside air",
    "room": "Room",
    "ventilation_dries": "ventilating dries",
    "ventilation_humidifies": "ventilating adds moisture",
    "ventilation_neutral": "ventilating: no change",
    "state_value": "{t} °C · {rh} %",        # label of an edge arrow
    # tooltips and info (d3-mollierhx MOLLIER_CHART_DEFAULT_TEXTS)
    "now": "Now",                            # tooltip title of the room point
    "tip_x": "x", "tip_t": "T", "tip_phi": "φ", "tip_h": "h", "tip_td": "Td",
    "freq_hours": "{n} h/year outside",
    "freq_days": "{n} days/year outside",
    "altitude": "{alt} m a.s.l.",
    "pressure": "{hpa} hPa",
    # legend and latest-row tooltip (plot_mollier_hx)
    "latest": "Latest",
    "comfort_zone": "Comfort zone",          # also: comfort_label
    "season_winter": "Winter",               # also: season_labels
    "season_spring": "Spring",
    "season_summer": "Summer",
    "season_fall": "Fall",
}

_SEASON_KEYS = ("Winter", "Spring", "Summer", "Fall")


def _fill(template, **values):
    """``"{t} °C"`` + ``t=5`` → ``"5 °C"``; unknown placeholders are kept
    (same rule as ``hxFormat`` in d3-mollierhx)."""
    import re
    return re.sub(r"\{(\w+)\}",
                  lambda m: str(values[m.group(1)]) if m.group(1) in values
                  else m.group(0), str(template))


def _merge_texts(defaults, user, name):
    """``defaults`` updated with the non-``None`` values of ``user``; unknown
    keys raise ``ValueError`` (catches typos in translations)."""
    out = dict(defaults)
    if not user:
        return out
    unknown = sorted(set(user) - set(defaults))
    if unknown:
        raise ValueError(
            f"unknown {name} key(s) {unknown}; valid keys: {sorted(defaults)}")
    out.update({k: str(v) for k, v in user.items() if v is not None})
    return out


def _mollier_texts(captions, texts, x_axis_title, comfort_label,
                   season_labels):
    """Resolve the ``captions`` / ``texts`` arguments of the h,x plots.

    The older arguments map onto the same keys (``x_axis_title`` →
    ``captions["x_axis_title"]``, ``comfort_label`` → ``texts["comfort_zone"]``,
    ``season_labels`` → ``texts["season_*"]``); a key given in ``captions`` /
    ``texts`` takes precedence. Returns ``(captions, texts)``, both complete.
    """
    caps = dict(MOLLIER_DEFAULT_CAPTIONS)
    if x_axis_title is not None:
        caps["x_axis_title"] = str(x_axis_title)
    caps = _merge_texts(caps, captions, "captions")
    base = dict(MOLLIER_DEFAULT_TEXTS)
    if comfort_label is not None:
        base["comfort_zone"] = str(comfort_label)
    for k in _SEASON_KEYS:
        if season_labels and season_labels.get(k) is not None:
            base["season_" + k.lower()] = str(season_labels[k])
    return caps, _merge_texts(base, texts, "texts")


def _resolve_pressure(pressure, altitude):
    """Air pressure [Pa] for the Mollier plots.

    An explicit ``pressure`` wins (``altitude`` is then only the label).
    Without it, the pressure follows from ``altitude`` via the ISA formula
    (:func:`pyedautils.psychrometrics.pressure_from_altitude`), else sea
    level (101325 Pa).
    """
    if pressure is not None:
        return float(pressure)
    if altitude is not None:
        from pyedautils._mollier import pressure_from_altitude
        return float(pressure_from_altitude(altitude))
    return 101325.0


def _altitude_pressure_labels(pressure, altitude, texts=None):
    """Top-right annotation strings ``(altitude, pressure)`` from the
    ``altitude`` / ``pressure`` templates of ``texts``.

    Pressure is shown in hPa. Altitude defaults to the value derived from the
    pressure via the inverse ISA barometric formula when not supplied.
    """
    texts = texts or MOLLIER_DEFAULT_TEXTS
    if altitude is None:
        altitude = (1 - (pressure / 101325.0) ** (1 / 5.25588)) / 2.25577e-5
    return (_fill(texts["altitude"], alt=f"{altitude:.0f}"),
            _fill(texts["pressure"], hpa=f"{pressure / 100.0:.0f}"))


def _js(obj):
    """JSON for inlining into a ``<script>`` (UTF-8 kept, ``</`` escaped so a
    user text cannot close the script element)."""
    import json
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _point_arg(value, name):
    """``(t [°C], rh [%])`` → ``{"t": t, "rh": rh}`` for the JS overlay
    helpers; ``None`` stays ``None``."""
    import math as _m
    if value is None:
        return None
    try:
        t, rh = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a (temperature °C, rel. humidity %) pair, "
            f"got {value!r}") from None
    if not (_m.isfinite(t) and _m.isfinite(rh)) or not 0 <= rh <= 100:
        raise ValueError(
            f"{name} needs a finite temperature and a relative humidity "
            f"between 0 and 100 %, got {value!r}")
    return {"t": t, "rh": rh}


#: Upper absolute-humidity bound [kg/kg] meaning "no upper limit" (far beyond
#: any saturation state on the chart, so the zone follows its T/φ bounds).
_OPEN_X_MAX = 1.0
_ZONE_DEFAULTS = {"temperature": (20, 26), "rel_humidity": (0.30, 0.65),
                  "abs_humidity": (0, 0.0115)}
# Legacy defaults of a band given with absolute ranges only: abs humidity
# wide open so the band is bounded by T/φ only.
_BAND_DEFAULTS = {"temperature": (20, 26), "rel_humidity": (0.30, 0.65),
                  "abs_humidity": (0, 0.030)}
_ZONE_KEYS = (("temperature", "t"), ("rel_humidity", "phi"),
              ("abs_humidity", "x"))


def _is_offset(v):
    """A scalar (not a (min, max) pair, not a bool) is a ± offset."""
    import numbers
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _abs_range(v):
    """Absolute-humidity (min, max) [kg/kg]; ``None`` (whole range or upper
    bound) means no upper limit, a ``None`` lower bound means 0."""
    if v is None:
        return [0, _OPEN_X_MAX]
    lo, hi = v
    return [0 if lo is None else lo, _OPEN_X_MAX if hi is None else hi]


def _comfort_ranges(zone):
    """Absolute ``{"t", "phi", "x"}`` ranges of a comfort-zone dict (``None``
    → defaults). ``"abs_humidity": None`` → no upper abs-humidity limit; a
    missing ``"abs_humidity"`` key keeps the default ``(0, 0.0115)``."""
    zone = zone or {}
    out = {}
    for key, short in _ZONE_KEYS:
        v = zone.get(key, _ZONE_DEFAULTS[key])
        out[short] = _abs_range(v) if short == "x" else list(v)
    return out


def _resolve_bands(orange, red, cz):
    """Resolve the orange / red warning bands to absolute ranges.

    Each band is falsy (not drawn), ``True`` (default offsets from
    :func:`pyedautils.comfort.comfort_warning_bands`) or a dict whose
    ``"temperature"`` / ``"rel_humidity"`` / ``"abs_humidity"`` values are
    either a ``(min, max)`` pair (absolute range) or a scalar (``±`` offset
    around the comfort-zone ranges ``cz`` in K, fraction, kg/kg). A dict with
    at least one offset fills its missing keys with the default offsets of
    that colour; a dict without any offset keeps the legacy absolute defaults.
    ``"abs_humidity": None`` means no upper limit. Where the red band is given
    as offsets it is widened to enclose the orange band.

    Returns ``(orange, red)``, each ``None`` or a dict with ``"t"``, ``"phi"``,
    ``"x"`` (absolute ranges), ``"color"`` and ``"label"``.
    """
    from pyedautils.comfort import comfort_warning_bands
    stage_defaults = comfort_warning_bands()

    def one(band, which, color):
        if not band:
            return None, set()
        if band is True:
            band, offset_mode = {}, True
        else:
            offset_mode = any(_is_offset(band.get(k)) for k, _ in _ZONE_KEYS)
        out, offset_keys = {}, set()
        for key, short in _ZONE_KEYS:
            if key in band:
                v = band[key]
            elif offset_mode:
                v = stage_defaults[which][key]
            else:
                v = _BAND_DEFAULTS[key]
            if _is_offset(v):
                if cz is None:
                    raise ValueError(
                        f"comfort_zone_{which}: offsets need a comfort zone "
                        "(got comfort_zone=False); pass absolute (min, max) "
                        "ranges instead")
                d = abs(float(v))
                # rounded to drop float noise (0.65 + 0.05 -> 0.7, not 0.70..01)
                lo = round(cz[short][0] - d, 9)
                hi = round(cz[short][1] + d, 9)
                if short == "phi":
                    lo, hi = max(0.0, lo), min(1.0, hi)
                elif short == "x":
                    lo = max(0.0, lo)
                    hi = _OPEN_X_MAX if cz["x"][1] >= _OPEN_X_MAX else hi
                out[short] = [lo, hi]
                offset_keys.add(short)
            elif short == "x":
                out[short] = _abs_range(v)
            else:
                out[short] = list(v)
        out["color"] = band.get("color", color)
        out["label"] = band.get("label") or None
        return out, offset_keys

    o, _ = one(orange, "orange", "#E67E22")
    r, r_off = one(red, "red", "#C0392B")
    if o is not None and r is not None:
        for short in r_off:      # red encloses orange (stages must not overlap)
            r[short] = [min(r[short][0], o[short][0]),
                        max(r[short][1], o[short][1])]
    return o, r


def _band_js(band):
    """Build (t, phi, x, has, color, label) JSON snippets for a dashed band.

    ``band`` is one resolved band of :func:`_resolve_bands` (absolute ranges)
    or ``None``, which returns inert ``"false"`` snippets.
    """
    import json
    if not band:
        return ("[0,0]", "[0,0]", "[0,0]", "false", json.dumps(None), "null")
    return (json.dumps(band["t"]), json.dumps(band["phi"]),
            json.dumps(band["x"]), "true", json.dumps(band["color"]),
            json.dumps(band["label"]) if band["label"] else "null")


# Inline JS for the cumulative-frequency contour lines ("risk hours/days"), ported
# from the d3-mollierhx demo. Kept as a plain string (not an f-string) so the JS
# braces stay literal; the handful of injected values use __TOKEN__ placeholders
# substituted in plot_mollier_hx. It reuses the page's `mollier`, `x`, `y`, `plot`,
# `svg`, `width`, `height`, `domainY`, `p` and the `dataRecords` array.
_FREQ_JS_TEMPLATE = r"""
  // ----- Cumulative-frequency contour lines -------------------------------
  let dataXY = dataRecords || [];
  let freqShow = __FREQ_SHOW__;
  let freqLines = plot.append("g");
  let freqHover = plot.append("g");
  let freqLabels = plot.append("g").style("pointer-events", "none");
  let freqTip = d3.select("#__TOOLTIP_ID__");

  const FREQ_XMIN = 0, FREQ_DX = 0.0005, FREQ_NX = 60;
  const FREQ_TMIN = -30, FREQ_DT = 0.5, FREQ_NY = 180;
  const FREQ_MIN_CELL_HOURS = 1;
  const FREQ_SIGMA = 2.0, FREQ_FOOT_R = 0, FREQ_FOOT_SIGMA = 0.7;
  const MIN_RING_AREA = 2.0;
  const FREQ_TARGETS_HOURS = [1, 5, 25, 100, 200, 400, 800, 1600];
  const FREQ_TARGETS_DAYS = [1, 5, 10, 25, 50, 100, 200, 300];

  function freqDays() { return __FREQ_DAYS__; }
  function freqSmooth() { return __FREQ_SMOOTH__; }
  function dayKey(d) { return d.ts ? d.ts.slice(0, 10) : 0; }

  function gaussBlur2D(grid, nx, ny, sigma) {
    let r = Math.max(1, Math.ceil(3 * sigma));
    let ker = new Array(2 * r + 1), ksum = 0;
    for (let k = -r; k <= r; k++) { let w = Math.exp(-(k * k) / (2 * sigma * sigma)); ker[k + r] = w; ksum += w; }
    for (let k = 0; k < ker.length; k++) ker[k] /= ksum;
    let tmp = new Array(nx * ny).fill(0), out = new Array(nx * ny).fill(0);
    for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
      let s = 0;
      for (let k = -r; k <= r; k++) { let j = Math.min(nx - 1, Math.max(0, ix + k)); s += grid[iy * nx + j] * ker[k + r]; }
      tmp[iy * nx + ix] = s;
    }
    for (let ix = 0; ix < nx; ix++) for (let iy = 0; iy < ny; iy++) {
      let s = 0;
      for (let k = -r; k <= r; k++) { let j = Math.min(ny - 1, Math.max(0, iy + k)); s += tmp[j * nx + ix] * ker[k + r]; }
      out[iy * nx + ix] = s;
    }
    return out;
  }

  function softFootprint(raw, nx, ny, R, sigma) {
    let tmp = new Array(nx * ny).fill(0);
    for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
      let m = 0;
      for (let k = -R; k <= R; k++) { let j = ix + k; if (j < 0 || j >= nx) continue; if (raw[iy * nx + j] > 0) { m = 1; break; } }
      tmp[iy * nx + ix] = m;
    }
    let foot = new Array(nx * ny).fill(0);
    for (let ix = 0; ix < nx; ix++) for (let iy = 0; iy < ny; iy++) {
      let m = 0;
      for (let k = -R; k <= R; k++) { let j = iy + k; if (j < 0 || j >= ny) continue; if (tmp[j * nx + ix] > 0) { m = 1; break; } }
      foot[iy * nx + ix] = m;
    }
    return gaussBlur2D(foot, nx, ny, sigma);
  }

  function ringArea(ring) {
    let n = ring.length;
    if (n > 1 && ring[0][0] === ring[n - 1][0] && ring[0][1] === ring[n - 1][1]) n--;
    let a = 0;
    for (let i = 0; i < n; i++) { let q = ring[(i + 1) % n]; a += ring[i][0] * q[1] - q[0] * ring[i][1]; }
    return a / 2;
  }

  function smoothRing(ring, iters) {
    let pts = ring;
    if (pts.length > 1 && pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1]) pts = pts.slice(0, -1);
    let n = pts.length;
    if (n < 4 || iters <= 0) { pts = pts.slice(); pts.push([pts[0][0], pts[0][1]]); return pts; }
    for (let it = 0; it < iters; it++) {
      let out = new Array(n);
      for (let i = 0; i < n; i++) {
        let a = pts[(i - 1 + n) % n], b = pts[i], c = pts[(i + 1) % n];
        out[i] = [(a[0] + 2 * b[0] + c[0]) / 4, (a[1] + 2 * b[1] + c[1]) / 4];
      }
      pts = out;
    }
    pts.push([pts[0][0], pts[0][1]]);
    return pts;
  }

  function topPointOfContour(c) {
    let best = null;
    c.coordinates.forEach(function (poly) { poly.forEach(function (ring) { ring.forEach(function (pt) { if (best === null || pt[1] > best[1]) best = pt; }); }); });
    return best;
  }

  let freqCache = null;
  function buildFreqContours(useDays) {
    if (freqCache && freqCache.ref === dataXY && freqCache.useDays === useDays) return freqCache.levels;
    let nx = FREQ_NX, ny = FREQ_NY;
    let empty = { ref: dataXY, useDays: useDays, levels: [] };
    let raw0 = new Array(nx * ny).fill(0), recs = [];
    dataXY.forEach(function (d) {
      let t = mollier.temperature(d.x, d.y);
      let ix = Math.floor((d.x - FREQ_XMIN) / FREQ_DX), iy = Math.floor((t - FREQ_TMIN) / FREQ_DT);
      if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) return;
      let cell = iy * nx + ix;
      raw0[cell] += 1;
      recs.push({ cell: cell, dk: useDays ? dayKey(d) : 0 });
    });
    let rawGrid = new Array(nx * ny).fill(0);
    let dayGrid = useDays ? new Array(nx * ny) : null;
    let allDays = useDays ? new Set() : null;
    recs.forEach(function (r) {
      if (raw0[r.cell] < FREQ_MIN_CELL_HOURS) return;
      rawGrid[r.cell] += 1;
      if (useDays) { (dayGrid[r.cell] || (dayGrid[r.cell] = [])).push(r.dk); allDays.add(r.dk); }
    });
    let totalH = d3.sum(rawGrid);
    if (totalH < 3) { freqCache = empty; return empty.levels; }
    let densGrid = gaussBlur2D(rawGrid, nx, ny, FREQ_SIGMA);
    let foot = softFootprint(rawGrid, nx, ny, FREQ_FOOT_R, FREQ_FOOT_SIGMA);
    for (let i = 0; i < densGrid.length; i++) densGrid[i] *= foot[i];
    let total = useDays ? allDays.size : totalH;
    let targets = (useDays ? FREQ_TARGETS_DAYS : FREQ_TARGETS_HOURS).filter(function (t) { return t < total; });
    if (!targets.length) { freqCache = empty; return empty.levels; }
    let cells = [];
    for (let i = 0; i < densGrid.length; i++) cells.push({ key: densGrid[i] + i * 1e-9, h: rawGrid[i], days: useDays ? (dayGrid[i] || null) : null });
    cells.sort(function (a, b) { return a.key - b.key; });
    let thr = [], ti = 0, cum = 0, union = useDays ? new Set() : null;
    for (let k = 0; k < cells.length && ti < targets.length; k++) {
      if (useDays) { let ds = cells[k].days; if (ds) for (let q = 0; q < ds.length; q++) union.add(ds[q]); cum = union.size; }
      else { cum += cells[k].h; }
      while (ti < targets.length && cum >= targets[ti]) {
        let dv = cells[k].key;
        if (thr.length === 0 || dv > thr[thr.length - 1].value) thr.push({ value: dv, label: targets[ti] });
        else thr[thr.length - 1].label = targets[ti];
        ti++;
      }
    }
    let raw = d3.contours().size([nx, ny]).thresholds(thr.map(function (o) { return o.value; }))(densGrid);
    let levels = raw.map(function (c, idx) {
      let rings = [];
      c.coordinates.forEach(function (poly) { poly.forEach(function (ring) { if (Math.abs(ringArea(ring)) >= MIN_RING_AREA) rings.push(ring); }); });
      return { label: thr[idx] ? thr[idx].label : Math.round(c.value), rings: rings };
    });
    freqCache = { ref: dataXY, useDays: useDays, levels: levels };
    return levels;
  }

  function updateSaturationClip() {
    let sat = [];
    let t0 = domainY[0] - 3, t1 = domainY[1] + 3, ns = 120;
    for (let i = 0; i <= ns; i++) { let t = t0 + (t1 - t0) * i / ns; let s = mollier.get_x_y(t, 1, p); sat.push([x(s.x), y(s.y)]); }
    let pad = Math.max(width, height) + 200;
    let pts = [[-pad, height + pad]].concat(sat).concat([[width + pad, -pad], [-pad, -pad]]);
    let cp = svg.select("#__SATCLIP_ID__");
    if (cp.empty()) cp = svg.append("clipPath").attr("id", "__SATCLIP_ID__");
    cp.selectAll("polygon").remove();
    cp.append("polygon").attr("points", pts.map(function (q) { return q.join(","); }).join(" "));
  }

  function drawFreqLines() {
    freqLines.selectAll("*").remove();
    freqLabels.selectAll("*").remove();
    freqHover.selectAll("*").remove();
    if (!freqShow || dataXY.length === 0) return;
    let useDays = freqDays();
    let levels = buildFreqContours(useDays);
    if (!levels.length) return;
    let iters = Math.max(0, Math.min(60, Math.round((freqSmooth() - 0.6) * 4)));
    let contours = levels.map(function (L) {
      let rings = L.rings.map(function (r) { return smoothRing(r, iters); });
      return { c: { type: "MultiPolygon", coordinates: [rings] }, label: L.label };
    });
    function gridToScreen(gx, gy) {
      let xv = FREQ_XMIN + gx * FREQ_DX, Tv = FREQ_TMIN + gy * FREQ_DT;
      let yv = mollier.get_x_y_tx(Tv, xv, p).y;
      return [x(xv), y(yv)];
    }
    let smoothLine = d3.line().curve(d3.curveBasisClosed);
    function contourPath(mp) {
      let dstr = "";
      mp.coordinates.forEach(function (poly) {
        poly.forEach(function (ring) {
          let r = ring;
          if (r.length > 1 && r[0][0] === r[r.length - 1][0] && r[0][1] === r[r.length - 1][1]) r = r.slice(0, -1);
          if (r.length < 3) return;
          let s = smoothLine(r.map(function (pp) { return gridToScreen(pp[0], pp[1]); }));
          if (s) dstr += s + " ";
        });
      });
      return dstr;
    }
    updateSaturationClip();
    let pathsG = freqLines.append("g").attr("clip-path", "url(#__SATCLIP_ID__)");
    pathsG.selectAll("path").data(contours).enter().append("path")
      .attr("d", function (d) { return contourPath(d.c); })
      .attr("fill", "none").attr("stroke", "#333").attr("stroke-width", 1.6).attr("opacity", 1);
    let unitTxt = useDays ? __FREQ_DAYS_TEXT__ : __FREQ_HOURS_TEXT__;
    let hitG = freqHover.append("g").attr("clip-path", "url(#__SATCLIP_ID__)");
    hitG.selectAll("path").data(contours).enter().append("path")
      .attr("d", function (d) { return contourPath(d.c); })
      .attr("fill", "none").attr("stroke", "transparent").attr("stroke-width", 8)
      .style("pointer-events", "stroke").style("cursor", "crosshair")
      .on("mouseover mousemove", function (event, d) {
        freqTip.style("opacity", 1).style("background-color", "white").style("color", "black")
          .text(hxFormat(unitTxt, { n: d.label }))
          .style("left", (event.pageX + 15) + "px").style("top", (event.pageY - 40) + "px");
      })
      .on("mouseout", function () { freqTip.style("opacity", 0); });
    let labelG = freqLabels.append("g").attr("font-family", "helvetica").attr("font-size", 13).attr("font-weight", "bold");
    contours.forEach(function (d) {
      let pt = topPointOfContour(d.c);
      if (!pt) return;
      let s = gridToScreen(pt[0], pt[1]);
      let g = labelG.append("g").attr("transform", "translate(" + s[0] + "," + s[1] + ")");
      let txt = g.append("text").attr("text-anchor", "middle").attr("dy", "0.32em").attr("fill", "#111").text(d.label);
      let bb = txt.node().getBBox();
      g.insert("rect", "text").attr("x", bb.x - 3).attr("y", bb.y - 1)
        .attr("width", bb.width + 6).attr("height", bb.height + 2)
        .attr("rx", 2).attr("fill", "white").attr("opacity", 0.92).attr("stroke", "#ddd").attr("stroke-width", 0.5);
    });
  }

  if (freqShow) drawFreqLines();
"""


def plot_mollier_hx(
    data: Optional[pd.DataFrame] = None,
    pressure: Optional[float] = None,
    altitude: Optional[float] = None,
    domain_x: Tuple[float, float] = (0.0, 0.020),
    domain_y: Tuple[float, float] = (-20.0, 50.0),
    comfort_zone: Optional[Dict[str, Tuple[float, float]]] = None,
    height: int = 700,
    convention: str = 'classical',
    highlight_latest: bool = True,
    highlight_color: Optional[str] = 'black',
    states: Optional[Sequence] = None,
    labels: Optional[List[str]] = None,
    comfort_label: str = 'Comfort zone',
    comfort_zone_orange: Optional[Dict[str, object]] = None,
    comfort_zone_red: Optional[Dict[str, object]] = None,
    show_temperature: bool = True,
    show_density: bool = True,
    show_rel_humidity: bool = True,
    show_enthalpy: bool = True,
    show_abs_humidity: bool = True,
    x_axis_title: str = 'absolute water content x [g/kg]',
    season_labels: Optional[Dict[str, str]] = None,
    show_frequency: bool = False,
    frequency_unit: str = 'hours',
    frequency_smoothing: float = 4.0,
    outdoor: Optional[Tuple[float, float]] = None,
    current: Optional[Tuple[float, float]] = None,
    outdoor_color: str = '#2563eb',
    ventilation_line: bool = True,
    ventilation_note: bool = True,
    ventilation_threshold: float = 0.0003,
    captions: Optional[Dict[str, str]] = None,
    texts: Optional[Dict[str, str]] = None,
) -> str:
    """
    Create a Mollier h,x-diagram (psychrometric chart) as self-contained HTML.

    Uses D3.js (v7) for fast SVG rendering with iso-lines for temperature,
    enthalpy, relative humidity and density, a comfort zone, and optional
    measured data points colour-coded by season with interactive hover
    tooltips. Optionally shows the outside air with the ventilation (mixing)
    line from the room point and a note on whether airing dries the room.

    Data points, comfort zone, warning bands and frequency lines are clipped
    to the axes. The room point and the outside point are drawn on top; a
    point outside the axes becomes an arrow on the edge pointing towards it,
    labelled with its value (e.g. ``6.8 °C · 89 %``).

    Every visible text can be replaced through ``captions`` (axis captions)
    and ``texts`` (everything else); the defaults are English. See
    :data:`MOLLIER_DEFAULT_CAPTIONS` and :data:`MOLLIER_DEFAULT_TEXTS` for
    the keys.

    Args:
        data: Optional DataFrame with columns [timestamp, humidity, temperature].
            humidity in %, temperature in °C.
        pressure: Air pressure in Pa used for all psychrometric relations
            and shown in the top-right corner in hPa. When ``None``
            (default), it is derived from ``altitude`` via the ISA
            barometric formula
            (:func:`pyedautils.psychrometrics.pressure_from_altitude`), or
            101325 Pa (sea level) without an altitude. An explicit pressure
            always wins; ``altitude`` is then only the label.
        altitude: Optional altitude in metres above sea level. Sets the
            pressure when ``pressure`` is not given (``altitude=450`` gives
            960 hPa) and is shown together with the pressure in the
            top-right corner. When ``None``, the shown altitude is derived
            from ``pressure`` via the inverse ISA barometric formula.
        domain_x: Range of absolute humidity [kg/kg] for the x-axis.
        domain_y: Range of the y-coordinate (≈ temperature at x=0) for the y-axis.
        comfort_zone: Dict with keys "temperature", "rel_humidity", "abs_humidity",
            each a (min, max) tuple. Defaults: T=[20, 26], phi=[0.30, 0.65],
            x=[0, 0.0115]. ``"abs_humidity": None`` (or a ``None`` upper
            bound, e.g. ``(0, None)``) means no abs-humidity limit — the zone
            is bounded by T and φ only, like ``hum_abs_band=None`` of
            :func:`pyedautils.comfort.comfort_compass_distribution`; a missing
            key keeps the default 0.0115 kg/kg cap. Pass ``False`` to disable
            the comfort zone.
        height: Diagram height in pixels. Default 700.
        convention: Mollier coordinate convention. ``'classical'`` (default)
            normalises enthalpy per kg of dry air (Recknagel/Sprenger style,
            isotherms tilt slightly up with x). ``'glueck'`` normalises per
            kg of moist air (per the Glück book, isotherms tilt slightly down).
        highlight_latest: When ``True`` (default), the row with the most recent
            timestamp is overlaid as a circle on top of the seasonal scatter
            points (the *room point*, also the start of the ventilation line).
            Set to ``False`` to disable. Ignored when ``current`` is given.
        highlight_color: CSS colour of the room point (latest row or
            ``current``). Default ``'black'``. Pass ``None`` to use the latest
            row's season colour (so the highlight is visible only by
            size/order, not by colour; black for ``current``).
        states: Optional sequence of ``MoistAirState`` objects (from
            :func:`pyedautils.psychrometrics.state`). When supplied, each state is
            drawn as a numbered point and consecutive states are connected by
            arrows — a psychrometric process chain in the spirit of
            psychrosim.com. Works alongside ``data`` (historical scatter is
            drawn underneath).
        labels: Optional sequence of strings used as point labels (default
            is the number ``"1"``, ``"2"``, …). Length must match ``states``.
        comfort_label: Legend caption for the comfort-zone polygon. Default
            ``"Comfort zone"``. Override e.g. to ``"Komfortzone"`` for German
            output or to a custom designation like ``"DIN EN 16798 Cat II"``.
            Same as ``texts["comfort_zone"]`` (which wins if both are given).
        comfort_zone_orange: Optional warning band drawn as a **dashed orange
            outline** (no fill). Off by default (``None``). A dict with keys
            ``"temperature"``, ``"rel_humidity"``, ``"abs_humidity"``, each
            either

            * a ``(min, max)`` tuple — **absolute** bounds in °C, 0–1 and
              kg/kg, e.g. ``{"temperature": (19, 27),
              "rel_humidity": (0.25, 0.70)}``; or
            * a scalar — a **± offset** around the ``comfort_zone`` bounds in
              K, rel.-humidity fraction and kg/kg, e.g.
              ``{"temperature": 1, "rel_humidity": 0.05,
              "abs_humidity": 0.001}`` for ±1 K / ±5 %-points / ±1 g/kg.

            Both forms may be mixed per key. A dict without any offset keeps
            the absolute defaults T=[20, 26], φ=[0.30, 0.65] for missing keys
            and leaves the abs. humidity unclipped (the band follows its T/φ
            bounds). A dict with at least one offset fills missing keys with
            the default offsets of the band (orange: 1 K / 0.05 / 0.001 kg/kg,
            red: 2.5 K / 0.10 / 0.0025 kg/kg). ``True`` uses all default
            offsets. ``"abs_humidity": None`` means no upper limit (as does an
            offset band around a comfort zone without abs-humidity limit).
            Optional ``"label"`` (legend caption) and ``"color"`` (CSS
            colour). The default offsets are the severity-stage limits of
            :func:`pyedautils.comfort.comfort_compass_distribution`; use
            :func:`pyedautils.comfort.comfort_warning_bands` to derive both
            bands from the same stage limits as the comfort compass. Offsets
            need a comfort zone (``ValueError`` with ``comfort_zone=False``).
        comfort_zone_red: Like ``comfort_zone_orange`` but drawn as a **dashed
            red outline** — e.g. the ±2.5 K / ±10 % band. Keys given as
            offsets are widened where needed to enclose the orange band, so
            the stages never overlap.
        show_temperature: Draw the iso-temperature lines. Default ``True``.
            The temperature y-axis itself is always shown (it is the chart's
            coordinate base) — this only toggles the blue iso-lines.
        show_density: Draw the iso-density curves together with their value
            labels and the ``"Density ρ [kg/m³]"`` caption. Default
            ``True``. Set ``False`` to hide everything density-related.
        show_rel_humidity: Draw the relative-humidity (φ) curves and their
            value labels. Default ``True``. The 100 % saturation curve (the
            chart boundary) is always drawn regardless of this flag.
        show_enthalpy: Draw the iso-enthalpy lines together with their value
            labels and the ``"Enthalpy [kJ/kg]"`` caption. Default
            ``True``. Set ``False`` to hide everything enthalpy-related.
        show_abs_humidity: Draw the vertical iso-lines of constant absolute
            humidity (at the bottom-axis tick values). Default ``True``.
        x_axis_title: Caption centred below the bottom axis. Default
            ``"absolute water content x [g/kg]"``. Pass ``""`` to omit it.
            Same as ``captions["x_axis_title"]`` (which wins if both are
            given).
        season_labels: Optional mapping from the English season keys
            ``"Winter"``, ``"Spring"``, ``"Summer"``, ``"Fall"`` to the labels
            shown in the seasonal scatter tooltips and legend. Defaults to the
            English keys themselves. Pass
            ``pyedautils.plots._constants._SEASON_LABELS_DE`` (or any custom
            dict) for German output. Same as ``texts["season_winter"]`` etc.
            (which win if both are given).
        show_frequency: Draw cumulative-frequency contour lines ("risk
            hours/days") over the scatter. Each line labelled N encloses all
            but N units of the year, so the outer lines mark rare conditions
            and the inner lines the typical core. Requires ``data``. Default
            ``False``.
        frequency_unit: Unit counted by the frequency lines — ``"hours"``
            (default; line N = N hours/year outside) or ``"days"`` (line N =
            N distinct calendar days/year with at least one hour outside).
        frequency_smoothing: Line-rounding amount for the frequency lines,
            ~0.6 (angular) to ~9 (very round). Default ``4.0``. Purely
            geometric — it rounds the curves but never enlarges the enclosed
            region.
        outdoor: Optional outside air as ``(temperature [°C], relative
            humidity [%])`` — the same units as the ``data`` columns, e.g.
            ``(4.0, 85)``. Drawn as a hollow circle (an edge arrow when outside
            the axes). With a room point (``current`` or the latest row) a
            dashed **ventilation line** joins the two: airing moves the room
            air along it. A note above the axes says whether airing dries the
            room (outside air holds more than ``ventilation_threshold`` less
            water), adds moisture (more), or makes no noticeable change — see
            :func:`pyedautils.comfort.ventilation_effect`.
        current: Optional room point as ``(temperature [°C], relative
            humidity [%])``, e.g. a live reading. Replaces the latest-row
            highlight as the room point; works without ``data``.
        outdoor_color: CSS colour of the outside point, the ventilation line
            and the note. Default ``'#2563eb'`` (blue).
        ventilation_line: Draw the ventilation line (and the note) between
            the room point and ``outdoor``. Default ``True``.
        ventilation_note: Show the ventilation note above the axes. Default
            ``True``; needs the ventilation line.
        ventilation_threshold: Difference in absolute humidity [kg/kg] up to
            which airing counts as "no change". Default ``0.0003`` (0.3 g/kg).
        captions: Optional dict overriding the axis captions, merged over
            :data:`MOLLIER_DEFAULT_CAPTIONS` (keys ``temperature``,
            ``rel_humidity``, ``density``, ``enthalpy``, ``x_axis``,
            ``x_axis_title``). ``""`` hides a caption. The right-edge captions
            shrink (8–12 px) to fit the chart height. Unknown keys raise
            ``ValueError``.
        texts: Optional dict overriding any other visible text, merged over
            :data:`MOLLIER_DEFAULT_TEXTS`: outside air and ventilation note,
            edge-arrow value label, tooltips, frequency-line tooltip,
            altitude / pressure line, legend (comfort zone, seasons, room
            point). ``""`` hides the altitude / pressure line or the note.
            Unknown keys raise ``ValueError``. Example (German)::

                texts={"outdoor": "Aussenluft", "room": "Raum",
                       "ventilation_dries": "Lüften trocknet",
                       "ventilation_humidifies": "Lüften befeuchtet",
                       "ventilation_neutral": "Lüften: keine Änderung",
                       "altitude": "{alt} m ü. M.", "latest": "Letzter Wert"}

    Returns:
        str: Self-contained HTML string with inline D3.js rendering.
            Can be saved to a file, used with ``streamlit.components.v1.html()``,
            or displayed in a Jupyter notebook via ``IPython.display.HTML()``.

    See Also:
        How to read the h,x diagram correctly, what the comfort zone and
        warning bands mean, and how to save energy:
        https://retomarek.github.io/pyedautils/guides/mollier_hx.html
    """
    import json

    from pyedautils._mollier import (
        _check_convention,
        get_x_y,
        rel_humidity as m_rel_humidity,
        temperature as m_temperature,
    )

    convention = _check_convention(convention)
    pressure = _resolve_pressure(pressure, altitude)
    js = _load_d3_js()
    caps, txt = _mollier_texts(captions, texts, x_axis_title, comfort_label,
                               season_labels)
    outdoor_pt = _point_arg(outdoor, "outdoor")
    current_pt = _point_arg(current, "current")

    # Season display labels (English keys -> shown label).
    season_labels = {k: txt["season_" + k.lower()] for k in _SEASON_KEYS}

    # Prepare data JSON
    data_json = "null"
    current_json = "null"
    latest = None
    if data is not None and not data.empty:
        df = data.copy()
        df.columns = ["timestamp", "humidity", "temperature"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["humidity", "temperature"])
        if not df.empty:
            t_arr = df["temperature"].values
            phi_arr = df["humidity"].values / 100.0
            x_arr, y_arr = get_x_y(t_arr, phi_arr, pressure, convention=convention)
            df["season"] = df["timestamp"].apply(_get_season_fast)
            records = []
            for i in range(len(df)):
                ts = df.iloc[i]["timestamp"]
                xv, yv = float(x_arr[i]), float(y_arr[i])
                records.append({
                    "x": xv, "y": yv,
                    "season": season_labels.get(df.iloc[i]["season"], "?"),
                    "ts": ts.strftime("%Y-%m-%d %H:%M"),
                    "temp": round(float(m_temperature(xv, yv, convention=convention)), 2),
                    "phi": round(float(m_rel_humidity(
                        xv, yv, pressure, convention=convention) * 100), 2),
                    "xg": round(xv * 1000, 2),
                })
            data_json = json.dumps(records)
            if highlight_latest and current_pt is None:
                # Find the record with the most recent timestamp (don't assume
                # the input DataFrame is sorted).
                idx_latest = df["timestamp"].values.argmax()
                latest = records[idx_latest]
                current_json = json.dumps(latest)

    # Room point (start of the ventilation line): an explicit ``current``
    # reading, else the highlighted latest row.
    room_json = "null"
    if current_pt is not None:
        room_json = json.dumps({
            "state": current_pt, "title": txt["now"], "ts": None,
            "color": highlight_color or "black", "legend": txt["room"]})
    elif latest is not None:
        season_key = {v: k for k, v in season_labels.items()}.get(latest["season"])
        room_json = json.dumps({
            "state": {"x": latest["x"], "y": latest["y"]},
            "title": txt["latest"], "ts": latest["ts"],
            "color": highlight_color or DEFAULT_SEASON_COLORS.get(season_key, "#999"),
            "legend": txt["latest"]})
    outdoor_json = json.dumps(outdoor_pt)
    outdoor_opts_js = json.dumps({
        "color": outdoor_color, "ventilationLine": bool(ventilation_line),
        "note": bool(ventilation_note),
        "threshold": abs(float(ventilation_threshold)),
    })
    texts_js = _js(txt)

    # Process-chain points. Each record carries both a short integer ``number``
    # (1-based, drawn inside the chart circle) and a free-form ``label``
    # (shown in the legend). The legend is rendered only when the user
    # supplied explicit labels — otherwise the labels would just repeat the
    # numbers.
    states_json = "null"
    state_legend_js = "false"
    if states is not None and len(states) > 0:
        if labels is not None and len(labels) != len(states):
            raise ValueError(
                f"labels length ({len(labels)}) must match states length "
                f"({len(states)})"
            )
        state_records = []
        for i, s in enumerate(states):
            number = str(i + 1)
            lab = labels[i] if labels is not None else number
            state_records.append({
                "x": float(s.x),
                "y": float(s.y),
                "number": number,
                "label": str(lab),
                "t": round(float(s.t), 2),
                "phi": round(float(s.phi) * 100, 2),
                "xg": round(float(s.x) * 1000, 2),
                "h": round(float(s.h), 2),
                "t_wb": round(float(s.t_wb), 2),
                "t_dp": round(float(s.t_dp), 2),
            })
        states_json = json.dumps(state_records)
        state_legend_js = "true" if labels is not None else "false"

    if comfort_zone is False:
        comfort_t, comfort_phi, comfort_x = "[0,0]", "[0,0]", "[0,0]"
        has_comfort_js = "false"
    else:
        cz = _comfort_ranges(comfort_zone)
        comfort_t = json.dumps(cz["t"])
        comfort_phi = json.dumps(cz["phi"])
        comfort_x = json.dumps(cz["x"])
        has_comfort_js = "true"
    band_o, band_r = _resolve_bands(
        comfort_zone_orange, comfort_zone_red,
        None if comfort_zone is False else cz)
    (orange_t, orange_phi, orange_x, has_orange_js,
     orange_color_js, orange_label_js) = _band_js(band_o)
    (red_t, red_phi, red_x, has_red_js,
     red_color_js, red_label_js) = _band_js(band_r)

    # Altitude / pressure annotation (top-right).
    info_alt, info_pressure = _altitude_pressure_labels(pressure, altitude, txt)
    info_lines_js = _js([v for v in (info_alt, info_pressure) if v])

    domain_x_js = json.dumps(list(domain_x))
    domain_y_js = json.dumps(list(domain_y))
    highlight_color_js = json.dumps(highlight_color)
    comfort_label_js = _js(txt["comfort_zone"])
    hx_opts_js = _js({
        "showTemperature": bool(show_temperature),
        "showDensity": bool(show_density),
        "showRelHumidity": bool(show_rel_humidity),
        "showEnthalpy": bool(show_enthalpy),
        "showAbsHumidity": bool(show_abs_humidity),
        "captions": caps,
    })

    import uuid
    uid = uuid.uuid4().hex[:8]
    diagram_id = f"mollier_{uid}"
    tooltip_id = f"tt_{uid}"
    clip_id = f"clip_{uid}"

    # Frequency-line JS: substitute the injected values into the template.
    freq_unit = str(frequency_unit).lower()
    if freq_unit not in ("hours", "days"):
        raise ValueError("frequency_unit must be 'hours' or 'days'")
    freq_js = (
        _FREQ_JS_TEMPLATE
        .replace("__FREQ_SHOW__", "true" if show_frequency else "false")
        .replace("__FREQ_DAYS__", "true" if freq_unit == "days" else "false")
        .replace("__FREQ_SMOOTH__", repr(float(frequency_smoothing)))
        .replace("__FREQ_HOURS_TEXT__", _js(txt["freq_hours"]))
        .replace("__FREQ_DAYS_TEXT__", _js(txt["freq_days"]))
        .replace("__SATCLIP_ID__", f"freqSat_{uid}")
        .replace("__TOOLTIP_ID__", tooltip_id)
    )

    season_colors = _js(
        {season_labels[k]: c for k, c in DEFAULT_SEASON_COLORS.items()})
    # Legend entries (Spring, Summer, Fall, Winter) and the set of "dark"
    # season labels whose tooltip background needs white text.
    season_legend_js = _js([
        {"label": season_labels[k], "color": DEFAULT_SEASON_COLORS[k]}
        for k in ("Spring", "Summer", "Fall", "Winter")
    ])
    dark_labels_js = _js([season_labels["Winter"], season_labels["Fall"]])

    return f"""<div id="{diagram_id}" style="width:100%;background:white;"></div>
<div id="{tooltip_id}" style="position:absolute;background:rgba(255,255,255,0.9);\
border-radius:4px;padding:6px 8px;pointer-events:none;\
font-family:Tahoma,Geneva,sans-serif;font-size:11px;\
box-shadow:2px 2px 6px rgba(0,0,0,0.2);opacity:0;"></div>
<script src="{_D3_URL}"></script>
<script>
(function() {{
{js["mollier_functions"]}
{js["coordinate_generator"]}
{js["draw_comfort"]}
  let convention = "{convention}";
  let mollier = createMollier(convention);
  let p = {pressure};
  let domainX = {domain_x_js};
  let domainY = {domain_y_js};
  let rangeT = {comfort_t};
  let rangePhi = {comfort_phi};
  let rangeX = {comfort_x};
  let dataRecords = {data_json};
  let currentRecord = {current_json};
  let roomPoint = {room_json};
  let outdoorPoint = {outdoor_json};
  let outdoorOpts = {outdoor_opts_js};
  let T = {texts_js};
  let statePoints = {states_json};
  let showStateLegend = {state_legend_js};
  let hasComfort = {has_comfort_js};
  let comfortLabel = {comfort_label_js};
  let infoLines = {info_lines_js};
  let hasOrange = {has_orange_js};
  let orangeT = {orange_t}; let orangePhi = {orange_phi}; let orangeX = {orange_x};
  let orangeColor = {orange_color_js}; let orangeLabel = {orange_label_js};
  let hasRed = {has_red_js};
  let redT = {red_t}; let redPhi = {red_phi}; let redX = {red_x};
  let redColor = {red_color_js}; let redLabel = {red_label_js};
  let highlightColor = {highlight_color_js};
  let colorMap = {season_colors};
  let seasonLegend = {season_legend_js};
  let darkLabels = {dark_labels_js};
  let hxOpts = {hx_opts_js};

  let Height = {height};
  let container = document.getElementById("{diagram_id}");
  let Width = container.getBoundingClientRect().width || 900;

  let margin = {{top: 40, right: 70, bottom: 50, left: 60}};
  let width = Width - margin.left - margin.right;
  let height = Height - margin.top - margin.bottom;

  let svg = d3.select("#{diagram_id}").append("svg")
    .attr("display", "block").attr("width", Width).attr("height", Height);
  let bg = svg.append("g");
  let defs = svg.append("defs");
  // Data, comfort zone, bands and frequency lines are clipped to the axes.
  let clipUrl = hxAddClipPath(svg, "{clip_id}", width, height);
  // Arrow head for process-chain segments.
  defs.append("marker")
    .attr("id", "arrow-{uid}")
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 9).attr("refY", 5)
    .attr("markerWidth", 6).attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", "#444");
  let plot = svg.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    .attr("clip-path", clipUrl);

  drawHXCoordinates(bg, Width, Height, margin, domainX, domainY, p, mollier, hxOpts);

  // Altitude / pressure annotation, top-right (dark grey).
  let infoG = svg.append("g").attr("class", "hx-info")
    .style("font-family", "Tahoma, Geneva, sans-serif");
  infoLines.forEach(function(t, i) {{
    infoG.append("text")
      .attr("x", margin.left + width)
      .attr("y", margin.top - 8 - 14 * (infoLines.length - 1 - i))
      .attr("text-anchor", "end").attr("fill", "#4d4d4d").attr("font-size", 12)
      .text(t);
  }});

  let x = d3.scaleLinear().range([0, width]).domain(domainX);
  let y = d3.scaleLinear().range([height, 0]).domain(domainY);
  let ctx = {{x: x, y: y, width: width, height: height, mollier: mollier, pressure: p}};
  let tooltip = d3.select("#{tooltip_id}");
  function showTip(event, html, bgColor, fgColor) {{
    tooltip.style("opacity", 1).style("background-color", bgColor)
      .style("color", fgColor).html(html)
      .style("left", (event.pageX + 15) + "px")
      .style("top", (event.pageY - 40) + "px");
  }}
  function hideTip() {{ tooltip.style("opacity", 0); }}
  function esc(s) {{
    return String(s).replace(/[&<>"]/g, function(c) {{
      return {{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}}[c];
    }});
  }}

  let line = d3.line().x(d => x(d.x)).y(d => y(d.y));
  let pathos = hasComfort ? createComfort(rangeT, rangePhi, rangeX, p, mollier) : null;
  if (hasComfort && pathos && pathos.length > 0) {{
    plot.append("path").attr("class", "hx-comfort").datum(pathos).attr("d", line)
      .attr("fill", "yellowgreen").attr("fill-opacity", 0.4)
      .attr("stroke", "yellowgreen");
  }}

  // Extra warning bands: dashed outline only (no fill).
  function drawBand(rangeT, rangePhi, rangeX, color, cls) {{
    let pb = createComfort(rangeT, rangePhi, rangeX, p, mollier);
    if (pb && pb.length > 0) {{
      plot.append("path").attr("class", cls).datum(pb).attr("d", line)
        .attr("fill", "none").attr("stroke", color)
        .attr("stroke-width", 1.8).attr("stroke-dasharray", "7,4");
    }}
  }}
  if (hasOrange) drawBand(orangeT, orangePhi, orangeX, orangeColor, "hx-band-orange");
  if (hasRed) drawBand(redT, redPhi, redX, redColor, "hx-band-red");

  if (dataRecords && dataRecords.length > 0) {{
    for (let i = dataRecords.length - 1; i > 0; i--) {{
      let j = Math.floor(Math.random() * (i + 1));
      [dataRecords[i], dataRecords[j]] = [dataRecords[j], dataRecords[i]];
    }}

    plot.append("g").attr("class", "hx-scatter")
      .selectAll("circle").data(dataRecords).enter().append("circle")
      .attr("cx", d => x(d.x)).attr("cy", d => y(d.y))
      .attr("r", 5).attr("fill", d => colorMap[d.season] || "#999")
      .attr("opacity", 0.4).attr("shape-rendering", "optimizeSpeed")
      .on("mouseover", function(event, d) {{
        d3.select(this).attr("r", 10).attr("opacity", 0.9);
        showTip(event, esc(d.ts) + "<br>" + esc(T.tip_x) + ": " + d.xg + " g/kg<br>"
            + esc(T.tip_t) + ": " + d.temp + " °C<br>" + esc(T.tip_phi) + ": " + d.phi + " %",
          colorMap[d.season] || "#999",
          (darkLabels.indexOf(d.season) >= 0) ? "white" : "black");
      }})
      .on("mouseout", function() {{
        d3.select(this).attr("r", 5).attr("opacity", 0.4);
        hideTip();
      }});
  }}

  // Process chain: numbered state points joined by arrows.
  if (statePoints && statePoints.length > 0) {{
    let chain = plot.append("g").attr("id", "process-chain-{uid}")
      .attr("class", "process-chain");

    for (let i = 1; i < statePoints.length; i++) {{
      let a = statePoints[i - 1];
      let b = statePoints[i];
      chain.append("line")
        .attr("x1", x(a.x)).attr("y1", y(a.y))
        .attr("x2", x(b.x)).attr("y2", y(b.y))
        .attr("stroke", "#444").attr("stroke-width", 2)
        .attr("marker-end", "url(#arrow-{uid})");
    }}

    let nodes = chain.selectAll("g.state").data(statePoints).enter()
      .append("g").attr("class", "state")
      .attr("transform", d => "translate(" + x(d.x) + "," + y(d.y) + ")")
      .style("cursor", "pointer")
      .on("mouseover", function(event, d) {{
        showTip(event, esc(d.label)
            + "<br>" + esc(T.tip_t) + ": " + d.t + " °C"
            + "<br>" + esc(T.tip_phi) + ": " + d.phi + " %"
            + "<br>" + esc(T.tip_x) + ": " + d.xg + " g/kg"
            + "<br>" + esc(T.tip_h) + ": " + d.h + " kJ/kg", "#222", "white");
      }})
      .on("mouseout", hideTip);

    nodes.append("circle")
      .attr("r", 11).attr("fill", "#222")
      .attr("stroke", "white").attr("stroke-width", 2);
    nodes.append("text")
      .attr("text-anchor", "middle").attr("dy", "0.35em")
      .attr("fill", "white").attr("font-size", "11px")
      .attr("font-weight", "bold")
      .text(d => d.number);
  }}
{freq_js}
  // ----- Room point, outside air, ventilation line (unclipped, on top) ----
  // Points outside the axes become an arrow on the edge with their value.
  let top = svg.append("g").attr("class", "hx-top")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
  function stateLines(s) {{
    let td = mollier.temperature_p_sat(s.phi * mollier.p_sat(s.t));
    return [esc(T.tip_x) + ": " + (s.x * 1000).toFixed(2) + " g/kg",
            esc(T.tip_t) + ": " + s.t.toFixed(1) + " °C",
            esc(T.tip_phi) + ": " + (s.phi * 100).toFixed(0) + " %",
            esc(T.tip_td) + ": " + (td == null || !isFinite(td) ? "–" : td.toFixed(1)) + " °C"];
  }}
  let roomState = roomPoint ? hxState(roomPoint.state, mollier, p) : null;
  if (outdoorPoint) {{
    let res = hxDrawOutdoor(top, ctx, outdoorPoint, roomState,
      Object.assign({{}}, outdoorOpts, {{texts: T}}));
    if (res.outdoor && res.marker) {{
      let lines = [esc(T.outdoor)].concat(stateLines(res.outdoor));
      if (res.note) lines.push(esc(res.note));
      res.marker.style("cursor", "pointer")
        .on("mouseover", function(event) {{
          showTip(event, lines.join("<br>"), outdoorOpts.color, "white");
        }})
        .on("mouseout", hideTip);
    }}
  }}
  if (roomPoint) {{
    let pt = hxDrawStatePoint(top, ctx, roomPoint.state,
      {{color: roomPoint.color, className: "hx-current current-point", texts: T}});
    if (pt) {{
      let lines = [esc(roomPoint.title) + (roomPoint.ts ? ": " + esc(roomPoint.ts) : "")]
        .concat(stateLines(pt.state));
      pt.marker.style("cursor", "pointer")
        .on("mouseover", function(event) {{
          if (!pt.outside) d3.select(this).attr("r", 10);
          showTip(event, lines.join("<br>"), roomPoint.color, "white");
        }})
        .on("mouseout", function() {{
          if (!pt.outside) d3.select(this).attr("r", 6);
          hideTip();
        }});
    }}
  }}

  // ----- Combined legend (bottom-right) ----------------------------------
  let legendItems = [];
  if (statePoints && statePoints.length > 0 && showStateLegend) {{
    for (let sp of statePoints) {{
      legendItems.push({{type: "process", number: sp.number, label: sp.label}});
    }}
  }}
  if (outdoorPoint && roomPoint && roomPoint.legend) {{
    legendItems.push({{type: "point", color: roomPoint.color, label: roomPoint.legend}});
  }}
  if (outdoorPoint && T.outdoor) {{
    legendItems.push({{type: "hollow", color: outdoorOpts.color, label: T.outdoor}});
  }}
  if (hasComfort && pathos && pathos.length > 0 && comfortLabel) {{
    legendItems.push({{type: "rect", color: "#9ACD32", label: comfortLabel}});
  }}
  if (hasOrange && orangeLabel) {{
    legendItems.push({{type: "dashes", color: orangeColor, label: orangeLabel}});
  }}
  if (hasRed && redLabel) {{
    legendItems.push({{type: "dashes", color: redColor, label: redLabel}});
  }}
  if (dataRecords && dataRecords.length > 0) {{
    for (let s of seasonLegend) {{
      if (s.label) legendItems.push({{type: "circle", color: s.color, label: s.label}});
    }}
  }}

  if (legendItems.length > 0) {{
    // Small charts: a slightly smaller legend in two columns, so it stays in
    // the empty corner below the saturation curve.
    let compact = height < 450;
    let scale = compact ? 0.85 : 1;
    let ncol = (compact && legendItems.length > 4) ? 2 : 1;
    let nrow = Math.ceil(legendItems.length / ncol);
    let rowH = 20, colGap = 16;
    let legendG = svg.append("g").attr("class", "hx-legend");
    let frame = legendG.append("rect")
      .attr("fill", "white").attr("stroke", "#bbb").attr("rx", 3);
    let groups = legendItems.map((item) => {{
      let g = legendG.append("g");
      if (item.type === "process") {{
        g.append("circle")
          .attr("cx", 10).attr("cy", 8).attr("r", 10)
          .attr("fill", "#222").attr("stroke", "white").attr("stroke-width", 1.5);
        g.append("text")
          .attr("x", 10).attr("y", 8).attr("text-anchor", "middle").attr("dy", "0.35em")
          .attr("fill", "white").attr("font-size", "10px").attr("font-weight", "bold")
          .text(item.number);
      }} else if (item.type === "rect") {{
        g.append("rect").attr("x", 2).attr("y", 1).attr("width", 14).attr("height", 14)
          .attr("fill", item.color).attr("opacity", 0.7);
      }} else if (item.type === "dashes") {{
        g.append("line").attr("x1", 1).attr("y1", 8).attr("x2", 17).attr("y2", 8)
          .attr("stroke", item.color).attr("stroke-width", 2)
          .attr("stroke-dasharray", "5,3");
      }} else if (item.type === "hollow") {{
        g.append("circle").attr("cx", 9).attr("cy", 8).attr("r", 5)
          .attr("fill", "white").attr("stroke", item.color).attr("stroke-width", 2.5);
      }} else if (item.type === "point") {{
        g.append("circle").attr("cx", 9).attr("cy", 8).attr("r", 5.5)
          .attr("fill", item.color).attr("stroke", "white").attr("stroke-width", 1.5);
      }} else {{
        g.append("circle").attr("cx", 9).attr("cy", 8).attr("r", 5)
          .attr("fill", item.color).attr("opacity", 0.7);
      }}
      g.append("text").attr("x", item.type === "process" ? 26 : 22).attr("y", 12)
        .style("font-family", "Tahoma, Geneva, sans-serif").style("font-size", "12px")
        .text(item.label);
      return g;
    }});
    // Column-major layout; each column as wide as its longest label.
    let colX = 0, legendW = 0;
    for (let c = 0; c < ncol; c++) {{
      let colW = 0;
      for (let r = 0; r < nrow; r++) {{
        let i = c * nrow + r;
        if (i >= groups.length) break;
        groups[i].attr("transform", "translate(" + colX + "," + (r * rowH + 5) + ")");
        colW = Math.max(colW, groups[i].node().getBBox().width);
      }}
      legendW = colX + colW;
      colX += colW + colGap;
    }}
    legendW = Math.max(ncol === 1 ? 110 : 0, legendW);
    let legendH = nrow * rowH + 10;
    frame.attr("x", -6).attr("y", -6)
      .attr("width", legendW + 12).attr("height", legendH + 6);
    // Anchor bottom-right of the plot area, clear of the enthalpy-axis
    // labels (plot-local x = width-20, y = height-20).
    let lx = Math.max(margin.left + 8, margin.left + width - scale * legendW - (compact ? 40 : 50));
    let ly = margin.top + height - scale * legendH - (compact ? 26 : 35);
    legendG.attr("transform", "translate(" + lx + "," + ly + ")"
      + (scale !== 1 ? " scale(" + scale + ")" : ""));
  }}
}})();
</script>"""


# Default comfort-donut palette — vivid, semantic (matches the report).
_DONUT_TEMP_COLORS = ("#3498DB", "#2ECC71", "#E74C3C")  # cold / in range / warm
_DONUT_HUM_COLORS = ("#F39C12", "#2ECC71", "#3498DB")   # dry / in range / humid
_DONUT_FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
_DONUT_MUTED = "#8A94A6"


def _donut_pct_text(vals, min_pct: float = 6.0):
    """In-wedge percent labels, suppressed for slices below *min_pct*."""
    total = sum(vals)
    if total <= 0:
        return ["" for _ in vals]
    return [f"{v / total * 100:.0f}%" if (v / total * 100) >= min_pct else ""
            for v in vals]


def _bullet_legend(names, vals, colors, count_label, cx, y_top, total):
    """Build a vertical bullet-point legend as a single centred annotation.

    The three entries are stacked as ``<br>`` lines inside one annotation:
    left-justified rows (so the bullets line up) within a block that is
    centred under the donut at column centre *cx*. Using one annotation
    keeps the line spacing tight (default line height).
    """
    lines = []
    for name, val, col in zip(names, vals, colors):
        pct = f"{val / total * 100:.0f}%" if total else "0%"
        lines.append(
            f"<span style='color:{col}'>●</span>  "
            f"{name} — <b>{val}</b> {count_label} "
            f"<span style='color:{_DONUT_MUTED}'>({pct})</span>"
        )
    return [dict(
        text="<br>".join(lines),
        x=cx, y=y_top, xref="paper", yref="paper", showarrow=False,
        xanchor="center", yanchor="top", align="left",
        font=dict(size=12, family=_DONUT_FONT, color="#3A4150"),
    )]


def plot_comfort_donuts(
    data: pd.DataFrame,
    temp_range: Tuple[float, float] = (20.0, 26.0),
    hum_range: Tuple[float, float] = (30.0, 65.0),
    title: Optional[str] = None,
    temp_colors: Tuple[str, str, str] = _DONUT_TEMP_COLORS,
    hum_colors: Tuple[str, str, str] = _DONUT_HUM_COLORS,
    count_label: str = "days",
    show_center_stats: bool = True,
) -> go.Figure:
    """Two donut charts showing time spent below / within / above comfort.

    The left donut splits temperature into *too cold* / *in range* / *too
    warm*, the right donut splits humidity into *too dry* / *in range* /
    *too humid*. Each slice shows its share inside the ring; the centre
    shows the average with the min–max range; and **each donut has its own
    vertical bullet-point legend** listing the slice counts and percentages
    (e.g. number of days when the input is daily data). Hover tooltips show
    the count and the percentage.

    Args:
        data: DataFrame with columns ``temperature`` [°C] and
            ``humidity`` [%rH] (extra columns are ignored).
        temp_range: ``(min, max)`` comfort temperature band [°C].
        hum_range: ``(min, max)`` comfort humidity band [%rH].
        title: Overall figure title. Default *None*.
        temp_colors: Colors for (cold, in range, warm).
        hum_colors: Colors for (dry, in range, humid).
        count_label: Unit word for the per-slice counts, used in the
            legend and tooltips (e.g. ``"days"`` -> ``"In range — 42 days"``).
        show_center_stats: Draw the average and min–max range in the centre
            of each donut. Default *True*.

    Returns:
        go.Figure: Plotly figure with two donut traces and a vertical
        bullet-point legend under each.

    See Also:
        :func:`plot_comfort_compass` for the direction of the deviations, and
        what the comfort zone and warning bands mean:
        https://retomarek.github.io/pyedautils/guides/comfort_compass.html#compass-comfort
    """
    df = data.copy()
    t = pd.to_numeric(df["temperature"], errors="coerce").dropna()
    h = pd.to_numeric(df["humidity"], errors="coerce").dropna()

    t_lo, t_hi = temp_range
    h_lo, h_hi = hum_range
    temp_vals = [int((t < t_lo).sum()), int(((t >= t_lo) & (t <= t_hi)).sum()),
                 int((t > t_hi).sum())]
    hum_vals = [int((h < h_lo).sum()), int(((h >= h_lo) & (h <= h_hi)).sum()),
                int((h > h_hi).sum())]

    temp_names = ["Too cold", "In range", "Too warm"]
    hum_names = ["Too dry", "In range", "Too humid"]
    sub_titles = ("Temperature", "Humidity")
    hover = ("<b>%{customdata}</b><br>%{value} " + count_label
             + " · %{percent}<extra></extra>")

    # Symmetric side-by-side domains; ring centres at x = 0.24 / 0.76.
    cx_l, cx_r, cy = 0.24, 0.76, 0.62
    left_dom = dict(x=[0.0, 0.48], y=[0.36, 0.90])
    right_dom = dict(x=[0.52, 1.0], y=[0.36, 0.90])

    common = dict(
        hole=0.64, sort=False, direction="clockwise", rotation=0,
        textinfo="text", textposition="inside",
        insidetextfont=dict(color="white", size=13, family=_DONUT_FONT),
        marker=dict(line=dict(color="white", width=2)),
        hovertemplate=hover, showlegend=False,
    )
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=temp_names, values=temp_vals, customdata=temp_names,
        marker_colors=list(temp_colors), domain=left_dom,
        text=_donut_pct_text(temp_vals), **common,
    ))
    fig.add_trace(go.Pie(
        labels=hum_names, values=hum_vals, customdata=hum_names,
        marker_colors=list(hum_colors), domain=right_dom,
        text=_donut_pct_text(hum_vals), **common,
    ))

    def _mean_text(series, unit, fmt):
        return "—" if series.empty else f"{series.mean():{fmt}} {unit}"

    def _range_text(series, unit, fmt):
        if series.empty:
            return ""
        return f"{series.min():{fmt}} – {series.max():{fmt}} {unit}"

    # Section titles above each ring.
    annotations = [
        dict(text=sub_titles[0], x=cx_l, y=0.97, font=dict(size=14, family=_DONUT_FONT)),
        dict(text=sub_titles[1], x=cx_r, y=0.97, font=dict(size=14, family=_DONUT_FONT)),
    ]
    for a in annotations:
        a.update(xref="paper", yref="paper", showarrow=False,
                 xanchor="center", yanchor="middle")

    if show_center_stats:
        center = [
            dict(text=f"<b>{_mean_text(t, '°C', '.1f')}</b>", x=cx_l, y=cy + 0.03,
                 font=dict(size=18, family=_DONUT_FONT, color="#2C3038")),
            dict(text=_range_text(t, "°C", ".1f"), x=cx_l, y=cy - 0.05,
                 font=dict(size=11, family=_DONUT_FONT, color=_DONUT_MUTED)),
            dict(text=f"<b>{_mean_text(h, '%', '.0f')}</b>", x=cx_r, y=cy + 0.03,
                 font=dict(size=18, family=_DONUT_FONT, color="#2C3038")),
            dict(text=_range_text(h, "%", ".0f"), x=cx_r, y=cy - 0.05,
                 font=dict(size=11, family=_DONUT_FONT, color=_DONUT_MUTED)),
        ]
        for a in center:
            a.update(xref="paper", yref="paper", showarrow=False,
                     xanchor="center", yanchor="middle", align="center")
        annotations += center

    # Vertical bullet-point legend (centred) under each donut.
    annotations += _bullet_legend(temp_names, temp_vals, temp_colors,
                                  count_label, cx_l, 0.26, sum(temp_vals))
    annotations += _bullet_legend(hum_names, hum_vals, hum_colors,
                                  count_label, cx_r, 0.26, sum(hum_vals))

    fig.update_layout(
        title_text=f"<b>{title}</b>" if title else None,
        title_font=dict(size=20, family=_DONUT_FONT), title_x=0.5,
        template="plotly_white",
        font=dict(family=_DONUT_FONT),
        annotations=annotations,
        showlegend=False,
        margin=dict(l=20, r=20, t=60 if title else 40, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Comfort compass — area-true polar glyph (Plotly barpolar)
# ---------------------------------------------------------------------------
# Two-axis colour model: temperature poles warm=red / cold=blue, humidity poles
# dry=orange / humid=violet; a mixed direction's hue is the OKLCH hue of the
# mean of its two pole colours (#E74C3C warm, #3498DB cold, #F39C12 dry,
# #8E5BD9 humid).
_COMPASS_OK = "#22C55E"
# Directions in drawing order, at THETA 0,45,...,315 (0 deg = East, CCW). With
# the default polar orientation this puts humid right, warm top, dry left,
# cold bottom — same axes as an h,x (Mollier) diagram.
_COMPASS_DIR_ORDER = ("f", "wf", "w", "wt", "t", "ct", "c", "kf")
_COMPASS_STAGE_ORDER = ("l", "d", "s")
_COMPASS_THETA = [0, 45, 90, 135, 180, 225, 270, 315]
# Severity colours: one fixed OKLCH lightness plane per stage (mild L=0.765,
# moderate L=0.6475, severe L=0.53) at each direction's hue, so "severe" is
# always the darkest colour in the glyph no matter the direction — the shade
# alone identifies the stage, even across arms and for colour-weak readers.
# Ramps validated (monotone L, step gaps >= 0.06, light end >= 2:1 on white).
_COMPASS_STAGE_COLORS = {
    "w": {"l": "#e49e93", "d": "#cf6e60", "s": "#b04336"},
    "wf": {"l": "#dc9dbb", "d": "#c66d9a", "s": "#a64278"},
    "f": {"l": "#baa7e2", "d": "#9a7ccf", "s": "#7954b3"},
    "kf": {"l": "#9eb0ea", "d": "#7289da", "s": "#4e63bf"},
    "c": {"l": "#82b9e5", "d": "#3d96d4", "s": "#0072af"},
    "ct": {"l": "#afb982", "d": "#8b955f", "s": "#69713d"},
    "t": {"l": "#d6a977", "d": "#bf7f29", "s": "#965d00"},
    "wt": {"l": "#e0a383", "d": "#cb7545", "s": "#ab4b00"},
}
# Neutral greys on the same three lightness planes — the severity key swatches.
_COMPASS_STAGE_GREYS = {"l": "#b2b2b2", "d": "#8e8e8e", "s": "#6c6c6c"}
_COMPASS_R0 = 1.0
_COMPASS_AREA = math.pi * _COMPASS_R0 ** 2
_COMPASS_WID_DEG = 40.0
_COMPASS_WID_RAD = math.radians(_COMPASS_WID_DEG)
_COMPASS_RMAX = math.sqrt(2 * _COMPASS_AREA / _COMPASS_WID_RAD) * 1.04
# English display names; override via direction_labels / names.
# Default texts (English, lower-case). Every one can be overridden per call via
# `names` / `direction_labels` / `stage_names` / `count_label` / `title`. German
# presets below mirror the d3-mollierhx wording; pass e.g. names=_COMPASS_NAMES_DE.
_COMPASS_NAMES = {
    "ok": "in range", "w": "too warm", "wf": "warm + humid", "f": "too humid",
    "kf": "cold + humid", "c": "too cold", "ct": "cold + dry", "t": "too dry",
    "wt": "warm + dry",
}
_COMPASS_LABELS = ["too humid", "warm +<br>humid", "too warm", "warm +<br>dry",
                   "too dry", "cold +<br>dry", "too cold", "cold +<br>humid"]
_COMPASS_STAGE_NAMES = {"l": "mild", "d": "moderate", "s": "severe"}

_COMPASS_NAMES_DE = {
    "ok": "im Zielband", "w": "zu warm", "wf": "warm + feucht", "f": "zu feucht",
    "kf": "kalt + feucht", "c": "zu kalt", "ct": "kalt + trocken",
    "t": "zu trocken", "wt": "warm + trocken",
}
_COMPASS_LABELS_DE = ["zu feucht", "warm +<br>feucht", "zu warm",
                      "warm +<br>trocken", "zu trocken", "kalt +<br>trocken",
                      "zu kalt", "kalt +<br>feucht"]
_COMPASS_STAGE_NAMES_DE = {"l": "leicht", "d": "mittel", "s": "stark"}


def _compass_legend(d, names, count_label, stage_names,
                    count_label_singular=None) -> dict:
    """Donut-style bullet legend: 'in range' first, then the deviation
    directions present, sorted by count (biggest first). Each row shows the
    count and its share, lower-cased like the donut legend. If any deviation
    is present, a severity key (grey swatches on the three stage lightness
    planes, light -> dark) is appended below the rows. With
    *count_label_singular* a count of exactly 1 uses the singular word
    (e.g. "1 Tag" instead of "1 Tage")."""
    total = sum(float(v) for v in d.values())
    g = lambda k: float(d.get(k, 0.0))

    def pct(cnt):
        return f"{cnt / total * 100:.0f}" if total else "0"

    def unit(cnt):
        if int(round(cnt)) == 1 and count_label_singular:
            return count_label_singular
        return count_label

    rows = [("ok", g("ok"), _COMPASS_OK, None)]
    deviations = []
    for k in _COMPASS_DIR_ORDER:
        cnt = sum(g(f"{k}_{s}") for s in _COMPASS_STAGE_ORDER)
        if cnt > 0:
            split = "/".join(pct(g(f"{k}_{s}")) for s in _COMPASS_STAGE_ORDER)
            deviations.append((k, cnt, _COMPASS_STAGE_COLORS[k]["d"], split))
    deviations.sort(key=lambda x: x[1], reverse=True)
    rows += deviations
    lines = []
    for k, cnt, col, split in rows:
        share = f"{pct(cnt)}%" if split is None else f"{pct(cnt)}% → {split}%"
        lines.append(
            f"<span style='color:{col}'>●</span>  {names[k]} — "
            f"<b>{int(round(cnt))}</b> {unit(cnt)} "
            f"<span style='color:{_DONUT_MUTED}'>({share})</span>"
        )
    if deviations:
        sep = f" <span style='color:{_DONUT_MUTED}'>/</span> "
        key = sep.join(
            f"<span style='color:{_COMPASS_STAGE_GREYS[st]}'>●</span> "
            f"<span style='color:{_DONUT_MUTED}'>{stage_names[st]}</span>"
            for st in _COMPASS_STAGE_ORDER
        )
        lines.append("")
        lines.append(key)
    return dict(
        text="<br>".join(lines), x=0.53, y=0.5, xref="paper", yref="paper",
        showarrow=False, xanchor="left", yanchor="middle", align="left",
        font=dict(size=13, family=_DONUT_FONT, color="#3A4150"),
    )


def _compass_wedges(g, frac, r_green, leg_names, stage_name, hov_fmt):
    """Stacked, area-true severity wedges per direction. A ring sector from
    r_prev to r_next over WID has area (WID/2)(r_next^2 - r_prev^2); choosing
    r_next so that the cumulative area equals (share * total area) makes every
    wedge area proportional to its share of days. Returns the barpolar arrays
    plus the outer radius of each arm (in _COMPASS_DIR_ORDER; ``r_green`` for
    a direction without days)."""
    th, rr, base, wid, col, hov, arms = [], [], [], [], [], [], []
    for ang, k in zip(_COMPASS_THETA, _COMPASS_DIR_ORDER):
        cum, r_prev = 0.0, r_green
        for st in _COMPASS_STAGE_ORDER:
            cnt = g(f"{k}_{st}")
            cum += frac(f"{k}_{st}")
            r_next = math.sqrt(r_green ** 2 + 2 * _COMPASS_AREA * cum / _COMPASS_WID_RAD)
            if r_next > r_prev + 1e-9:
                th.append(ang); base.append(r_prev); rr.append(r_next - r_prev)
                wid.append(_COMPASS_WID_DEG); col.append(_COMPASS_STAGE_COLORS[k][st])
                hov.append(hov_fmt(f"{leg_names[k]} · {stage_name[st]}", cnt))
            r_prev = r_next
        arms.append(r_prev)
    return th, rr, base, wid, col, hov, arms


# Text layout of the compass (sizes in px). The radial range is in glyph units;
# to know whether a text fits, the figure size is turned into the screen radius
# of the polar subplot (see _compass_polar_px).
_COMPASS_FIXED_RANGE = _COMPASS_RMAX * 1.15   # radial range with scale="fixed"
_COMPASS_LABEL_FONT = 11                      # direction labels
_COMPASS_CHAR_W = 0.52                        # mean glyph width per font px
_COMPASS_LINE_H = 1.25                        # line height per font px
_COMPASS_LABEL_GAP_PX = 4                     # arm end -> label anchor
_COMPASS_LABEL_PAD_PX = 7                     # min. distance label box -> glyph
_COMPASS_LABEL_WRAP = 14                      # wrap longer label lines at spaces
_COMPASS_DEFAULT_WIDTH = 700                  # Plotly's default figure width
# Outward anchor per direction (order of _COMPASS_THETA), so a label never
# reaches back into its arm or the green disc.
_COMPASS_LABEL_POS = ["middle right", "top right", "top center", "top left",
                      "middle left", "bottom left", "bottom center",
                      "bottom right"]


def _compass_plain(text) -> str:
    """*text* without HTML tags (for width estimates)."""
    out, depth = [], 0
    for ch in str(text):
        if ch == "<":
            depth += 1
        elif ch == ">" and depth:
            depth -= 1
        elif not depth:
            out.append(ch)
    return "".join(out)


def _compass_wrap(text, width=_COMPASS_LABEL_WRAP) -> str:
    """Break label lines longer than *width* characters at spaces, so long
    custom labels do not run off the figure. Existing ``<br>`` breaks are
    kept; lines with HTML tags are left alone."""
    out = []
    for line in str(text).split("<br>"):
        if len(line) <= width or "<" in line or " " not in line:
            out.append(line)
            continue
        cur = ""
        for word in line.split(" "):
            cand = f"{cur} {word}" if cur else word
            if cur and len(cand) > width:
                out.append(cur)
                cur = word
            else:
                cur = cand
        out.append(cur)
    return "<br>".join(out)


def _compass_text_box(text, font):
    """Rough (width, height) in px of a Plotly text with ``<br>`` breaks."""
    lines = str(text).split("<br>")
    w = max(len(_compass_plain(ln)) for ln in lines) * _COMPASS_CHAR_W * font
    return w, len(lines) * font * _COMPASS_LINE_H


def _compass_polar_px(height, width, show_legend, has_title):
    """Screen radius (px) of the polar subplot, its centre in paper y and the
    plot-area height (px). Plotly fits the polar circle into the square that
    fits the domain; with ``width=None`` Plotly's default 700 px is assumed."""
    plot_w = (width or _COMPASS_DEFAULT_WIDTH) - 40
    plot_h = height - (66 if has_title else 30) - 20
    x0, x1 = (0.0, 0.44) if show_legend else (0.06, 0.94)
    y0, y1 = 0.06, (0.88 if has_title else 0.96)
    radius = max(min(plot_w * (x1 - x0), plot_h * (y1 - y0)) / 2, 1.0)
    return radius, (y0 + y1) / 2, max(plot_h, 1.0), (x0 + x1) / 2


def _compass_label_bounds(height, width, show_legend, has_title):
    """Room for the direction labels in px around the polar centre (x0, y0,
    x1, y1, y up): the figure edges, the legend column on the right and the
    title block on top."""
    fig_w = width or _COMPASS_DEFAULT_WIDTH
    plot_w = fig_w - 40
    top = 66 if has_title else 30
    plot_h = height - top - 20
    x0, x1 = (0.0, 0.44) if show_legend else (0.06, 0.94)
    y0, y1 = 0.06, (0.88 if has_title else 0.96)
    cx = 20 + plot_w * (x0 + x1) / 2              # from the left figure edge
    cy = 20 + plot_h * (y0 + y1) / 2              # from the bottom figure edge
    right = (20 + plot_w * 0.53 - 6) if show_legend else fig_w - 2
    return (2 - cx, 2 - cy, right - cx, height - top + 10 - cy)


def _compass_box(ang, dist, w, h, pos):
    """Box (x0, y0, x1, y1) in px, y up, of a text of size w x h anchored
    with Plotly textposition *pos* at *dist* px from the centre at *ang*."""
    ax = dist * math.cos(math.radians(ang))
    ay = dist * math.sin(math.radians(ang))
    v, hz = pos.split()
    y0, y1 = {"top": (ay, ay + h), "bottom": (ay - h, ay)}.get(
        v, (ay - h / 2, ay + h / 2))
    x0, x1 = {"right": (ax, ax + w), "left": (ax - w, ax)}.get(
        hz, (ax - w / 2, ax + w / 2))
    return x0, y0, x1, y1


def _compass_hits_glyph(box, arms_px, r0_px):
    """True if the box comes closer than the padding to the dashed circle,
    the green disc or any arm (arms are sectors of _COMPASS_WID_DEG)."""
    x0, y0, x1, y1 = box
    pad = _COMPASS_LABEL_PAD_PX
    nx = max(2, int((x1 - x0) / 3) + 1)
    ny = max(2, int((y1 - y0) / 3) + 1)
    pts = ([(x0 + (x1 - x0) * i / (nx - 1), y) for i in range(nx)
            for y in (y0, y1)]
           + [(x, y0 + (y1 - y0) * j / (ny - 1)) for j in range(ny)
              for x in (x0, x1)])
    half = _COMPASS_WID_DEG / 2
    for x, y in pts:
        d = math.hypot(x, y)
        lim = r0_px
        if d < max(arms_px) + pad:
            a = math.degrees(math.atan2(y, x)) % 360
            for th, arm in zip(_COMPASS_THETA, arms_px):
                # angular half-width of the padding at this distance
                da = math.degrees(pad / max(d, 1.0))
                if abs((a - th + 180) % 360 - 180) <= half + da:
                    lim = max(lim, arm)
        if d < lim + pad:
            return True
    return False


def _compass_place_labels(arms, labels, k, bounds=None):
    """Place each label at *k* px per glyph unit: start just beyond the end of
    its own arm (at least the dashed circle) and move out until its text box
    clears the whole glyph. If *bounds* (x0, y0, x1, y1 in px around the
    centre, e.g. the figure edges) are given and a box crosses them, the
    label's font is reduced (down to 7 px) so long labels are not cut off.
    Returns ``[(anchor distance px, box, font size), ...]``."""
    arms_px = [max(a, 0.0) * k for a in arms]
    r0_px = _COMPASS_R0 * k
    placed = []
    for ang, arm, lab, pos in zip(_COMPASS_THETA, arms_px, labels,
                                  _COMPASS_LABEL_POS):
        font = float(_COMPASS_LABEL_FONT)
        while True:
            w, h = _compass_text_box(lab, font)
            dist = max(arm, r0_px) + _COMPASS_LABEL_GAP_PX
            box = _compass_box(ang, dist, w, h, pos)
            for _ in range(400):
                if not _compass_hits_glyph(box, arms_px, r0_px):
                    break
                dist += 1.5
                box = _compass_box(ang, dist, w, h, pos)
            inside = bounds is None or (
                box[0] >= bounds[0] and box[1] >= bounds[1]
                and box[2] <= bounds[2] and box[3] <= bounds[3])
            if inside or font <= 7:
                break
            font = max(7.0, font - 0.5)
        placed.append((dist, box, font))
    return placed


def _compass_fit_range(arms, labels, radius_px):
    """Smallest radial range (zoomed to this glyph, scale="fit") that keeps the
    glyph and its labels (placed by _compass_place_labels) inside the polar
    subplot's square. Never wider than the fixed-scale range."""
    k_hi = radius_px / (max(max(arms), _COMPASS_R0) * 1.04)   # px per unit
    k_lo = radius_px / _COMPASS_FIXED_RANGE

    def fits(k):
        if not labels:
            return True
        return all(max(abs(c) for c in box) <= radius_px
                   for _, box, _ in _compass_place_labels(arms, labels, k))

    if fits(k_hi):
        return radius_px / k_hi
    if not fits(k_lo):
        return _COMPASS_FIXED_RANGE
    for _ in range(20):
        mid = (k_lo + k_hi) / 2
        k_lo, k_hi = (mid, k_hi) if fits(mid) else (k_lo, mid)
    return radius_px / k_lo


def _compass_ok_text(pct_text, word, r_px):
    """How to fill the green disc of screen radius *r_px*. Returns
    ``(mode, pct_font, word_font, gap)`` with mode ``"both"`` (share above the
    in-range word, a tight block), ``"pct"`` (share only) or *None*."""
    f = _COMPASS_LABEL_FONT
    pct_font = min(30.0, max(f * 1.3, r_px * 0.42))
    word_font = max(f * 0.9, pct_font * 0.48)
    gap = pct_font * 0.12
    pct_w = len(pct_text) * 0.62 * pct_font
    block_h = pct_font * 0.75 + gap + word_font
    block_w = max(pct_w, len(_compass_plain(word)) * 0.56 * word_font)
    # keep a margin to the disc edge: the block's corners must stay inside
    if r_px > 0 and block_h <= r_px * 1.2 and block_w <= r_px * 1.55:
        return "both", pct_font, word_font, gap
    if r_px > 0 and pct_font <= r_px * 1.2 and pct_w <= r_px * 1.4:
        return "pct", pct_font, word_font, gap
    return None, pct_font, word_font, gap


def plot_comfort_compass(
    distribution,
    title: Optional[str] = None,
    show_direction_labels: bool = True,
    show_legend: bool = True,
    count_label: str = "days",
    direction_labels: Optional[List[str]] = None,
    names: Optional[Dict[str, str]] = None,
    stage_names: Optional[Dict[str, str]] = None,
    fixed_scale: bool = True,
    stats_text: Optional[str] = None,
    count_label_singular: Optional[str] = None,
    height: int = 460,
    scale: Optional[str] = None,
    width: Optional[int] = None,
) -> go.Figure:
    """Area-true "comfort compass" glyph for one room / group.

    A single equal-area polar glyph summarising how a room's days split across
    the nine comfort states (see
    :func:`pyedautils.comfort.comfort_compass_distribution`):

    - **Centre** — a green disc whose *area* is the share of days *in range*.
      It says what green means: the share and the in-range word
      (``names["ok"]``, default ``"in range"``), e.g. ``52%`` / ``in range``.
      When the disc is too small for both lines only the share is printed
      inside, and when even that does not fit, nothing. Without the legend
      (``show_legend=False``) a caption below the glyph then gives the share
      instead, e.g. ``● 8% in range``; with the legend its first row does.
    - **Eight wedges** — one per deviation direction; the *wedge area* is the
      share of days, and the three radial sub-segments (inner → outer) split
      that share by severity. Severity is encoded as lightness on three fixed
      planes shared by all directions — mild is always the lightest shade in
      the glyph, severe always the darkest — so the shade alone identifies the
      stage. Orientation matches an h,x diagram: warm top, cold bottom, humid
      right, dry left; mixed directions blend the hues of their two poles.
    - **Direction labels** — each label sits just beyond the end of its own
      arm (at least the dashed circle), anchored outward and moved out until
      its text box clears the whole glyph (disc and neighbouring arms). Label
      lines longer than 14 characters are wrapped at spaces; a label that
      would still run off the figure is drawn smaller (down to 7 px).
    - Every glyph has the **same total area** (= 100 % of days) — only the
      shape tells the story. With ``scale="fixed"`` (the default, same as
      ``fixed_scale=True``) the radial axis is framed to the worst-case
      extent, so the dashed reference circle is drawn at the **same size in
      every chart** and glyphs stay directly comparable by area. With
      ``scale="fit"`` the glyph is zoomed so that it and its labels fill the
      panel; the circle — and thus the scale — then differs from chart to
      chart, so compare such charts by their percentages only.

    The percentages are listed in a donut-style bullet legend on the right —
    each deviation row shows the count, the total share and its
    mild/moderate/severe split (e.g. ``(19% → 5/4/10%)``) — followed by a grey
    severity key (mild / moderate / severe swatches on the same three
    lightness planes). Hovering a segment shows only that segment: direction ·
    stage, its count and share.

    Args:
        distribution: Mapping of comfort-compass categories to **counts**, as
            returned by :func:`pyedautils.comfort.comfort_compass_distribution`
            (keys ``"ok"`` and ``"<direction>_<stage>"``; missing keys count as
            0). Percentages and areas are derived from the counts.
        title: Figure title; a muted subtitle underneath shows the total count
            (e.g. ``265 days``). Default *None*.
        show_direction_labels: Draw the eight direction labels around the rose.
            Default *True*.
        show_legend: Draw the bullet legend with the per-direction counts and
            percentages. Default *True*.
        count_label: Unit word for the counts in legend, hover and title
            subtitle (e.g. ``"days"``, ``"hours"``). Default ``"days"``.
        direction_labels: Optional 8 labels for the rose (order: humid,
            warm+humid, warm, warm+dry, dry, cold+dry, cold, cold+humid). Use
            ``"<br>"`` for line breaks. Default English; pass
            ``pyedautils.plots.comfort._COMPASS_LABELS_DE`` for German.
        names: Optional mapping overriding the direction names in legend and
            hover and the in-range word in the green disc / caption (keys
            ``"ok"`` and ``"f","wf","w","wt","t","ct","c","kf"``; missing keys
            keep the English default). Pass ``_COMPASS_NAMES_DE`` for German.
        stage_names: Optional mapping overriding the severity-stage names shown
            in the hover and the severity key (keys ``"l"`` mild, ``"d"``
            moderate, ``"s"`` severe). Default English; pass
            ``_COMPASS_STAGE_NAMES_DE`` for German.
        fixed_scale: Keep the radial axis fixed to the worst-case extent so the
            dashed 100 % reference circle is the same size in every chart,
            keeping glyphs from different rooms/periods directly comparable.
            Default ``True``. ``False`` is the same as ``scale="fit"``.
            Ignored when ``scale`` is given.
        stats_text: Optional extra text block (e.g. min/mean/max readings)
            rendered left-aligned **below the legend** (``<br>`` for line
            breaks, inline HTML spans allowed). With ``show_legend=False`` it
            is drawn on its own in the legend column. Default *None* (off).
        count_label_singular: Optional singular unit word used in the legend,
            hover and title subtitle when a count is exactly 1 (e.g. pass
            ``count_label="Tage", count_label_singular="Tag"`` for German).
            Default *None* (always use ``count_label``).
        height: Figure height in pixels. Default 460.
        scale: ``"fixed"`` (same 100 % circle in every chart) or ``"fit"``
            (zoomed to this glyph and its labels). Takes precedence over
            ``fixed_scale``; default *None* uses ``fixed_scale``.
        width: Optional figure width in pixels. Default *None* (Plotly's
            default / container width). Whether the texts fit into the green
            disc is estimated for this width (700 px when *None*).

    All visible texts can be set by the caller: ``title``, ``count_label`` /
    ``count_label_singular``, ``names`` (incl. the in-range word),
    ``direction_labels``, ``stage_names`` and ``stats_text``; the remaining
    characters are numbers, ``%`` and symbols. Example for German::

        from pyedautils.plots.comfort import (
            _COMPASS_LABELS_DE, _COMPASS_NAMES_DE, _COMPASS_STAGE_NAMES_DE)
        plot_comfort_compass(dist, title="Schlafzimmer",
                             names=_COMPASS_NAMES_DE,
                             direction_labels=_COMPASS_LABELS_DE,
                             stage_names=_COMPASS_STAGE_NAMES_DE,
                             count_label="Tage", count_label_singular="Tag")

    Returns:
        go.Figure

    Raises:
        ValueError: If ``scale`` is not ``None``, ``"fixed"`` or ``"fit"``.

    See Also:
        How to read the compass, compare rooms (``scale``), and what
        the arms mean for comfort and energy:
        https://retomarek.github.io/pyedautils/guides/comfort_compass.html
    """
    if scale is None:
        scale = "fixed" if fixed_scale else "fit"
    if scale not in ("fixed", "fit"):
        raise ValueError(f"scale must be 'fixed' or 'fit', got {scale!r}")
    d = dict(distribution)
    g = lambda k: float(d.get(k, 0.0))
    leg_names = {**_COMPASS_NAMES, **(names or {})}
    total = sum(float(v) for v in d.values())
    frac = lambda k: (g(k) / total) if total else 0.0

    ok = frac("ok")
    r_green = _COMPASS_R0 * math.sqrt(max(ok, 0.0))

    stage_name = {**_COMPASS_STAGE_NAMES, **(stage_names or {})}

    def _unit(cnt):
        if int(round(cnt)) == 1 and count_label_singular:
            return count_label_singular
        return count_label

    def _hov(name, cnt):
        pct = cnt / total * 100 if total else 0.0
        return (f"<b>{name}</b><br>{int(round(cnt))} {_unit(cnt)} "
                f"({pct:.0f}%)")

    th, rr, base, wid, col, hov, arms = _compass_wedges(
        g, frac, r_green, leg_names, stage_name, _hov)
    labels = ([_compass_wrap(t) for t in (direction_labels or _COMPASS_LABELS)]
              if show_direction_labels else None)

    # Radial-axis range. "fixed" frames to the worst-case extent, so
    # r=_COMPASS_R0 (the dashed reference circle) maps to the same screen
    # radius in every chart and glyphs stay comparable. "fit" zooms so that
    # glyph and labels fill the polar square.
    radius_px, cy, plot_h, cx = _compass_polar_px(height, width, show_legend,
                                                  bool(title))
    r_range = (_COMPASS_FIXED_RANGE if scale == "fixed"
               else _compass_fit_range(arms, labels, radius_px))
    px_per_unit = radius_px / r_range

    circ = list(range(0, 361, 4))
    fig = go.Figure()
    # 100 %-reference circle (dashed) = footprint if everything were in range.
    fig.add_trace(go.Scatterpolar(
        r=[_COMPASS_R0] * len(circ), theta=circ, mode="lines",
        line=dict(color="#cccccc", width=0.8, dash="dot"),
        hoverinfo="skip", showlegend=False))
    # green centre disc (area ~ in range) — a full-circle bar so it is hoverable.
    if r_green > 0:
        fig.add_trace(go.Barpolar(
            r=[r_green], theta=[0], width=[360], base=[0],
            marker=dict(color=_COMPASS_OK, line=dict(color="white", width=1.5)),
            hovertext=[_hov(leg_names["ok"], g("ok"))],
            hovertemplate="%{hovertext}<extra></extra>", showlegend=False))
    if th:
        fig.add_trace(go.Barpolar(
            r=rr, theta=th, base=base, width=wid,
            marker=dict(color=col, line=dict(color="white", width=1.5)),
            hovertext=hov, hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False))
    # Inside the green disc: share above the in-range word as one tight,
    # centred block, or the share alone, as far as it fits (the legend /
    # caption carries the value otherwise).
    ok_pct = f"{ok * 100:.0f}%"
    ok_mode, pct_font, word_font, ok_gap = _compass_ok_text(
        ok_pct, leg_names["ok"], r_green * px_per_unit)
    ok_font = dict(color="white", family=_DONUT_FONT)
    if ok_mode == "both":
        # Distance between the visual centres of the digits (cap height
        # ~0.72 em) and the word (x-height ~0.5 em plus descender).
        sep = pct_font * 0.36 + ok_gap + word_font * 0.45
        for text, size, dy, name in ((ok_pct, pct_font, sep / 2, "ok_pct"),
                                     (leg_names["ok"], word_font, -sep / 2,
                                      "ok_word")):
            fig.add_trace(go.Scatterpolar(
                r=[abs(dy) / px_per_unit], theta=[90 if dy > 0 else 270],
                mode="text", text=[text], textposition="middle center",
                textfont=dict(size=round(size, 1), **ok_font), name=name,
                cliponaxis=False, hoverinfo="skip", showlegend=False))
    elif ok_mode == "pct":
        fig.add_trace(go.Scatterpolar(
            r=[0], theta=[0], mode="text", text=[ok_pct],
            textposition="middle center", name="ok_pct",
            textfont=dict(size=round(pct_font, 1), **ok_font),
            hoverinfo="skip", showlegend=False))
    # Each label just beyond the end of its own arm (at least the dashed
    # circle), anchored outward and moved out until its text box clears the
    # whole glyph.
    placed = (_compass_place_labels(
        arms, labels, px_per_unit,
        _compass_label_bounds(height, width, show_legend, bool(title)))
        if labels else [])
    if labels:
        lab_cols = [_COMPASS_STAGE_COLORS[k]["s"] for k in _COMPASS_DIR_ORDER]
        fig.add_trace(go.Scatterpolar(
            r=[dist / px_per_unit for dist, _, _ in placed],
            theta=_COMPASS_THETA, mode="text",
            text=list(labels), textposition=_COMPASS_LABEL_POS,
            textfont=dict(size=[font for _, _, font in placed], color=lab_cols,
                          family=_DONUT_FONT),
            cliponaxis=False, hoverinfo="skip", showlegend=False))

    annotations = ([_compass_legend(d, leg_names, count_label, stage_name,
                                    count_label_singular)]
                   if show_legend else [])
    if stats_text:
        if annotations:
            # Below the legend rows, left-aligned with them: the legend is one
            # annotation, so appending keeps the block attached regardless of
            # the legend's height.
            annotations[0]["text"] += (
                f"<br><br><span style='color:{_DONUT_MUTED}'>{stats_text}</span>")
        else:
            annotations.append(dict(
                text=stats_text, x=0.53, y=0.5, xref="paper", yref="paper",
                showarrow=False, xanchor="left", yanchor="middle", align="left",
                font=dict(size=13, family=_DONUT_FONT, color=_DONUT_MUTED),
            ))
    if ok_mode != "both" and not show_legend and total > 0:
        # The disc cannot say what green means: caption just below the lowest
        # drawn part (dashed circle, arm or label).
        low = [_COMPASS_R0 * px_per_unit] + [-box[1] for _, box, _ in placed]
        for ang, arm in zip(_COMPASS_THETA, arms):
            sin_max = max(-math.sin(math.radians(ang + da))
                          for da in (-_COMPASS_WID_DEG / 2, 0,
                                     _COMPASS_WID_DEG / 2))
            low.append(arm * px_per_unit * sin_max)
        bottom_px = min(max(low), radius_px)
        annotations.append(dict(
            text=(f"<span style='color:{_COMPASS_OK}'>●</span> "
                  f"<b>{ok_pct}</b> {leg_names['ok']}"),
            x=cx, y=cy - (bottom_px + 6) / plot_h, xref="paper", yref="paper",
            showarrow=False, xanchor="center", yanchor="top",
            font=dict(size=12, family=_DONUT_FONT, color="#3A4150"),
        ))
    polar_x = [0.0, 0.44] if show_legend else [0.06, 0.94]
    title_text = None
    if title:
        title_text = f"<b>{title}</b>"
        if total:
            title_text += (f"<br><span style='font-size:12px;color:"
                           f"{_DONUT_MUTED}'>{int(round(total))} "
                           f"{_unit(total)}</span>")
    fig.update_layout(
        title_text=title_text,
        title_font=dict(size=20, family=_DONUT_FONT), title_x=0.5,
        template="plotly_white", font=dict(family=_DONUT_FONT), height=height,
        polar=dict(
            domain=dict(x=polar_x, y=[0.06, 0.88 if title else 0.96]),
            bgcolor="white",
            radialaxis=dict(range=[0, r_range], visible=False),
            angularaxis=dict(visible=False, rotation=0, direction="counterclockwise"),
        ),
        annotations=annotations,
        showlegend=False,
        dragmode=False,   # the glyph is not meant to be zoomed/panned
        modebar=dict(remove=["zoom", "pan", "select", "lasso", "zoomin",
                             "zoomout", "autoscale", "resetscale"]),
        margin=dict(l=20, r=20, t=66 if title else 30, b=20),
    )
    if width:
        fig.update_layout(width=width)
    return fig


# Colours used by the Streamlit overheating page.
_OVERHEAT_ROOM_COLOR = "#0D7377"        # room-temperature line (teal)
_OVERHEAT_THRESHOLD_COLOR = "#f59e0b"   # comfort threshold (amber)
_OVERHEAT_MARKER_COLOR = "#ef4444"      # overheating samples (red)

_MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_overheating_timeseries(
    aligned: pd.DataFrame,
    method: str = "adaptive",
    summer_only: bool = True,
    business_hours_only: bool = False,
    title: str = "Overheating — temperature curve",
    ylab: str = "Temperature [°C]",
    height: int = 400,
) -> go.Figure:
    """Room-temperature curve with the comfort threshold and overheating marks.

    Mirrors the temperature plot of the Streamlit overheating page: the room
    temperature as a line, the (adaptive or fixed) comfort threshold as a
    dashed line, and the samples above the threshold highlighted as red
    markers.

    Args:
        aligned: DataFrame indexed by timestamp with columns
            ``[t_room, t_oa, t_oa_48h]`` (see
            :func:`pyedautils.comfort.align_hourly`).
        method: ``"adaptive"`` (SIA 180 curve) or ``"fixed"`` (26.5 °C),
            passed to :func:`pyedautils.comfort.overheating_hours`.
        summer_only: Restrict the threshold/markers to the SIA 180 summer
            half-year. Default ``True``.
        business_hours_only: Restrict to 07:00–22:00. Default ``False``.
        title: Plot title.
        ylab: Y-axis label.
        height: Figure height in pixels. Default 400.

    Returns:
        go.Figure
    """
    _, threshold = overheating_hours(
        aligned, method=method, summer_only=summer_only,
        business_hours_only=business_hours_only,
    )
    thr_name = ("SIA 180 limit" if method == "adaptive"
                else f"{FIXED_OVERHEATING_THRESHOLD:.1f} °C")

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=aligned.index, y=aligned["t_room"],
        mode="lines", line=dict(color=_OVERHEAT_ROOM_COLOR, width=1.4),
        name="Room temperature",
        hovertemplate="%{x}<br>T_room = %{y:.1f} °C<extra></extra>",
    ))
    if not threshold.empty:
        # The threshold is defined only inside the SIA 180 summer half-year (and,
        # if business_hours_only, only 07:00-22:00). Reindex it onto the full
        # timeline so the gaps (winter, nights) become NaN: with connectgaps=False
        # the dashed line then breaks at the gaps instead of being drawn straight
        # across them (e.g. connecting one summer's end to the next summer's start).
        thr_line = threshold.reindex(aligned.index)
        fig.add_trace(go.Scattergl(
            x=thr_line.index, y=thr_line.values,
            mode="lines", connectgaps=False,
            line=dict(color=_OVERHEAT_THRESHOLD_COLOR, width=1.4, dash="dash"),
            name=thr_name,
            hovertemplate="%{x}<br>Limit = %{y:.1f} °C<extra></extra>",
        ))
        room_in_window = aligned["t_room"].reindex(threshold.index)
        over = room_in_window.where(room_in_window > threshold)
        fig.add_trace(go.Scattergl(
            x=over.index, y=over.values,
            mode="markers",
            marker=dict(color=_OVERHEAT_MARKER_COLOR, size=4, opacity=0.7),
            name="Overheating",
            hovertemplate="%{x}<br>T_room = %{y:.1f} °C<extra></extra>",
        ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        height=height,
        yaxis_title=ylab,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_overheating_bar(
    monthly: pd.DataFrame,
    month_col: str = "month",
    hours_col: str = "hours",
    title: str = "Overheating hours per month",
    ylab: str = "Hours [h]",
    color: str = _OVERHEAT_THRESHOLD_COLOR,
    height: int = 400,
) -> go.Figure:
    """Bar chart of overheating hours per calendar month.

    Mirrors the monthly bar of the Streamlit overheating page: one amber bar
    per month (Jan–Dec, missing months shown as zero) with the hour count
    labelled above each bar.

    Args:
        monthly: DataFrame with a month column (1–12) and an hours column,
            e.g. the output of
            :func:`pyedautils.comfort.overheating_per_month`.
        month_col: Month column name (values 1–12). Default ``"month"``.
        hours_col: Overheating-hours column name. Default ``"hours"``.
        title: Plot title.
        ylab: Y-axis label.
        color: Bar colour.
        height: Figure height in pixels. Default 400.

    Returns:
        go.Figure
    """
    full = pd.DataFrame({"month": range(1, 13)})
    if not monthly.empty:
        m = monthly[[month_col, hours_col]].rename(
            columns={month_col: "month", hours_col: "hours"})
        full = full.merge(m, on="month", how="left")
    else:
        full["hours"] = 0.0
    full["hours"] = pd.to_numeric(full["hours"], errors="coerce").fillna(0.0)
    full["label"] = [_MONTH_LABELS[m - 1] for m in full["month"]]

    fig = go.Figure(go.Bar(
        x=full["label"], y=full["hours"],
        marker_color=color,
        text=[f"{int(h)}h" if h > 0 else "" for h in full["hours"]],
        textposition="outside",
        hovertemplate="%{x}: %{y:.0f} h<extra></extra>",
    ))
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        height=height,
        yaxis_title=ylab,
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# Temperature & humidity time-series colours.
_TS_TEMP_COLOR = "#E53935"     # red — temperature points
_TS_HUM_COLOR = "#2563EB"      # blue — humidity points
_TS_GREEN = "#4d7c0f"          # darker green — visible band edge lines + swatch
_TS_ORANGE = "#E67E22"         # orange warning limits
_TS_RED = "#C0392B"            # red critical limits


def plot_temp_humidity_timeseries(
    data: pd.DataFrame,
    temp_band: Optional[Tuple[float, float]] = (20.0, 26.0),
    hum_band: Optional[Tuple[float, float]] = (30.0, 65.0),
    temp_band_orange: Optional[Tuple[float, float]] = (19.0, 27.0),
    hum_band_orange: Optional[Tuple[float, float]] = (25.0, 70.0),
    temp_band_red: Optional[Tuple[float, float]] = (17.5, 28.5),
    hum_band_red: Optional[Tuple[float, float]] = (20.0, 75.0),
    show_hourly: bool = True,
    show_daily_mean: bool = True,
    temp_title: Optional[str] = None,
    hum_title: Optional[str] = None,
    title: str = "Temperature & humidity over time",
    height: int = 600,
) -> go.Figure:
    """Temperature and humidity time series with comfort and warning limits.

    Two stacked subplots (temperature on top, humidity below, shared time axis).
    Each shows the hourly readings as faint dots and the daily-mean as a bold
    line, plus — when given — the green **comfort target band** (filled, with a
    solid edge line top and bottom) and the **moderate** and **severe** warning
    limits as dashed threshold lines (the same levels used by the Mollier
    comfort zone and the kg-zh comfort compass). Hover labels use ``T`` / ``φ``
    and the ``%Y-%m-%d %H:%M`` date format of the Mollier h,x chart.

    The daily-mean line is reindexed onto a continuous daily grid, so gaps in
    the data (missing days) appear as **breaks** rather than straight connecting
    lines (``connectgaps=False``).

    Args:
        data: DataFrame with ``timestamp``, ``temperature`` [°C] and
            ``humidity`` [%rH] columns (read by name, order-independent).
        temp_band / hum_band: ``(low, high)`` comfort target band (green fill +
            solid edge lines). Defaults to the Mollier h,x comfort zone
            (T 20–26 °C, φ 30–65 %). ``None`` to omit.
        temp_band_orange / hum_band_orange: ``(low, high)`` moderate warning
            limits, drawn as dashed orange lines. Defaults to ± 1 K / ± 5 %
            around the comfort band. ``None`` to omit.
        temp_band_red / hum_band_red: ``(low, high)`` severe warning limits,
            drawn as dashed red lines. Defaults to ± 2.5 K / ± 10 % around the
            comfort band. ``None`` to omit.
        show_hourly: Overlay the raw readings as faint dots. Default ``True``.
        show_daily_mean: Draw the daily-mean line. Default ``True``.
        temp_title / hum_title: Optional title shown above the temperature /
            humidity subplot (e.g. ``"Temperature"``, ``"abs. Humidity"``).
            ``None`` to omit.
        title: Figure title.
        height: Figure height in pixels. Default 600.

    Returns:
        go.Figure
    """
    from plotly.subplots import make_subplots

    df = data.copy()
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df["timestamp"] = pd.to_datetime(df[ts_col])
    cols = [c for c in ("temperature", "humidity") if c in df.columns]
    df = df[["timestamp"] + cols].sort_values("timestamp")

    # Daily means on a continuous daily grid -> missing days become NaN so the
    # line breaks at gaps instead of being drawn straight across them.
    daily = None
    if show_daily_mean and not df.empty:
        daily = df.set_index("timestamp").resample("D").mean().reset_index()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=(temp_title or "", hum_title or ""))

    def _series(row, col, color, unit, fmt, sym, legend):
        if col not in df.columns:
            return
        name = col.capitalize()
        if show_hourly:
            fig.add_trace(go.Scattergl(
                x=df["timestamp"], y=df[col], mode="markers",
                marker=dict(color=color, size=5, opacity=0.25),
                name=f"{name} (hourly)", showlegend=False, legend=legend,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + sym
                + " = %{y:" + fmt + "} " + unit + "<extra></extra>"),
                row=row, col=1)
        if daily is not None:
            fig.add_trace(go.Scattergl(
                x=daily["timestamp"], y=daily[col], mode="lines",
                line=dict(color="black", width=1.8), connectgaps=False,
                name=name, legend=legend,
                hovertemplate="%{x|%Y-%m-%d}<br>" + sym
                + " = %{y:" + fmt + "} " + unit + "<extra></extra>"),
                row=row, col=1)
    _series(1, "temperature", _TS_TEMP_COLOR, "°C", ".1f", "T", "legend")
    _series(2, "humidity", _TS_HUM_COLOR, "%", ".0f", "φ", "legend2")

    def _limits(row, band, orange, red):
        if band:
            fig.add_hrect(y0=min(band), y1=max(band), row=row, col=1,
                          fillcolor="rgba(154,205,50,0.4)", line_width=0,
                          layer="below")
            for y in band:   # solid edge line top and bottom of the green band
                fig.add_hline(y=y, row=row, col=1, layer="below",
                              line=dict(color=_TS_GREEN, width=2))
        for lim, color in ((orange, _TS_ORANGE), (red, _TS_RED)):
            if lim:
                for y in lim:   # warning limits behind the data, like the band
                    fig.add_hline(y=y, row=row, col=1, layer="below",
                                  line=dict(color=color, width=2, dash="dash"))
    _limits(1, temp_band, temp_band_orange, temp_band_red)
    _limits(2, hum_band, hum_band_orange, hum_band_red)

    # Legend swatches — add_hrect / add_hline create no legend entries.
    # One legend per subplot, so each diagram carries its own band swatches.
    for legend, row, bnd, org, red in (
            ("legend", 1, temp_band, temp_band_orange, temp_band_red),
            ("legend2", 2, hum_band, hum_band_orange, hum_band_red)):
        for show, color, name, dash in (
                (bool(bnd), _TS_GREEN, "Comfort band", "solid"),
                (bool(org), _TS_ORANGE, "Moderate", "dash"),
                (bool(red), _TS_RED, "Severe", "dash")):
            if show:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="lines", name=name,
                    line=dict(color=color, width=2, dash=dash),
                    hoverinfo="skip", legend=legend), row=row, col=1)

    # No horizontal grid: the y grid lines would be drawn on top of the
    # (background) comfort band and warning lines and hide the ones that sit on
    # a grid value. The bands/limits provide the horizontal reference instead.
    fig.update_yaxes(showgrid=False)
    fig.update_yaxes(title_text="Temperature [°C]", row=1, col=1)
    fig.update_yaxes(title_text="Humidity [%rH]", row=2, col=1)
    fig.update_layout(
        title_text=f"<b>{title}</b>", title_font=dict(size=20), title_x=0.5,
        template="plotly_white", height=height,
        legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.01),
        legend2=dict(yanchor="top", y=0.47, xanchor="left", x=1.01),
        margin=dict(l=10, r=150, t=60, b=10),
    )
    return fig


def _load_d3_chart_js():
    """Load the bundled d3-mollierhx sources for the stateful MollierChart."""
    from importlib import resources as _res
    d = _res.files("pyedautils") / "data" / "d3_mollierhx"
    return "\n".join(
        (d / f).read_text(encoding="utf-8")
        for f in ("mollierFunctions.js", "coordinateGenerator.js",
                  "drawComfort.js", "mollierChart.js")
    )


def plot_mollier_hx_fast(
    data: Optional[pd.DataFrame] = None,
    pressure: Optional[float] = None,
    altitude: Optional[float] = None,
    domain_x: Tuple[float, float] = (0.0, 0.020),
    domain_y: Tuple[float, float] = (-20.0, 50.0),
    comfort_zone: Optional[Dict[str, Tuple[float, float]]] = None,
    height: int = 700,
    convention: str = 'classical',
    highlight_latest: bool = True,
    highlight_color: Optional[str] = 'black',
    states: Optional[Sequence] = None,
    labels: Optional[List[str]] = None,
    comfort_label: str = 'Comfort zone',
    comfort_zone_orange: Optional[Dict[str, object]] = None,
    comfort_zone_red: Optional[Dict[str, object]] = None,
    show_temperature: bool = True,
    show_density: bool = True,
    show_rel_humidity: bool = True,
    show_enthalpy: bool = True,
    show_abs_humidity: bool = True,
    x_axis_title: str = 'absolute water content x [g/kg]',
    season_labels: Optional[Dict[str, str]] = None,
    show_frequency: bool = False,
    frequency_unit: str = 'hours',
    frequency_smoothing: float = 4.0,
    show_points: bool = True,
    mean_enthalpy: bool = False,
    mean_temperature: bool = False,
    outdoor: Optional[Tuple[float, float]] = None,
    current: Optional[Tuple[float, float]] = None,
    outdoor_color: str = '#2563eb',
    ventilation_line: bool = True,
    ventilation_note: bool = True,
    ventilation_threshold: float = 0.0003,
    captions: Optional[Dict[str, str]] = None,
    texts: Optional[Dict[str, str]] = None,
) -> str:
    """Stateful, fast variant of :func:`plot_mollier_hx`.

    Renders with the d3-mollierhx ``MollierChart`` class: the coordinate grid is
    drawn once and the overlays (measurement points, comfort zone, warning bands,
    frequency lines, process chain, room and outside point) update in place. The
    created chart is exposed on ``window["mollier_<uid>"]`` so an embedding app
    can call its update methods (``setData``, ``setComfort``, ``setFrequency``,
    ``setCurrent``, ``setOutdoor``, ``setTexts`` ...) without re-rendering — the
    basis for fast, interactive dashboards.

    Same inputs as :func:`plot_mollier_hx` (including ``altitude`` setting
    the pressure when ``pressure`` is not given, offset warning bands,
    ``"abs_humidity": None`` for no abs-humidity limit, ``outdoor`` /
    ``current`` with the ventilation line and note, and ``captions`` /
    ``texts``), plus:

    Args:
        show_points: Draw the measurement points. ``False`` keeps only the
            frequency contour lines. Default ``True``.
        mean_enthalpy: Draw the mean-enthalpy curve (h_m, binned by
            temperature). Default ``False``.
        mean_temperature: Draw the mean-temperature curve (t_m, binned by
            enthalpy). Default ``False``.

    Differences to :func:`plot_mollier_hx`: the chart has no legend, so
    ``comfort_label``, ``season_labels`` and the legend keys of ``texts``
    (``comfort_zone``, ``season_*``, ``room``) are accepted but not shown; the
    comfort zone is drawn only when ``comfort_zone`` is given. Data, comfort
    zone, bands and frequency lines are clipped to the axes; room and outside
    points outside the axes become edge arrows with their value.

    Returns:
        str: Self-contained HTML (loads D3 v7 from the CDN + an inline bundle).

    See Also:
        :func:`plot_mollier_hx` for the full-featured variant. How to read the
        diagram, comfort and saving energy:
        https://retomarek.github.io/pyedautils/guides/mollier_hx.html
    """
    import json
    import uuid

    from pyedautils._mollier import _check_convention
    convention = _check_convention(convention)
    pressure = _resolve_pressure(pressure, altitude)
    cid = "mollier_" + uuid.uuid4().hex[:8]
    caps, txt = _mollier_texts(captions, texts, x_axis_title, comfort_label,
                               season_labels)
    outdoor_pt = _point_arg(outdoor, "outdoor")
    current_pt = _point_arg(current, "current")
    freq_unit = str(frequency_unit).lower()
    if freq_unit not in ("hours", "days"):
        raise ValueError("frequency_unit must be 'hours' or 'days'")

    records, latest, latest_ts = [], None, None
    if data is not None and len(data) > 0:
        df = data.copy()
        df.columns = ["timestamp", "humidity", "temperature"]
        stamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for k, row in enumerate(df.itertuples(index=False)):
            ts, hum, temp = row[0], row[1], row[2]
            if hum is None or temp is None or pd.isna(hum) or pd.isna(temp):
                continue
            rec = {
                "t": float(temp), "rh": float(hum),
                "time": None if (ts is None or pd.isna(ts)) else str(ts),
            }
            records.append(rec)
            st = stamps.iloc[k]
            if not pd.isna(st) and (latest_ts is None or st > latest_ts):
                latest, latest_ts = rec, st

    chain = []
    if states:
        for i, s in enumerate(states):
            chain.append({
                "x": float(s.x), "y": float(s.y), "number": str(i + 1),
                "label": str(labels[i]) if labels and i < len(labels) else str(i + 1),
                "t": round(float(s.t), 1), "phi": round(float(s.phi) * 100, 1),
                "xg": round(float(s.x) * 1000, 2), "h": round(float(s.h), 1),
            })

    def _band(b):
        if not b:
            return None
        return {"t": b["t"], "phi": b["phi"], "x": b["x"], "color": b["color"]}

    # Offsets of the warning bands refer to the comfort zone (its defaults
    # when ``comfort_zone`` is None, i.e. the zone itself is not drawn here).
    cz = None if comfort_zone is False else _comfort_ranges(comfort_zone)
    band_o, band_r = _resolve_bands(comfort_zone_orange, comfort_zone_red, cz)

    # Room point: an explicit ``current`` reading, else the latest row.
    room, room_color_js = None, json.dumps(highlight_color or "black")
    if current_pt is not None:
        room = current_pt
    elif highlight_latest and latest is not None:
        room = {"t": latest["t"], "rh": latest["rh"]}
        txt["now"] = txt["latest"] + ": " + latest_ts.strftime("%Y-%m-%d %H:%M")
        if highlight_color is None:   # season colour of the chart's palette
            room_color_js = ("c.seasonColors[seasonIndex(toDate("
                             + json.dumps(latest["time"]) + "))]")

    chart_texts = dict(txt)
    if altitude is not None:   # MollierChart derives it from the pressure
        chart_texts["altitude"] = _fill(txt["altitude"], alt=f"{altitude:.0f}")

    opts = {
        "height": int(height), "convention": convention, "pressure": float(pressure),
        "domainX": list(domain_x), "domainY": list(domain_y),
        "lineOpts": {
            "showTemperature": bool(show_temperature), "showDensity": bool(show_density),
            "showRelHumidity": bool(show_rel_humidity), "showEnthalpy": bool(show_enthalpy),
            "showAbsHumidity": bool(show_abs_humidity), "xAxisTitle": caps["x_axis_title"],
        },
        "captions": caps,
        "texts": chart_texts,
        "seasonLabels": [txt["season_" + k] for k in ("winter", "spring", "summer", "fall")],
    }
    freq = {"show": bool(show_frequency), "days": (freq_unit == "days"),
            "smoothing": float(frequency_smoothing)}

    calls = ["var c=new MollierChart('#" + cid + "'," + _js(opts) + ");"]
    if records:
        calls.append("c.setData(" + json.dumps(records) + ");")
    if chain:
        calls.append("c.setProcessChain(" + _js(chain) + ");")
    calls.append("c.setComfort(" + json.dumps(
        _comfort_ranges(comfort_zone) if comfort_zone else None) + ");")
    calls.append("c.setBands(" + json.dumps(_band(band_o))
                 + "," + json.dumps(_band(band_r)) + ");")
    calls.append("c.setFrequency(" + json.dumps(freq) + ");")
    if not show_points:
        calls.append("c.setShowPoints(false);")
    if mean_enthalpy or mean_temperature:
        calls.append("c.setMeanLines(" + json.dumps(
            {"h": bool(mean_enthalpy), "t": bool(mean_temperature)}) + ");")
    if room is not None:
        calls.append("c.setCurrent(" + json.dumps(room)
                     + ",{color:" + room_color_js + "});")
    if outdoor_pt is not None:
        calls.append("c.setOutdoor(" + json.dumps(outdoor_pt) + "," + _js({
            "color": outdoor_color, "ventilationLine": bool(ventilation_line),
            "note": bool(ventilation_note),
            "threshold": abs(float(ventilation_threshold))}) + ");")
    calls.append("window['" + cid + "']=c;")

    js = _load_d3_chart_js()
    return (
        '<div id="' + cid + '" style="width:100%"></div>\n'
        '<script src="' + _D3_URL + '"></script>\n'
        '<script>' + js + '</script>\n'
        '<script>(function(){' + " ".join(calls) + '})();</script>'
    )
