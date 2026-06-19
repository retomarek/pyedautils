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


def _load_d3_js():
    """Load bundled D3.js source files for the Mollier diagram."""
    from importlib import resources as _res
    d3_dir = _res.files("pyedautils") / "data" / "d3_mollier"
    return {
        "mollier_functions": (d3_dir / "mollier_functions.js").read_text(encoding="utf-8"),
        "coordinate_generator": (d3_dir / "CoordinateGenerator.js").read_text(encoding="utf-8"),
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


def _altitude_pressure_labels(pressure, altitude):
    """Top-right annotation strings ``(altitude_json, pressure_json)``.

    Pressure is shown in hPa. Altitude defaults to the value derived from the
    pressure via the inverse ISA barometric formula when not supplied.
    """
    import json
    if altitude is None:
        altitude = (1 - (pressure / 101325.0) ** (1 / 5.25588)) / 2.25577e-5
    return (json.dumps(f"{altitude:.0f} m ü.M."),
            json.dumps(f"{pressure / 100.0:.0f} hPa"))


def _band_js(zone, default_color):
    """Build (t, phi, x, has, color, label) JSON snippets for a dashed band.

    ``zone`` is a dict with absolute bounds — ``"temperature"`` (°C) and
    ``"rel_humidity"`` (0–1) tuples, plus optional ``"abs_humidity"`` (kg/kg;
    e.g. ``(0, 0.012)`` for a 12 g/kg ceiling), ``"color"`` and ``"label"``.
    Returns inert ``"false"`` snippets when ``zone`` is falsy.
    """
    import json
    if not zone:
        return ("[0,0]", "[0,0]", "[0,0]", "false",
                json.dumps(default_color), "null")
    t = json.dumps(list(zone.get("temperature", (20, 26))))
    phi = json.dumps(list(zone.get("rel_humidity", (0.30, 0.65))))
    # Wide default abs-humidity range so the band is bounded by T/φ only
    # (not clipped horizontally) unless the caller sets it explicitly.
    xx = json.dumps(list(zone.get("abs_humidity", (0, 0.030))))
    color = json.dumps(zone.get("color", default_color))
    label = json.dumps(zone["label"]) if zone.get("label") else "null"
    return (t, phi, xx, "true", color, label)


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
    let unitTxt = useDays ? " days/year outside" : " h/year outside";
    let hitG = freqHover.append("g").attr("clip-path", "url(#__SATCLIP_ID__)");
    hitG.selectAll("path").data(contours).enter().append("path")
      .attr("d", function (d) { return contourPath(d.c); })
      .attr("fill", "none").attr("stroke", "transparent").attr("stroke-width", 8)
      .style("pointer-events", "stroke").style("cursor", "crosshair")
      .on("mouseover mousemove", function (d) {
        freqTip.style("opacity", 1).style("background-color", "white").style("color", "black")
          .html(d.label + unitTxt)
          .style("left", (d3.event.pageX + 15) + "px").style("top", (d3.event.pageY - 40) + "px");
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
    pressure: float = 101325.0,
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
) -> str:
    """
    Create a Mollier h,x-diagram (psychrometric chart) as self-contained HTML.

    Uses D3.js for fast SVG rendering with iso-lines for temperature, enthalpy,
    relative humidity and density, a comfort zone, and optional measured data
    points colour-coded by season with interactive hover tooltips.

    Args:
        data: Optional DataFrame with columns [timestamp, humidity, temperature].
            humidity in %, temperature in °C.
        pressure: Air pressure in Pa. Default 101325 (sea level). Shown in
            the top-right corner of the chart in hPa.
        altitude: Optional altitude in metres above sea level, shown together
            with the pressure in the top-right corner. When ``None``, it is
            derived from ``pressure`` via the inverse ISA barometric formula.
        domain_x: Range of absolute humidity [kg/kg] for the x-axis.
        domain_y: Range of the y-coordinate (≈ temperature at x=0) for the y-axis.
        comfort_zone: Dict with keys "temperature", "rel_humidity", "abs_humidity",
            each a (min, max) tuple. Defaults: T=[20, 26], phi=[0.30, 0.65],
            x=[0, 0.0115]. Pass ``False`` to disable the comfort zone.
        height: Diagram height in pixels. Default 700.
        convention: Mollier coordinate convention. ``'classical'`` (default)
            normalises enthalpy per kg of dry air (Recknagel/Sprenger style,
            isotherms tilt slightly up with x). ``'glueck'`` normalises per
            kg of moist air (per the Glück book, isotherms tilt slightly down).
        highlight_latest: When ``True`` (default), the row with the most recent
            timestamp is overlaid as a circle on top of the seasonal scatter
            points. Set to ``False`` to disable.
        highlight_color: CSS colour used for the latest-row overlay. Default
            ``'black'``. Pass ``None`` to use that row's season colour (so the
            highlight is visible only by size/order, not by colour).
        states: Optional sequence of ``MoistAirState`` objects (from
            :func:`pyedautils._mollier.state`). When supplied, each state is
            drawn as a numbered point and consecutive states are connected by
            arrows — a psychrometric process chain in the spirit of
            psychrosim.com. Works alongside ``data`` (historical scatter is
            drawn underneath).
        labels: Optional sequence of strings used as point labels (default
            is the index ``"0"``, ``"1"``, …). Length must match ``states``.
        comfort_label: Legend caption for the comfort-zone polygon. Default
            ``"Comfort zone"``. Override e.g. to ``"Komfortzone"`` for German
            output or to a custom designation like ``"DIN EN 16798 Cat II"``.
        comfort_zone_orange: Optional second zone drawn as a **dashed orange
            outline** (no fill). Same structure as ``comfort_zone`` — a dict
            with absolute bounds ``"temperature"`` (°C) and ``"rel_humidity"``
            (0–1) tuples, plus optional ``"abs_humidity"`` (kg/kg, defaults to
            an unclipped wide range so the band follows the T/φ bounds),
            ``"label"`` (legend caption) and ``"color"`` (CSS colour). Pass the
            absolute limits, e.g. ``{"temperature": (19, 27),
            "rel_humidity": (0.25, 0.70)}`` for a ±1 K / ±5 % band (a
            temperature *difference* is expressed in kelvin, hence "K").
        comfort_zone_red: Like ``comfort_zone_orange`` but drawn as a **dashed
            red outline** — e.g. the ±2.5 K / ±10 % band.
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
            labels and the ``"Enthalpy h [kJ/kg]"`` caption. Default
            ``True``. Set ``False`` to hide everything enthalpy-related.
        show_abs_humidity: Draw the vertical iso-lines of constant absolute
            humidity (at the bottom-axis tick values). Default ``True``.
        x_axis_title: Caption centred below the bottom axis. Default
            ``"absolute water content x [g/kg]"``. Pass ``""`` to omit it.
        season_labels: Optional mapping from the English season keys
            ``"Winter"``, ``"Spring"``, ``"Summer"``, ``"Fall"`` to the labels
            shown in the seasonal scatter tooltips and legend. Defaults to the
            English keys themselves. Pass
            ``pyedautils.plots._constants._SEASON_LABELS_DE`` (or any custom
            dict) for German output.
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

    Returns:
        str: Self-contained HTML string with inline D3.js rendering.
            Can be saved to a file, used with ``streamlit.components.v1.html()``,
            or displayed in a Jupyter notebook via ``IPython.display.HTML()``.
    """
    import json

    from pyedautils._mollier import (
        _check_convention,
        get_x_y,
        rel_humidity as m_rel_humidity,
        temperature as m_temperature,
    )

    convention = _check_convention(convention)
    js = _load_d3_js()

    # Season display labels — English keys by default; callers may pass a
    # mapping (e.g. _SEASON_LABELS_DE) for other languages.
    if season_labels is None:
        season_labels = {k: k for k in DEFAULT_SEASON_COLORS}

    # Prepare data JSON
    data_json = "null"
    current_json = "null"
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
            if highlight_latest:
                # Find the record with the most recent timestamp (don't assume
                # the input DataFrame is sorted).
                idx_latest = df["timestamp"].values.argmax()
                current_json = json.dumps(records[idx_latest])

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
        cz = comfort_zone or {}
        comfort_t = json.dumps(list(cz.get("temperature", (20, 26))))
        comfort_phi = json.dumps(list(cz.get("rel_humidity", (0.30, 0.65))))
        comfort_x = json.dumps(list(cz.get("abs_humidity", (0, 0.0115))))
        has_comfort_js = "true"
    (orange_t, orange_phi, orange_x, has_orange_js,
     orange_color_js, orange_label_js) = _band_js(comfort_zone_orange, "#E67E22")
    (red_t, red_phi, red_x, has_red_js,
     red_color_js, red_label_js) = _band_js(comfort_zone_red, "#C0392B")

    # Altitude / pressure annotation (top-right).
    info_alt_js, info_pressure_js = _altitude_pressure_labels(pressure, altitude)

    domain_x_js = json.dumps(list(domain_x))
    domain_y_js = json.dumps(list(domain_y))
    highlight_color_js = json.dumps(highlight_color)
    comfort_label_js = json.dumps(comfort_label)
    hx_opts_js = json.dumps({
        "showTemperature": bool(show_temperature),
        "showDensity": bool(show_density),
        "showRelHumidity": bool(show_rel_humidity),
        "showEnthalpy": bool(show_enthalpy),
        "showAbsHumidity": bool(show_abs_humidity),
        "xAxisTitle": x_axis_title,
    })

    import uuid
    uid = uuid.uuid4().hex[:8]
    diagram_id = f"mollier_{uid}"
    tooltip_id = f"tt_{uid}"
    plot_id = f"plot_{uid}"
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
        .replace("__SATCLIP_ID__", f"freqSat_{uid}")
        .replace("__TOOLTIP_ID__", tooltip_id)
    )

    season_colors = json.dumps(
        {season_labels[k]: c for k, c in DEFAULT_SEASON_COLORS.items()})
    # Legend entries (Spring, Summer, Fall, Winter) and the set of "dark"
    # season labels whose tooltip background needs white text.
    season_legend_js = json.dumps([
        {"label": season_labels[k], "color": DEFAULT_SEASON_COLORS[k]}
        for k in ("Spring", "Summer", "Fall", "Winter")
    ])
    dark_labels_js = json.dumps([season_labels["Winter"], season_labels["Fall"]])

    return f"""<div id="{diagram_id}" style="width:100%;background:white;"></div>
<div id="{tooltip_id}" style="position:absolute;background:rgba(255,255,255,0.9);\
border-radius:4px;padding:6px 8px;pointer-events:none;\
font-family:Tahoma,Geneva,sans-serif;font-size:11px;\
box-shadow:2px 2px 6px rgba(0,0,0,0.2);opacity:0;"></div>
<script src="https://d3js.org/d3.v5.min.js"></script>
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
  let statePoints = {states_json};
  let showStateLegend = {state_legend_js};
  let hasComfort = {has_comfort_js};
  let comfortLabel = {comfort_label_js};
  let infoAlt = {info_alt_js};
  let infoPressure = {info_pressure_js};
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
    .attr("width", Width).attr("height", Height);
  let bg = svg.append("g").attr("id", "{plot_id}");
  let defs = svg.append("defs");
  defs.append("svg:clipPath")
    .attr("id", "{clip_id}").append("svg:rect")
    .attr("width", width).attr("height", height);
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
    .attr("clip-path", "url(#{clip_id})");

  drawHXCoordinates(bg, Width, Height, margin, domainX, domainY, p, mollier, hxOpts);

  // Altitude / pressure annotation, top-right (dark grey).
  let infoG = svg.append("g")
    .style("font-family", "Tahoma, Geneva, sans-serif");
  infoG.append("text")
    .attr("x", margin.left + width).attr("y", margin.top - 22)
    .attr("text-anchor", "end").attr("fill", "#4d4d4d").attr("font-size", 12)
    .text(infoAlt);
  infoG.append("text")
    .attr("x", margin.left + width).attr("y", margin.top - 8)
    .attr("text-anchor", "end").attr("fill", "#4d4d4d").attr("font-size", 12)
    .text(infoPressure);

  let x = d3.scaleLinear().range([0, width]).domain(domainX);
  let y = d3.scaleLinear().range([height, 0]).domain(domainY);

  let line = d3.line().x(d => x(d.x)).y(d => y(d.y));
  let pathos = hasComfort ? createComfort(rangeT, rangePhi, rangeX, p, mollier) : null;
  if (hasComfort && pathos && pathos.length > 0) {{
    plot.append("path").datum(pathos).attr("d", line)
      .attr("fill", "yellowgreen").attr("fill-opacity", 0.4)
      .attr("stroke", "yellowgreen");
  }}

  // Extra warning bands: dashed outline only (no fill).
  function drawBand(rangeT, rangePhi, rangeX, color) {{
    let pb = createComfort(rangeT, rangePhi, rangeX, p, mollier);
    if (pb && pb.length > 0) {{
      plot.append("path").datum(pb).attr("d", line)
        .attr("fill", "none").attr("stroke", color)
        .attr("stroke-width", 1.8).attr("stroke-dasharray", "7,4");
    }}
  }}
  if (hasOrange) drawBand(orangeT, orangePhi, orangeX, orangeColor);
  if (hasRed) drawBand(redT, redPhi, redX, redColor);

  if (dataRecords && dataRecords.length > 0) {{
    let tooltip = d3.select("#{tooltip_id}");

    for (let i = dataRecords.length - 1; i > 0; i--) {{
      let j = Math.floor(Math.random() * (i + 1));
      [dataRecords[i], dataRecords[j]] = [dataRecords[j], dataRecords[i]];
    }}

    plot.selectAll("circle").data(dataRecords).enter().append("circle")
      .attr("cx", d => x(d.x)).attr("cy", d => y(d.y))
      .attr("r", 5).attr("fill", d => colorMap[d.season] || "#999")
      .attr("opacity", 0.4).attr("shape-rendering", "optimizeSpeed")
      .on("mouseover", function(d) {{
        d3.select(this).attr("r", 10).attr("opacity", 0.9);
        tooltip.style("opacity", 1)
          .style("background-color", colorMap[d.season] || "#999")
          .style("color",
            (darkLabels.indexOf(d.season) >= 0) ? "white" : "black")
          .html(d.ts + "<br>x: " + d.xg + " g/kg<br>T: " + d.temp
            + " °C<br>φ: " + d.phi + " %")
          .style("left", (d3.event.pageX + 15) + "px")
          .style("top", (d3.event.pageY - 40) + "px");
      }})
      .on("mouseout", function(d) {{
        d3.select(this).attr("r", 5).attr("opacity", 0.4);
        tooltip.style("opacity", 0);
      }});

    // Latest record drawn on top of all seasonal points.
    if (currentRecord) {{
      let latestFill = highlightColor !== null
        ? highlightColor
        : (colorMap[currentRecord.season] || "#999");
      plot.append("g").attr("id", "current-point")
        .append("circle")
          .datum(currentRecord)
          .attr("cx", x(currentRecord.x)).attr("cy", y(currentRecord.y))
          .attr("r", 6).attr("fill", latestFill)
          .attr("stroke", "white").attr("stroke-width", 1.5)
          .style("cursor", "pointer")
          .on("mouseover", function(d) {{
            d3.select(this).attr("r", 10);
            tooltip.style("opacity", 1)
              .style("background-color", latestFill)
              .style("color", "white")
              .html("Latest: " + d.ts
                + "<br>x: " + d.xg + " g/kg<br>T: " + d.temp
                + " °C<br>φ: " + d.phi + " %")
              .style("left", (d3.event.pageX + 15) + "px")
              .style("top", (d3.event.pageY - 40) + "px");
          }})
          .on("mouseout", function() {{
            d3.select(this).attr("r", 6);
            tooltip.style("opacity", 0);
          }});
    }}

  }}

  // Process chain: numbered state points joined by arrows.
  if (statePoints && statePoints.length > 0) {{
    let tooltip = d3.select("#{tooltip_id}");
    let chain = plot.append("g").attr("id", "process-chain");

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
      .on("mouseover", function(d) {{
        tooltip.style("opacity", 1)
          .style("background-color", "#222")
          .style("color", "white")
          .html(d.label
            + "<br>T: " + d.t + " °C"
            + "<br>φ: " + d.phi + " %"
            + "<br>x: " + d.xg + " g/kg"
            + "<br>h: " + d.h + " kJ/kg")
          .style("left", (d3.event.pageX + 15) + "px")
          .style("top", (d3.event.pageY - 40) + "px");
      }})
      .on("mouseout", function() {{ tooltip.style("opacity", 0); }});

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
  // ----- Combined legend (bottom-right) ----------------------------------
  let legendItems = [];
  if (statePoints && statePoints.length > 0 && showStateLegend) {{
    for (let sp of statePoints) {{
      legendItems.push({{type: "process", number: sp.number, label: sp.label}});
    }}
  }}
  if (hasComfort && pathos && pathos.length > 0) {{
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
      legendItems.push({{type: "circle", color: s.color, label: s.label}});
    }}
  }}

  if (legendItems.length > 0) {{
    let rowH = 20;
    let legendH = legendItems.length * rowH + 10;
    let legendW = 150;
    // Anchor bottom-right of the plot area, but keep enough margin from the
    // right and bottom edges so that the enthalpy-axis labels (which sit at
    // plot-local x = width-20 and y = height-20) stay visible.
    let lx = margin.left + width - legendW - 50;
    let ly = margin.top + height - legendH - 35;
    let legendG = svg.append("g")
      .attr("transform", "translate(" + lx + "," + ly + ")");
    legendG.append("rect")
      .attr("x", -6).attr("y", -6)
      .attr("width", legendW + 12).attr("height", legendH + 6)
      .attr("fill", "white").attr("stroke", "#bbb").attr("rx", 3);
    legendItems.forEach((item, i) => {{
      let g = legendG.append("g")
        .attr("transform", "translate(0," + (i * rowH + 5) + ")");
      if (item.type === "process") {{
        g.append("circle")
          .attr("cx", 10).attr("cy", 8).attr("r", 10)
          .attr("fill", "#222").attr("stroke", "white").attr("stroke-width", 1.5);
        g.append("text")
          .attr("x", 10).attr("y", 8).attr("text-anchor", "middle").attr("dy", "0.35em")
          .attr("fill", "white").attr("font-size", "10px").attr("font-weight", "bold")
          .text(item.number);
        g.append("text").attr("x", 26).attr("y", 12)
          .style("font-family", "Tahoma, Geneva, sans-serif").style("font-size", "12px")
          .text(item.label);
      }} else if (item.type === "rect") {{
        g.append("rect").attr("x", 2).attr("y", 1).attr("width", 14).attr("height", 14)
          .attr("fill", item.color).attr("opacity", 0.7);
        g.append("text").attr("x", 22).attr("y", 12)
          .style("font-family", "Tahoma, Geneva, sans-serif").style("font-size", "12px")
          .text(item.label);
      }} else if (item.type === "dashes") {{
        g.append("line").attr("x1", 1).attr("y1", 8).attr("x2", 17).attr("y2", 8)
          .attr("stroke", item.color).attr("stroke-width", 2)
          .attr("stroke-dasharray", "5,3");
        g.append("text").attr("x", 22).attr("y", 12)
          .style("font-family", "Tahoma, Geneva, sans-serif").style("font-size", "12px")
          .text(item.label);
      }} else {{
        g.append("circle").attr("cx", 9).attr("cy", 8).attr("r", 5)
          .attr("fill", item.color).attr("opacity", 0.7);
        g.append("text").attr("x", 22).attr("y", 12)
          .style("font-family", "Tahoma, Geneva, sans-serif").style("font-size", "12px")
          .text(item.label);
      }}
    }});
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
# dry=orange / humid=violet; mixed directions get the mean colour of their poles.
_COMPASS_C_WARM = "#E74C3C"
_COMPASS_C_COLD = "#3498DB"
_COMPASS_C_DRY = "#F39C12"
_COMPASS_C_WET = "#8E5BD9"
_COMPASS_OK = "#22C55E"
# Directions in drawing order, at THETA 0,45,...,315 (0 deg = East, CCW). With
# the default polar orientation this puts humid right, warm top, dry left,
# cold bottom — same axes as an h,x (Mollier) diagram.
_COMPASS_DIR_ORDER = ("f", "wf", "w", "wt", "t", "ct", "c", "kf")
_COMPASS_STAGE_ORDER = ("l", "d", "s")
_COMPASS_THETA = [0, 45, 90, 135, 180, 225, 270, 315]
_COMPASS_POLES = {
    "w": (_COMPASS_C_WARM,), "wf": (_COMPASS_C_WARM, _COMPASS_C_WET),
    "f": (_COMPASS_C_WET,), "kf": (_COMPASS_C_COLD, _COMPASS_C_WET),
    "c": (_COMPASS_C_COLD,), "ct": (_COMPASS_C_COLD, _COMPASS_C_DRY),
    "t": (_COMPASS_C_DRY,), "wt": (_COMPASS_C_WARM, _COMPASS_C_DRY),
}
_COMPASS_STAGE_MIX = {"l": 0.62, "d": 0.32, "s": 0.0}   # lighten toward white
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


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _compass_mix(hexes) -> Tuple[float, float, float]:
    """Mean of one or more hex colours in RGB (the mixed-direction colour)."""
    rgbs = [_hex_to_rgb(c) for c in hexes]
    return tuple(sum(ch) / len(ch) for ch in zip(*rgbs))


def _compass_rgb(rgb) -> str:
    return f"rgb({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f})"


def _compass_shade(rgb, stage: str) -> str:
    """Lighten an RGB tuple toward white by the stage amount -> css rgb()."""
    m = _COMPASS_STAGE_MIX[stage]
    return _compass_rgb(tuple(c + (255 - c) * m for c in rgb))


def _compass_legend(d, names, count_label) -> dict:
    """Donut-style bullet legend: 'in range' first, then the deviation
    directions present, sorted by count (biggest first). Each row shows the
    count and its share, lower-cased like the donut legend."""
    total = sum(float(v) for v in d.values())
    g = lambda k: float(d.get(k, 0.0))
    rows = [("ok", g("ok"), _COMPASS_OK)]
    deviations = []
    for k in _COMPASS_DIR_ORDER:
        cnt = sum(g(f"{k}_{s}") for s in _COMPASS_STAGE_ORDER)
        if cnt > 0:
            deviations.append((k, cnt, _compass_rgb(_compass_mix(_COMPASS_POLES[k]))))
    deviations.sort(key=lambda x: x[1], reverse=True)
    rows += deviations
    lines = []
    for k, cnt, col in rows:
        pct = f"{cnt / total * 100:.0f}%" if total else "0%"
        lines.append(
            f"<span style='color:{col}'>●</span>  {names[k]} — "
            f"<b>{int(round(cnt))}</b> {count_label} "
            f"<span style='color:{_DONUT_MUTED}'>({pct})</span>"
        )
    return dict(
        text="<br>".join(lines), x=0.46, y=0.5, xref="paper", yref="paper",
        showarrow=False, xanchor="left", yanchor="middle", align="left",
        font=dict(size=13, family=_DONUT_FONT, color="#3A4150"),
    )


def plot_comfort_compass(
    distribution,
    title: Optional[str] = None,
    show_direction_labels: bool = True,
    show_legend: bool = True,
    count_label: str = "days",
    direction_labels: Optional[List[str]] = None,
    names: Optional[Dict[str, str]] = None,
    stage_names: Optional[Dict[str, str]] = None,
    height: int = 460,
) -> go.Figure:
    """Area-true "comfort compass" glyph for one room / group.

    A single equal-area polar glyph summarising how a room's days split across
    the nine comfort states (see
    :func:`pyedautils.comfort.comfort_compass_distribution`):

    - **Centre** — a green disc whose *area* is the share of days *in range*
      (the percentage is printed inside).
    - **Eight wedges** — one per deviation direction; the *wedge area* is the
      share of days, and the three radial sub-segments (light → strong, shaded
      light → saturated) split that share by severity. Orientation matches an
      h,x diagram: warm top, cold bottom, humid right, dry left; mixed
      directions use the mean colour of their two poles.
    - Every glyph has the **same total area** (= 100 % of days) — only the
      shape tells the story.

    The percentages are listed in a donut-style bullet legend on the right.

    Args:
        distribution: Mapping of comfort-compass categories to **counts**, as
            returned by :func:`pyedautils.comfort.comfort_compass_distribution`
            (keys ``"ok"`` and ``"<direction>_<stage>"``; missing keys count as
            0). Percentages and areas are derived from the counts.
        title: Figure title. Default *None*.
        show_direction_labels: Draw the eight direction labels around the rose.
            Default *True*.
        show_legend: Draw the bullet legend with the per-direction counts and
            percentages. Default *True*.
        count_label: Unit word for the legend counts (e.g. ``"days"``).
        direction_labels: Optional 8 labels for the rose (order: humid,
            warm+humid, warm, warm+dry, dry, cold+dry, cold, cold+humid). Use
            ``"<br>"`` for line breaks. Default English; pass
            ``pyedautils.plots.comfort._COMPASS_LABELS_DE`` for German.
        names: Optional mapping overriding the legend / hover direction names
            (keys ``"ok"`` and ``"f","wf","w","wt","t","ct","c","kf"``). Default
            English; pass ``_COMPASS_NAMES_DE`` for German.
        stage_names: Optional mapping overriding the severity-stage names shown
            in the hover (keys ``"l"`` mild, ``"d"`` moderate, ``"s"`` severe).
            Default English; pass ``_COMPASS_STAGE_NAMES_DE`` for German.
        height: Figure height in pixels. Default 520.

    All visible texts are overridable for localisation (``title``,
    ``count_label``, ``names``, ``direction_labels``, ``stage_names``), mirroring
    the ``season_labels`` pattern of :func:`plot_mollier_hx`.

    Returns:
        go.Figure
    """
    d = dict(distribution)
    g = lambda k: float(d.get(k, 0.0))
    leg_names = {**_COMPASS_NAMES, **(names or {})}
    total = sum(float(v) for v in d.values())
    frac = lambda k: (g(k) / total) if total else 0.0

    ok = frac("ok")
    r_green = _COMPASS_R0 * math.sqrt(max(ok, 0.0))

    stage_name = {**_COMPASS_STAGE_NAMES, **(stage_names or {})}

    def _hov(name, cnt):
        pct = cnt / total * 100 if total else 0.0
        return (f"<b>{name}</b><br>{int(round(cnt))} {count_label} "
                f"({pct:.0f}%)")

    # Stacked, area-true severity wedges per direction. A ring sector from
    # r_prev to r_next over WID has area (WID/2)(r_next^2 - r_prev^2); choosing
    # r_next so that the cumulative area equals (share * total area) makes every
    # wedge area proportional to its share of days.
    th, rr, base, wid, col, hov = [], [], [], [], [], []
    for ang, k in zip(_COMPASS_THETA, _COMPASS_DIR_ORDER):
        base_rgb = _compass_mix(_COMPASS_POLES[k])
        cum, r_prev = 0.0, r_green
        for st in _COMPASS_STAGE_ORDER:
            cnt = g(f"{k}_{st}")
            cum += frac(f"{k}_{st}")
            r_next = math.sqrt(r_green ** 2 + 2 * _COMPASS_AREA * cum / _COMPASS_WID_RAD)
            if r_next > r_prev + 1e-9:
                th.append(ang); base.append(r_prev); rr.append(r_next - r_prev)
                wid.append(_COMPASS_WID_DEG); col.append(_compass_shade(base_rgb, st))
                hov.append(_hov(f"{leg_names[k]} · {stage_name[st]}", cnt))
            r_prev = r_next

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
            marker=dict(color=_COMPASS_OK, line=dict(width=0)),
            hovertext=[_hov(leg_names["ok"], g("ok"))],
            hovertemplate="%{hovertext}<extra></extra>", showlegend=False))
    if th:
        fig.add_trace(go.Barpolar(
            r=rr, theta=th, base=base, width=wid,
            marker=dict(color=col, line=dict(color="white", width=0.8)),
            hovertext=hov, hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False))
    # Centre percentage — only when the green disc is large enough to hold it
    # legibly (below ~15 % it would overflow the disc); the value is in the legend.
    if ok * 100 >= 15:
        fig.add_trace(go.Scatterpolar(
            r=[0], theta=[0], mode="text", text=[f"{ok * 100:.0f}%"],
            textfont=dict(color="white", size=14, family=_DONUT_FONT),
            hoverinfo="skip", showlegend=False))
    if show_direction_labels:
        labels = direction_labels or _COMPASS_LABELS
        lab_cols = [_compass_rgb(_compass_mix(_COMPASS_POLES[k]))
                    for k in _COMPASS_DIR_ORDER]
        fig.add_trace(go.Scatterpolar(
            r=[_COMPASS_RMAX * 1.07] * 8, theta=_COMPASS_THETA, mode="text",
            text=labels,
            textfont=dict(size=11, color=lab_cols, family=_DONUT_FONT),
            hoverinfo="skip", showlegend=False))

    annotations = [_compass_legend(d, leg_names, count_label)] if show_legend else []
    polar_x = [0.0, 0.44] if show_legend else [0.06, 0.94]
    fig.update_layout(
        title_text=f"<b>{title}</b>" if title else None,
        title_font=dict(size=20, family=_DONUT_FONT), title_x=0.5,
        template="plotly_white", font=dict(family=_DONUT_FONT), height=height,
        polar=dict(
            domain=dict(x=polar_x, y=[0.06, 0.90 if title else 0.96]),
            bgcolor="white",
            radialaxis=dict(range=[0, _COMPASS_RMAX * 1.15], visible=False),
            angularaxis=dict(visible=False, rotation=0, direction="counterclockwise"),
        ),
        annotations=annotations,
        showlegend=False,
        dragmode=False,   # the glyph is not meant to be zoomed/panned
        modebar=dict(remove=["zoom", "pan", "select", "lasso", "zoomin",
                             "zoomout", "autoscale", "resetscale"]),
        margin=dict(l=20, r=20, t=60 if title else 30, b=20),
    )
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
