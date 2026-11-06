"""Render the repository's brand images with headless Chromium (Playwright).

- docs/images/branding_*.png: three example plots made with pyedautils and the
  bundled sample data (Mollier h,x diagram, comfort compass, energy signature),
  used by the social preview.
- docs/social-preview.png (1280x640) from scripts/social-preview.html. Upload it
  manually under the repository's Settings -> General -> Social preview (there is
  no API for it).
- docs/banner.png (1280x320) from scripts/banner.html, the README header.
- docs/images/favicon.png (192x192) from docs/logo.svg and docs/images/logo-docs.png
  (sidebar logo, from scripts/docs-logo.html), for the Jupyter Book.

docs/logo.svg is generated separately by scripts/make_logo.py.

Needs network access (plotly.js, D3 and the Manrope font come from CDNs). Run
from the repository root with uv (no Node needed):

    uv run --no-project --with playwright python -m playwright install chromium   # once
    uv run --no-project --with playwright --with-editable . python scripts/render_branding.py
"""

import tempfile
from importlib import resources
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

from pyedautils.comfort import comfort_compass_distribution
from pyedautils.plots import plot_comfort_compass, plot_energy_signature_pes, plot_mollier_hx

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DATA = resources.files("pyedautils") / "data"


def plot_pages():
    """HTML pages of the example plots, as (name, html, viewport width, height)."""
    room = pd.read_csv(DATA / "room_climate_sample.csv", parse_dates=["timestamp"])

    mollier = plot_mollier_hx(data=room, domain_x=(0.0, 0.013), domain_y=(0.0, 32.0),
                              comfort_zone={"temperature": (8, 18), "rel_humidity": (0.40, 0.70),
                                            "abs_humidity": (0, 0.0115)},
                              height=560)

    dist = comfort_compass_distribution(room.set_index("timestamp"), temp_band=(8, 20), hum_band=(35, 70))
    compass = plot_comfort_compass(dist, show_legend=False, fixed_scale=False, height=420)
    compass.update_layout(width=420, margin=dict(l=10, r=10, t=10, b=10))

    pes = pd.read_csv(DATA / "bldg_engy_sig_pes.csv")
    signature = plot_energy_signature_pes(pes, title="Proposed Energy Signature")
    signature.update_layout(width=640, height=440, showlegend=False, margin=dict(l=70, r=30, t=70, b=60))

    return [
        ("branding_mollier_hx", mollier, 760, 580),
        ("branding_comfort_compass", compass.to_html(include_plotlyjs="cdn"), 420, 420),
        ("branding_energy_signature", signature.to_html(include_plotlyjs="cdn"), 640, 440),
    ]


def render_plots(browser, tmp):
    for name, html, width, height in plot_pages():
        src = Path(tmp) / f"{name}.html"
        src.write_text(html, encoding="utf-8")
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
        page.goto(src.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(800)
        page.screenshot(path=str(DOCS / "images" / f"{name}.png"))
        print(f"docs/images/{name}.png")
        page.close()


def render_template(browser, template, out, height):
    page = browser.new_page(viewport={"width": 1280, "height": height}, device_scale_factor=1)
    page.goto((ROOT / "scripts" / template).as_uri(), wait_until="networkidle")
    page.evaluate("document.fonts.ready")
    page.screenshot(path=str(DOCS / out))
    print(f"docs/{out}")
    page.close()


def render_docs_logo(browser):
    page = browser.new_page(viewport={"width": 260, "height": 76}, device_scale_factor=2)
    page.goto((ROOT / "scripts" / "docs-logo.html").as_uri(), wait_until="networkidle")
    page.evaluate("document.fonts.ready")
    page.screenshot(path=str(DOCS / "images" / "logo-docs.png"), omit_background=True)
    print("docs/images/logo-docs.png")
    page.close()


def render_favicon(browser, tmp):
    src = Path(tmp) / "favicon.html"
    src.write_text(f'<body style="margin:0"><img src="{(DOCS / "logo.svg").as_uri()}" width="192" height="192"></body>',
                   encoding="utf-8")
    page = browser.new_page(viewport={"width": 192, "height": 192}, device_scale_factor=1)
    page.goto(src.as_uri())
    page.wait_for_timeout(200)
    page.screenshot(path=str(DOCS / "images" / "favicon.png"), omit_background=True)
    print("docs/images/favicon.png")
    page.close()


with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
    browser = p.chromium.launch(args=["--hide-scrollbars", "--allow-file-access-from-files"])
    try:
        render_plots(browser, tmp)
        render_favicon(browser, tmp)
        render_docs_logo(browser)
        render_template(browser, "social-preview.html", "social-preview.png", 640)
        render_template(browser, "banner.html", "banner.png", 320)
    finally:
        browser.close()
