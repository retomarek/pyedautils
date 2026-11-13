"""Colour palettes for :func:`pyedautils.plots.comfort.plot_comfort_compass`.

1:1 port of ``src/palettes.js`` of retomarek/comfort-compass-card (the Home
Assistant card), so the plot and the card offer the same palettes and derive
the same colours. Keep the two in sync.

Every palette keeps the meaning of the glyph: warm is reddish at the top,
cold bluish at the bottom, humid on the right, dry on the left, the mixtures
in between, and "in range" stands out in the centre. A palette gives one base
colour per direction; the three severity stages are derived from it in OKLCH
(same hue, stepped lightness), so mild -> severe always reads as a ramp. A
palette may give separate ``dark`` and ``light`` variants; ``dark=True``
picks the dark one.

Ramps: ``"down"`` (default) = mild light -> severe dark, for most themes;
``"up"`` = mild dim -> severe bright, for glowing colours on dark backgrounds.
"""

import math
from functools import lru_cache
from typing import Dict

_DIRS = ("f", "wf", "w", "wt", "t", "ct", "c", "kf")
_STAGES = ("l", "d", "s")

# --- colour maths (sRGB <-> OKLab / OKLCH) ----------------------------------


def _to_lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _to_gam(c: float) -> float:
    # Negative (out-of-gamut) values take the linear branch, as in the card.
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * c ** (1 / 2.4) - 0.055


def _hex_to_rgb(hex_: str):
    h = hex_.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    n = int(h, 16)
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]


def _js_round(v: float) -> int:
    """Math.round (half up), not Python's banker's rounding."""
    return int(math.floor(v + 0.5))


def _rgb_to_hex(rgb) -> str:
    return "#" + "".join(
        f"{_js_round(min(1.0, max(0.0, v)) * 255):02x}" for v in rgb)


def _cbrt(v: float) -> float:
    return math.copysign(abs(v) ** (1 / 3), v)


def _rgb_to_oklch(rgb):
    r, g, b = (_to_lin(v) for v in rgb)
    l_ = _cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = _cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = _cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    L = 0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_
    return L, math.hypot(A, B), math.atan2(B, A)


def _oklch_to_rgb(L: float, C: float, H: float):
    A, B = C * math.cos(H), C * math.sin(H)
    l_ = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3
    m_ = (L - 0.1055345942 * A - 0.0638541728 * B) ** 3
    s_ = (L - 0.0894841775 * A - 1.291485548 * B) ** 3
    lin = [
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.707614701 * s_,
    ]
    return [_to_gam(v) for v in lin]


def _in_gamut(rgb) -> bool:
    return all(-1e-4 <= v <= 1 + 1e-4 for v in rgb)


def _lch(L: float, C: float, H: float) -> str:
    """OKLCH -> hex, lowering the chroma until the colour fits into sRGB."""
    L = min(0.98, max(0.12, L))
    rgb = _oklch_to_rgb(L, C, H)
    i = 0
    while i < 30 and not _in_gamut(rgb):
        C *= 0.92
        rgb = _oklch_to_rgb(L, C, H)
        i += 1
    return _rgb_to_hex(rgb)


def lightness(hex_: str) -> float:
    """OKLCH lightness (0..1) of a hex colour."""
    return _rgb_to_oklch(_hex_to_rgb(hex_))[0]


def _luminance(hex_: str) -> float:
    r, g, b = (_to_lin(v) for v in _hex_to_rgb(hex_))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two hex colours."""
    x, y = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (x + 0.05) / (y + 0.05)


INK_DARK = "#1f2733"
INK_LIGHT = "#ffffff"


def ink_for(hex_: str) -> str:
    """Text colour (white or dark ink) with the better contrast on a fill."""
    return (INK_LIGHT if contrast(hex_, INK_LIGHT) >= contrast(hex_, INK_DARK)
            else INK_DARK)


# Background the label contrast is checked against (Home Assistant's default
# dark / light card background, also the plot background).
BG = {"dark": "#1c1c1c", "light": "#ffffff"}


def _readable(hex_: str, dark: bool) -> str:
    """*hex_* stepped in lightness until it reaches WCAG AA (4.5:1) on the
    background."""
    bg = BG["dark"] if dark else BG["light"]
    L, C, H = _rgb_to_oklch(_hex_to_rgb(hex_))
    out = hex_
    i = 1
    while i <= 20 and contrast(out, bg) < 4.5:
        out = _lch(L + (0.025 if dark else -0.025) * i, C, H)
        i += 1
    return out


def _ramp(base: str, direction: str = "down", spread: float = 1.0):
    """Three stages from one base colour; *spread* scales the lightness steps."""
    L, C, H = _rgb_to_oklch(_hex_to_rgb(base))
    if direction == "up":
        return {"l": _lch(L - 0.24 * spread, C * 0.6, H),
                "d": _lch(L - 0.11 * spread, C * 0.85, H), "s": base}
    return {"l": _lch(L + 0.13 * spread, C * 0.75, H), "d": base,
            "s": _lch(L - 0.13 * spread, C * 1.02, H)}


# --- palettes ----------------------------------------------------------------
#
# ok: "in range" colour. Directions: w warm, wf warm + humid, f humid,
# kf cold + humid, c cold, ct cold + dry, t dry, wt warm + dry.

DEFAULT_PALETTE = "default"

_PALETTES: Dict[str, dict] = {
    # The original look: soft, clear, works on light and dark backgrounds.
    "default": {
        "ok": "#22C55E",
        "okInk": "#ffffff",
        "stages": {
            "w": {"l": "#e49e93", "d": "#cf6e60", "s": "#b04336"},
            "wf": {"l": "#dc9dbb", "d": "#c66d9a", "s": "#a64278"},
            "f": {"l": "#baa7e2", "d": "#9a7ccf", "s": "#7954b3"},
            "kf": {"l": "#9eb0ea", "d": "#7289da", "s": "#4e63bf"},
            "c": {"l": "#82b9e5", "d": "#3d96d4", "s": "#0072af"},
            "ct": {"l": "#afb982", "d": "#8b955f", "s": "#69713d"},
            "t": {"l": "#d6a977", "d": "#bf7f29", "s": "#965d00"},
            "wt": {"l": "#e0a383", "d": "#cb7545", "s": "#ab4b00"},
        },
        "greys": {"l": "#b2b2b2", "d": "#8e8e8e", "s": "#6c6c6c"},
        # The original ink: dark text on the light "mild" stage, white otherwise.
        "stageInk": {"l": INK_DARK, "d": INK_LIGHT, "s": INK_LIGHT},
    },
    # Glowing colours: bright on dark backgrounds, deep and saturated on light ones.
    "neon": {
        "dark": {
            "ramp": "up", "ok": "#39FF14",
            "base": {"w": "#FF3366", "wf": "#FF2BD6", "f": "#B44CFF", "kf": "#7A6BFF",
                     "c": "#2E9BFF", "ct": "#00F0D0", "t": "#FFE600", "wt": "#FF8A00"},
        },
        "light": {
            "ok": "#00C853",
            "base": {"w": "#F0004E", "wf": "#D600AA", "f": "#9620F0", "kf": "#4F45FF",
                     "c": "#0080FF", "ct": "#00B3A1", "t": "#F2B800", "wt": "#FF6D00"},
        },
    },
    # Based on Okabe & Ito (2002), the classic colour-blind-safe set:
    # vermillion, orange, yellow, blue, sky blue and reddish purple; violet
    # and slate fill the gaps.
    "okabe_ito": {
        "ok": "#009E73",
        "base": {"w": "#D55E00", "wf": "#CC79A7", "f": "#9B5FC0", "kf": "#56B4E9",
                 "c": "#0072B2", "ct": "#8FA3AD", "t": "#F0E442", "wt": "#E69F00"},
    },
    # Paul Tol's "muted" scheme: colour-blind safe and calm.
    "tol_muted": {
        "ok": "#117733",
        "base": {"w": "#CC6677", "wf": "#882255", "f": "#AA4499", "kf": "#332288",
                 "c": "#6699CC", "ct": "#44AA99", "t": "#DDCC77", "wt": "#999933"},
    },
    # Strong, clearly separated colours with high contrast, for low vision and
    # bright rooms.
    "high_contrast": {
        "dark": {
            "ramp": "up", "spread": 1.2, "ok": "#3DDC4A",
            "base": {"w": "#FF4D4D", "wf": "#FF5CC8", "f": "#C77DFF", "kf": "#8C8CFF",
                     "c": "#4DB8FF", "ct": "#2EE6D6", "t": "#FFD60A", "wt": "#FF9F1C"},
        },
        "light": {
            "spread": 1.2, "ok": "#008A00",
            "base": {"w": "#C00000", "wf": "#A0007A", "f": "#6A00C8", "kf": "#2A2AD4",
                     "c": "#0057B8", "ct": "#006B6B", "t": "#9A6A00", "wt": "#C04A00"},
        },
    },
    # Soft pastels, like chalk on paper.
    "pastel": {
        "spread": 0.75, "ok": "#7FD1A0",
        "base": {"w": "#F49C9C", "wf": "#EFA0CB", "f": "#C3AEF0", "kf": "#A6B6F4",
                 "c": "#93CBF2", "ct": "#9ED8CF", "t": "#F3D28E", "wt": "#F6B48F"},
    },
    # Material Design / Home Assistant's named colours.
    "material": {
        "ok": "#4CAF50",
        "base": {"w": "#F44336", "wf": "#E91E63", "f": "#9C27B0", "kf": "#3F51B5",
                 "c": "#2196F3", "ct": "#009688", "t": "#FFC107", "wt": "#FF9800"},
    },
    # Nord: arctic frost blues and aurora colours.
    "nord": {
        "ok": "#A3BE8C",
        "base": {"w": "#BF616A", "wf": "#B48EAD", "f": "#8C7FC0", "kf": "#5E81AC",
                 "c": "#81A1C1", "ct": "#8FBCBB", "t": "#EBCB8B", "wt": "#D08770"},
    },
    # Dracula (dark) and Alucard (light).
    "dracula": {
        "dark": {
            "ramp": "up", "ok": "#50FA7B",
            "base": {"w": "#FF5555", "wf": "#FF79C6", "f": "#BD93F9", "kf": "#8A9BEA",
                     "c": "#8BE9FD", "ct": "#7FE3C4", "t": "#F1FA8C", "wt": "#FFB86C"},
        },
        "light": {
            "ok": "#14710A",
            "base": {"w": "#CB3A2A", "wf": "#A3144D", "f": "#644AC9", "kf": "#3A5BA0",
                     "c": "#036A96", "ct": "#1F7A6E", "t": "#846E15", "wt": "#A34D14"},
        },
    },
    # Catppuccin: Mocha on dark backgrounds, Latte on light ones.
    "catppuccin": {
        "dark": {
            "ramp": "up", "ok": "#A6E3A1",
            "base": {"w": "#F38BA8", "wf": "#F5C2E7", "f": "#CBA6F7", "kf": "#B4BEFE",
                     "c": "#89B4FA", "ct": "#94E2D5", "t": "#F9E2AF", "wt": "#FAB387"},
        },
        "light": {
            "ok": "#40A02B",
            "base": {"w": "#D20F39", "wf": "#EA76CB", "f": "#8839EF", "kf": "#7287FD",
                     "c": "#1E66F5", "ct": "#179299", "t": "#DF8E1D", "wt": "#FE640B"},
        },
    },
    # Solarized accent colours, the same on light and dark.
    "solarized": {
        "ok": "#859900",
        "base": {"w": "#DC322F", "wf": "#D33682", "f": "#6C71C4", "kf": "#4A7ECB",
                 "c": "#268BD2", "ct": "#2AA198", "t": "#B58900", "wt": "#CB4B16"},
    },
    # Gruvbox: warm retro colours.
    "gruvbox": {
        "dark": {
            "ramp": "up", "ok": "#B8BB26",
            "base": {"w": "#FB4934", "wf": "#E8687A", "f": "#D3869B", "kf": "#A396B8",
                     "c": "#83A598", "ct": "#8EC07C", "t": "#FABD2F", "wt": "#FE8019"},
        },
        "light": {
            "ok": "#79740E",
            "base": {"w": "#9D0006", "wf": "#A8325A", "f": "#8F3F71", "kf": "#5D5A8C",
                     "c": "#076678", "ct": "#427B58", "t": "#B57614", "wt": "#AF3A03"},
        },
    },
    # Tokyo Night: deep blue night with neon accents.
    "tokyo_night": {
        "dark": {
            "ramp": "up", "ok": "#9ECE6A",
            "base": {"w": "#F7768E", "wf": "#FF7AC6", "f": "#BB9AF7", "kf": "#7AA2F7",
                     "c": "#7DCFFF", "ct": "#73DACA", "t": "#E0AF68", "wt": "#FF9E64"},
        },
        "light": {
            "ok": "#485E30",
            "base": {"w": "#8C4351", "wf": "#9A3D7C", "f": "#5A4A78", "kf": "#34548A",
                     "c": "#0F4B6E", "ct": "#33635C", "t": "#8F5E15", "wt": "#965027"},
        },
    },
    # Bauhaus: primary red, yellow and blue, a design classic.
    "bauhaus": {
        "ok": "#2F8F46",
        "base": {"w": "#D62828", "wf": "#B5226B", "f": "#6B3FA0", "kf": "#2F4FA8",
                 "c": "#1D4E9E", "ct": "#2D7F8C", "t": "#F2C12E", "wt": "#EE7A1A"},
    },
    # Vintage: faded print colours, like an old atlas.
    "vintage": {
        "ok": "#7A9A5B",
        "base": {"w": "#B5523B", "wf": "#A85A6E", "f": "#7E6B8F", "kf": "#5E6F8E",
                 "c": "#4F7A94", "ct": "#6E8B74", "t": "#C9A35B", "wt": "#C27A45"},
    },
    # Earth: forest, clay, sand and lake.
    "earth": {
        "ok": "#5E8C31",
        "base": {"w": "#A0442D", "wf": "#8B4C6A", "f": "#6A5A8E", "kf": "#4E6A8E",
                 "c": "#3E7CA6", "ct": "#5F8F7A", "t": "#D2A24C", "wt": "#C06C2E"},
    },
    # Sunset: coral, rose and violet over a cool sea.
    "sunset": {
        "ok": "#2BB673",
        "base": {"w": "#FF5E5B", "wf": "#F2587F", "f": "#A35CD1", "kf": "#5C6BD6",
                 "c": "#2E86C1", "ct": "#2AB7A9", "t": "#FFC145", "wt": "#FF8C42"},
    },
    # Minimal: only "in range" in colour, all deviations in neutral greys.
    "minimal": {
        "ok": "#22C55E", "okInk": "#ffffff",
        "base": {k: "#8A94A6" for k in _DIRS},
    },
}

PALETTE_NAMES = tuple(_PALETTES)


@lru_cache(maxsize=None)
def resolve_palette(name: str = DEFAULT_PALETTE, dark: bool = False) -> dict:
    """Resolved palette for a background.

    Returns ``{"ok", "ok_ink", "stages": {dir: {l, d, s}}, "greys": {l, d, s},
    "ink": {dir: {l, d, s}}, "labels": {dir: colour}}``. ``labels`` are the
    direction-label colours: the lightest stage on a dark background, the
    darkest on a light one, stepped until they reach WCAG AA (4.5:1).

    Raises:
        ValueError: If *name* is not one of :data:`PALETTE_NAMES`.
    """
    if name not in _PALETTES:
        raise ValueError(
            f"palette must be one of {', '.join(PALETTE_NAMES)}, got {name!r}")
    entry = _PALETTES[name]
    if "dark" in entry or "light" in entry:
        p = (entry.get("dark") if dark else entry.get("light")) \
            or entry.get("dark") or entry.get("light")
    else:
        p = entry
    direction = p.get("ramp", "down")
    spread = p.get("spread", 1.0)
    stages = p.get("stages") or {
        k: _ramp(p["base"][k], direction, spread) for k in _DIRS}
    greys = p.get("greys") or _ramp(
        "#b4b4b4" if direction == "up" else "#8e8e8e", direction, spread)
    ink = {k: {st: (p["stageInk"][st] if "stageInk" in p
                    else ink_for(stages[k][st])) for st in _STAGES}
           for k in _DIRS}
    labels = {}
    for k in _DIRS:
        by = sorted(_STAGES, key=lambda st: lightness(stages[k][st]))
        labels[k] = _readable(stages[k][by[-1] if dark else by[0]], dark)
    return {
        "ok": p["ok"],
        "ok_ink": p.get("okInk") or ink_for(p["ok"]),
        "stages": stages,
        "greys": greys,
        "ink": ink,
        "labels": labels,
    }
