"""Generate docs/logo.svg: an energy signature (daily points and the fitted
heating line with its balance point) on the dark rounded tile shared with the
sister projects (retomarek/mollier-hx-card, retomarek/comfort-compass-card,
hslu-ige-laes/d3-mollierhx).

    python scripts/make_logo.py

Standard library only. Re-run scripts/render_branding.py afterwards.
"""

import random
from pathlib import Path

# plot area in the 512 x 512 icon
L, T, W, H = 84, 76, 352, 360
T_MIN, T_MAX, P_MAX = -12.0, 28.0, 11.0
TB, SLOPE, BASE = 15.0, 0.34, 0.9  # balance temperature, kW/K, standby + hot water


def sx(t):
    return L + (t - T_MIN) / (T_MAX - T_MIN) * W


def sy(p):
    return T + H - p / P_MAX * H


def power(t):
    return BASE + SLOPE * max(TB - t, 0.0)


rnd = random.Random(3)
dots = []
for i in range(24):
    t = T_MIN + 2 + (T_MAX - T_MIN - 4) * (i + rnd.random()) / 24
    p = power(t) + rnd.gauss(0, 0.75 if t < TB else 0.35)
    dots.append(f'<circle cx="{sx(t):.1f}" cy="{sy(max(p, 0.2)):.1f}" r="13" fill="#ffd43b" '
                'fill-opacity="0.92" stroke="#0c1520" stroke-width="4"/>')
rnd.shuffle(dots)

grid = "".join(f'<path d="M{L},{sy(p):.1f} H{L + W}" stroke="#63c1ff" stroke-opacity="0.35" stroke-width="3"/>'
               for p in (2.5, 5.0, 7.5))
line = f"M{sx(T_MIN):.1f},{sy(power(T_MIN)):.1f} L{sx(TB):.1f},{sy(BASE):.1f} L{sx(T_MAX):.1f},{sy(BASE):.1f}"
bx, by = sx(TB), sy(BASE)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <title>pyedautils</title>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#1d3350"/>
      <stop offset="1" stop-color="#0c1520"/>
    </linearGradient>
    <clipPath id="plot"><rect x="{L}" y="{T}" width="{W}" height="{H}"/></clipPath>
  </defs>
  <rect width="512" height="512" rx="112" fill="url(#bg)"/>
  <g clip-path="url(#plot)">
    {grid}
    {"".join(dots)}
    <path d="{line}" fill="none" stroke="#0c1520" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="{line}" fill="none" stroke="#63c1ff" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
  </g>
  <circle cx="{bx:.1f}" cy="{by:.1f}" r="17" fill="#0c1520"/>
  <circle cx="{bx:.1f}" cy="{by:.1f}" r="10.5" fill="#ffffff"/>
  <path d="M{L},{T} V{T + H} H{L + W}" fill="none" stroke="#ffffff" stroke-opacity="0.9" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
'''

out = Path(__file__).resolve().parent.parent / "docs" / "logo.svg"
out.write_text(svg, encoding="utf-8")
print(out)
