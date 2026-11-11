#!/usr/bin/env python3
"""Draw the figure at the top of README.md from the table in docs/demo-page.md.

    python make_figure.py

Writes docs/images/page-time-light.svg and docs/images/page-time-dark.svg, and prints the sentence the figure's alt
text uses.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "docs", "demo-page.md")
OUT = os.path.join(HERE, "docs", "images")
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"text": "#0b0b0b", "muted": "#52514e", "bar": "#2a78d6"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "bar": "#3987e5"},
}
# (start of the row in the table, label in the figure)
ROWS = [
    ("| RTX 5080 Laptop | gpu |", "RTX 5080 Laptop, fp16"),
    ("| Intel NPU | npu |", "Intel NPU, int8 weights"),
    ("| CPU | cpu |", "CPU, int8 weights"),
]
TITLE = "d1-3B in Edge through WebNN: time to answer one question"
NOTE = "Milliseconds for the page's example questions, one laptop, 2026-11-09."
WIDTH, BAR, VALUE_ROOM = 860, 22, 120


def times():
    """Return [(label, low ms, high ms)], read from the demo page's table."""
    if not os.path.exists(SOURCE):
        sys.exit(f"make_figure: {SOURCE} not found")
    with open(SOURCE, encoding="utf-8") as f:
        lines = f.read().splitlines()
    out = []
    for row, label in ROWS:
        line = next((l for l in lines if l.startswith(row)), None)
        if line is None:
            sys.exit(f"make_figure: no row {row!r} in docs/demo-page.md")
        match = re.fullmatch(r"([\d.]+)–([\d.]+)", line.strip("|").split("|")[5].strip())
        if match is None:
            sys.exit(f"make_figure: no time range in the row {row!r}")
        out.append((label, match.group(1), match.group(2)))
    return out


def text(x, y, body, size, fill, weight="400"):
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" fill="{fill}">{body}</text>'


def draw(rows, theme, desc):
    c = THEMES[theme]
    unit = (WIDTH - VALUE_ROOM) / max(float(high) for _, _, high in rows)
    parts = [text(0, 28, TITLE, 20, c["text"], "600"), text(0, 52, NOTE, 14, c["muted"])]
    y = 88
    for label, low, high in rows:
        parts.append(text(0, y, label, 15, c["text"]))
        w = float(high) * unit
        parts.append(f'<rect x="0" y="{y + 8:g}" width="{w:g}" height="{BAR}" rx="4" fill="{c["bar"]}"/>')
        parts.append(text(w + 8, y + 8 + 16, f"{low} to {high} ms", 15, c["text"], "600"))
        y += 62
    height = y - 62 + 8 + BAR + 8
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:g}" width="{WIDTH}" height="{height:g}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
        f'<title id="t">{TITLE}</title>\n<desc id="d">{desc}</desc>\n' + "\n".join(parts) + "\n</svg>\n"
    )


def main():
    rows = times()
    desc = TITLE + ". " + " ".join(f"{label}: {low} to {high} ms." for label, low, high in rows)
    os.makedirs(OUT, exist_ok=True)
    for theme in THEMES:
        with open(os.path.join(OUT, f"page-time-{theme}.svg"), "w", encoding="utf-8") as f:
            f.write(draw(rows, theme, desc))
    print(desc)


if __name__ == "__main__":
    main()
