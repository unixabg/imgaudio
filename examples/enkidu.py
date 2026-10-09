#!/usr/bin/env python3
"""
enkidu.py — words from the Epic of Gilgamesh, written into sound.

Enkidu, Gilgamesh's wild friend, is shaped from clay by the goddess Aruru and
lives among the beasts before he ever meets a person. This script draws two
passages about him as pictures and encodes each into sound, so the words
travel as sound and come back as writing when decoded:

  examples/enkidu-clay.wav     Aruru shapes Enkidu from clay
  examples/enkidu-beasts.wav   Enkidu grazes with the gazelles

  python3 imgaudio.py --rows 300 --cols 600 decode examples/enkidu-clay.wav clay.png

In the web page's Sound mode they read best at Detail: Fine. What travels is
the shape of the writing, not speech: the sound is a chord whose notes trace
the letters.

Text: R. Campbell Thompson, "The Epic of Gilgamish" (Luzac, London, 1928;
preface dated 1927), the First Tablet, Column II. Public domain: published
before 1929 (US), and the translator died in 1941. The lines are as printed,
with Thompson's bracketed restorations kept.

Run from the repository root:  python3 examples/enkidu.py
"""

import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import imgaudio  # noqa: E402

PASSAGES = {
    "enkidu-clay": [
        "Finger'd some clay, on the desert she moulded (it): [(thus) on the desert]",
        "Enkidu made she, a warrior, (as he were) born (and) begotten,",
    ],
    "enkidu-beasts": [
        "E'en with gazelles did he pasture on herbage, along with the cattle",
        "Drank he his fill, with the beasts did his heart delight at the water.",
    ],
}

# Two lines of text fill the picture's height; the width gives each letter
# enough columns to survive at the web page's Fine detail (300 x 600).
ENCODE = dict(imgaudio.DEFAULTS, rows=300, cols=600, col_sec=0.035,
              auto_prep=False, color=False, lens="raw", gamma=1.0)


def font(size):
    for name in ("DejaVuSerif-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)   # Pillow 10.1+


def draw(lines, path):
    # The bottom of the picture is the lowest pitches, where the decoder
    # resolves pitch most coarsely (below about 300 Hz letters smear), so
    # the writing stays in the upper two thirds.
    f = font(56)
    widths = [f.getbbox(line)[2] for line in lines]
    w, h = max(widths) + 60, 300
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        d.text((30, 50 + i * 95), line, fill=255, font=f, anchor="lm")
    im.save(path)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for name, lines in PASSAGES.items():
            png = Path(tmp) / f"{name}.png"
            draw(lines, png)
            out = ROOT / "examples" / f"{name}.wav"
            imgaudio.encode(str(png), str(out), **ENCODE)


if __name__ == "__main__":
    main()
