#!/usr/bin/env python3
"""
hidden-pictures.py — make two test clips with pictures hidden in them.

  examples/hidden-word.wav   the word "imgaudio" drawn at 10-18 kHz (log scale),
                             mixed quietly under a melody with chirps and a
                             rising whistle. Invisible in the default
                             80 Hz-8 kHz view; read it with the full range:
      python3 imgaudio.py --sr 44100 --f-lo 40 --f-hi 20000 decode examples/hidden-word.wav word.png

  examples/hidden-stone.wav  the site icon (a translators' stone) drawn across
                             40 Hz-20 kHz on a linear scale, the way most
                             spectrogram tools draw. Squashed on the log scale;
                             in shape on the linear one:
      python3 imgaudio.py --sr 44100 --f-lo 40 --f-hi 20000 --freq-scale linear \
          --rows 300 --cols 300 decode examples/hidden-stone.wav stone.png

Both are 44.1 kHz mono WAVs, so they also work in the web page's Sound mode
with Frequency range set to Full.

Run from the repository root:  python3 examples/hidden-pictures.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import imgaudio  # noqa: E402

SR = 44100
OUT = ROOT / "examples"
ENCODE = dict(imgaudio.DEFAULTS, sr=SR, auto_prep=False, color=False, lens="raw")


def melody(seconds=20.0):
    """The web page's sample sound: a slow melody, birdlike chirps and a
    rising whistle, so the hidden word has something to hide under."""
    t = np.arange(int(SR * seconds)) / SR
    x = np.zeros_like(t)
    for i, f in enumerate([220, 262, 330, 392, 330, 262, 294, 349]):
        s, n = int(i * 2.4 * SR), int(2.0 * SR)
        tt = np.arange(min(n, len(x) - s)) / SR
        x[s:s + len(tt)] += 0.35 * np.sin(2 * np.pi * f * tt) * np.exp(-1.2 * tt)
    for k in range(12):
        s, n = int((0.8 + k * 1.55) * SR), int(0.18 * SR)
        tt = np.arange(n) / SR
        x[s:s + n] += 0.25 * np.hanning(n) * np.sin(2 * np.pi * (2500 * tt + 1500 / 0.18 * tt ** 2 / 2))
    sweep = (t > 14) & (t < 19)
    ts = t[sweep] - 14
    x[sweep] += 0.2 * np.sin(2 * np.pi * (100 * ts + 390 / 2 * ts ** 2))
    return x / np.abs(x).max()


def word_image(path):
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 150)
    except OSError:
        font = ImageFont.load_default(size=150)      # Pillow 10.1+
    im = Image.new("L", (1200, 300), 0)
    ImageDraw.Draw(im).text((600, 150), "imgaudio", fill=255, font=font, anchor="mm")
    im.save(path)


def save(path, x):
    wavfile.write(path, SR, (x / np.abs(x).max() * 0.9 * 32767).astype(np.int16))
    print(f"wrote {path.relative_to(ROOT)}  ({len(x) / SR:.1f}s)")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1. A word up high, under the melody.
        word_image(tmp / "word.png")
        imgaudio.encode(str(tmp / "word.png"), str(tmp / "word.wav"),
                        **dict(ENCODE, rows=120, cols=400, f_lo=10000, f_hi=18000, gamma=1.0))
        _, w = wavfile.read(tmp / "word.wav")
        w = w.astype(np.float32) / 32767
        m = melody(len(w) / SR)
        n = min(len(w), len(m))
        save(OUT / "hidden-word.wav", m[:n] * 0.8 + w[:n] * 0.25)

        # 2. The site icon on a linear scale across the whole range.
        icon = ROOT / "web" / "icons" / "icon-512.png"
        imgaudio.encode(str(icon), str(tmp / "stone.wav"),
                        **dict(ENCODE, rows=300, cols=300, f_lo=40, f_hi=20000,
                               auto_prep=True, freq_scale="linear"))
        _, s = wavfile.read(tmp / "stone.wav")
        save(OUT / "hidden-stone.wav", s.astype(np.float32) / 32767)


if __name__ == "__main__":
    main()
