#!/usr/bin/env python3
"""
seikilos.py — the Seikilos epitaph (1st or 2nd century CE), the oldest
complete piece of music whose melody and rhythm can both be read.

Seikilos carved a short song on a grave marker near Tralles in Asia Minor.
Over each syllable is a Greek letter naming its pitch, and above some
letters a mark for length: a bar doubles it, a bar with a hook triples it.
An arc joins two notes on one syllable. This script reads those letters
directly, so the melody comes from the stone rather than from a modern
arrangement:

  examples/seikilos.wav

  python3 imgaudio.py --palette octave --f-lo 200 --f-hi 2000 decode examples/seikilos.wav seikilos.png

Sources
  - Letters and length marks: the inscription as rendered in the public-
    domain (CC0) Wikimedia Commons image "Epitaph of Seikilos, lyrics and
    ancient musical score-notation.png". Each line adds up to twelve time
    units, two bars of 6/8, which is how the song is usually transcribed.
  - Letter pitches: C=A, Z=E', I=D', K=C#', O=B, Φ=G, X=F#, Γ=E, as tabled in
    "The Epitaph of Seikilos" (IMEKO TC4 Archaeo 2016).
  - The stone is in the National Museum of Denmark, Copenhagen.

Choices made here
  - The final syllable: the image reaches its edge at A then F#. Winnington-
    Ingram describes the cadence as F# to E, and the pitch table includes E
    (Γ), which appears nowhere else, so the song ends A F# E here.
  - The dots above some letters (stigmai) are not played; their meaning is
    debated.
  - Tuning is Pythagorean (stacked fifths, like notate.py's babylonian
    scale), at the pitch the table gives, with A at 440 Hz. Many
    transcriptions sit a fourth lower.
  - Tempo and tone are arbitrary.

Run from the repository root:  python3 examples/seikilos.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import notate  # noqa: E402

A4 = 440.0
# Greek vocal-notation letter -> pitch, Pythagorean ratios from A4
PITCH = {
    "Z": 3 / 2,      # E5
    "I": 4 / 3,      # D5
    "K": 81 / 64,    # C#5
    "O": 9 / 8,      # B4
    "C": 1 / 1,      # A4
    "Φ": 8 / 9,      # G4
    "X": 27 / 32,    # F#4
    "Γ": 3 / 4,      # E4
}

# (syllable, [(letter, time units), ...]); twelve units per line.
SONG = [
    # Hóson zêis, phaínou   (while you live, shine)
    ("Ho", [("C", 1)]), ("son", [("Z", 2)]), ("zēs", [("Z", 3)]),
    ("phai", [("K", 1), ("I", 1)]), ("nou", [("Z", 1), ("I", 3)]),
    # mēdèn hólōs sù lypoû   (have no grief at all)
    ("mē", [("K", 2)]), ("den", [("I", 1)]), ("ho", [("Z", 1)]),
    ("lōs", [("I", 1), ("K", 1)]), ("sy", [("O", 1)]), ("ly", [("C", 2)]),
    ("pou", [("O", 1), ("Φ", 2)]),
    # pròs olígon ésti tò zên   (life exists only a short while)
    ("pros", [("C", 1)]), ("o", [("K", 1)]), ("li", [("Z", 1)]), ("gon", [("I", 1)]),
    ("es", [("K", 1), ("I", 1)]), ("ti", [("K", 1)]), ("to", [("C", 2)]),
    ("zēn", [("O", 1), ("Φ", 2)]),
    # tò télos ho khrónos apaiteî   (time demands its toll)
    ("to", [("C", 1)]), ("te", [("K", 1)]), ("los", [("O", 1)]), ("ho", [("I", 1)]),
    ("chro", [("Z", 1)]), ("nos", [("K", 1)]), ("ap", [("C", 1)]), ("ai", [("C", 2)]),
    ("tei", [("C", 1), ("X", 1), ("Γ", 1)]),
]

UNIT_BEATS = 1.0        # one time unit = one beat
BPM = 200               # 0.3 s per unit
SR = 22050


def notes():
    out, beat = [], 0.0
    for _, letters in SONG:
        for letter, units in letters:
            out.append((beat, units * UNIT_BEATS, A4 * PITCH[letter], 100))
            beat += units * UNIT_BEATS
    out[-1] = (out[-1][0], out[-1][1] + 3, out[-1][2], out[-1][3])   # let the end ring
    return out, beat


def main():
    ns, total = notes()
    assert total == 48, f"expected four lines of twelve units, got {total}"
    audio = notate.synthesize(ns, BPM, sr=SR, decay=2.0)
    path = ROOT / "examples" / "seikilos.wav"
    wavfile.write(path, SR, (audio * 32767).astype(np.int16))
    print(f"wrote {path.relative_to(ROOT)}  ({len(audio) / SR:.1f}s, {len(ns)} notes)")


if __name__ == "__main__":
    main()
