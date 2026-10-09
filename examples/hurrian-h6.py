#!/usr/bin/env python3
"""
hurrian-h6.py — the opening of the Hymn to Nikkal (Hurrian hymn h.6, Ugarit,
about 1400 BCE), rendered from the tablet's own notation.

The notation under the hymn is a list of Akkadian names for pairs of lyre
strings, each followed by a number. This script turns that list into sound
and writes two readings, because scholars disagree on how to play it:

  examples/h6-together.wav  both strings of each pair at once (Kilmer)
  examples/h6-in-turn.wav   the two strings one after the other (Dumbrill)

Look at either one with the shared palette:
  python3 imgaudio.py --palette octave decode examples/h6-together.wav h6.png

What is from the sources, and what is a choice made here:

  From the sources
    - The notation of the first two lines, as transcribed from Laroche,
      Ugaritica V (1968), in NOTATION below. The remaining lines are in that
      edition and have not been added; add them to NOTATION once checked.
    - Which strings each term names (the Old Babylonian string-pair table;
      Hurrian forms on the tablet, Akkadian names in comments).
    - The colophon: a song in the nid qabli tuning.

  Choices made here (each is one reading among several)
    - Strings 1-7 rise in pitch and are tuned to notate.py's "babylonian"
      scale (the nid qablim tuning named in the colophon, on Kilmer's rising
      reading), stacked fifths and fourths from A3. Real string order, register
      and the effect of the nid qabli tuning are debated.
    - The numbers are read as repeats. Others read them as durations or beats.
    - "uš-ta-ma-a-ri" ends line 1 but is not a string-pair term; it is skipped.
    - Tempo, tone and loudness are arbitrary. Rhythm is not recoverable.

Run from the repository root:  python3 examples/hurrian-h6.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import notate  # noqa: E402

# Hurrian term on the tablet -> (first string, second string)
STRING_PAIRS = {
    "qablite":     (5, 2),   # qablītum, "middle" (a fourth)
    "irbute":      (2, 7),   # rebûttum, "fourth"
    "šaḫri":       (7, 5),   # šērum, "song"
    "išarte":      (2, 6),   # išartum, "straight" (a fifth)
    "titimišarte": (3, 5),   # titur išartim, "bridge of the išartum"
    "zirte":       (4, 6),   # ṣerdum
    "šaššate":     (1, 6),   # šalšatum, "third"
    # Not yet needed by the lines below:
    "nīš tuḫrim":  (1, 5), "embūbum": (3, 7), "nīd qablim": (4, 1),
    "isqum":       (1, 3), "titur qablītim": (2, 4), "kitmum": (6, 3),
    "pītum":       (7, 4),
}

# The first two notation lines of h.6 (Laroche 1968), as (term, number).
NOTATION = [
    # line 1: qáb-li-te 3 ir-bu-te 1 qáb-li-te 3 ša-aḫ-ri 1 i-šar-te 10 uš-ta-ma-a-ri
    ("qablite", 3), ("irbute", 1), ("qablite", 3), ("šaḫri", 1), ("išarte", 10),
    # line 2: ti-ti-mi-šar-te 2 zi-ir-te 1 ša-[a]ḫ-ri 2 ša-aš-ša-te 2 ir-bu-te 2
    ("titimišarte", 2), ("zirte", 1), ("šaḫri", 2), ("šaššate", 2), ("irbute", 2),
]

ROOT_HZ = notate.midi_to_hz(notate.note_name_to_midi("A3"))
TUNING = notate.RATIO_SCALES["nid_qablim"]          # the colophon's tuning; equals "babylonian"
BPM = 80
SR = 22050


def string_hz(n):
    return ROOT_HZ * TUNING[n - 1]


def events(together):
    """notate.py note tuples (start_beat, dur_beats, freq_hz, velocity)."""
    notes, beat = [], 0.0
    for term, count in NOTATION:
        a, b = STRING_PAIRS[term]
        for _ in range(count):
            if together:
                notes += [(beat, 1.0, string_hz(a), 96), (beat, 1.0, string_hz(b), 96)]
                beat += 1.0
            else:
                notes += [(beat, 0.5, string_hz(a), 96), (beat + 0.5, 0.5, string_hz(b), 88)]
                beat += 1.0
    return notes


def main():
    for name, together in [("h6-together", True), ("h6-in-turn", False)]:
        notes = events(together)
        audio = notate.synthesize(notes, BPM, sr=SR, decay=3.0)
        path = ROOT / "examples" / f"{name}.wav"
        wavfile.write(path, SR, (audio * 32767).astype(np.int16))
        print(f"wrote {path.relative_to(ROOT)}  ({len(audio) / SR:.1f}s, {len(notes)} notes)")


if __name__ == "__main__":
    main()
