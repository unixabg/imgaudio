"""
reveal.py — show what is unusual for its pitch.

Each row of a spectrogram is one pitch. This lens learns how each pitch
normally behaves across the whole clip, its typical level and how much it
usually varies, and shows only how far each moment departs from that. It is
a standard score per row (median and median absolute deviation, so a few
loud events don't skew it), the same idea as noise whitening used when
searching recordings for faint signals.

Steady things fade: hum, drones, a constant noise floor, a held note. A quiet
pitch that suddenly does something lights up, even when it is far quieter
than the loudest sound in the clip. In a test, a picture buried 20 dB under
noise and mains hum went from barely matching the original (r = 0.09) to
plainly visible (r = 0.33).

Rows stay pitches, so the octave light palette still means pitch, and the
result can be encoded back into sound to hear only what stood out.

Params (via --lens-params):
    smooth    blur in pixels before scoring; calms noise speckle so shapes
              hold together. Default 1.0. Use 0 for the sharpest detail.
    zmax      how unusual counts as full brightness, in deviations.
              Default 6. Lower shows more; higher keeps only the rarest.
    contrast  gamma on the result; below 1 lifts faint detail. Default 1.0.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


NAME = "reveal"
DESCRIPTION = "what is unusual for its pitch — fades steady sound, lifts the rare"


def analyze(grid, params):
    smooth   = float(params.get("smooth", "1.0"))
    zmax     = float(params.get("zmax", "6"))
    contrast = float(params.get("contrast", "1.0"))

    g = grid.astype(np.float32)
    if smooth > 0:
        g = gaussian_filter(g, smooth)
    median = np.median(g, axis=1, keepdims=True)
    spread = np.median(np.abs(g - median), axis=1, keepdims=True) * 1.4826
    score = (g - median) / (spread + 1e-4)
    out = np.clip(score / zmax, 0.0, 1.0) ** contrast
    return out.astype(np.float32)
