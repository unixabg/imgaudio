"""
_example.py — template for new motion lenses. Copy this file, rename it
(drop the leading underscore) and edit. The loader skips files starting
with `_`, so this one never appears in --motion-lens.

A motion lens reads a run of video frames and returns:
  lines   a few plain sentences for the text summary, one finding each
  tracks  records over time; motion.py turns each one into a voice in the WAV
          (a gliding tone, or notes on an instrument) and lists it in the text

This example splits the picture into three bands (top, middle, bottom) and
reports how much changes in each, frame to frame. It is meant to be read,
not used: the real work in a lens is deciding what to measure.
"""

import numpy as np

# Required: short identifier used by --motion-lens. Lowercase, no spaces.
NAME = "example"

# Required: one line shown in --help.
DESCRIPTION = "template: how much changes in the top, middle and bottom of the picture"

# Optional: the unit of your tracks' values, shown in the text summary.
UNIT = "brightness change"


def read(frames, times, params):
    """Read motion from frames.

    Args:
        frames: (n, rows, cols) uint8 numpy array, greyscale, row 0 at the top.
                Already cut to --box, --start and --duration.
        times:  (n,) float numpy array, each frame's real time in seconds.
                Cameras do not always record evenly; use these, not n / fps.
        params: dict of str -> str from --motion-params key=val,key=val.
                Read with a default: floor = float(params.get("floor", "2"))

    Returns:
        dict(lines=[str, ...],
             tracks=[dict(name=str, t=times array, value=array, size=float), ...])
        t and value are the same length. size is a typical swing of value
        (motion.py maps plus or minus `size` onto plus or minus seven
        semitones for the gliding tone). Return no tracks if nothing moved.
    """
    floor = float(params.get("floor", "2"))
    d = np.abs(np.diff(frames.astype(np.float32), axis=0))      # (n-1, rows, cols)
    d = np.maximum(d - floor, 0)
    t = (times[1:] + times[:-1]) / 2
    lines, tracks = [], []
    for name, band in zip(("top", "middle", "bottom"),
                          np.array_split(np.arange(frames.shape[1]), 3)):
        change = d[:, band].mean(axis=(1, 2))
        lines.append(f"{name}: average change {change.mean():.2f}, "
                     f"most at {t[int(np.argmax(change))]:.1f} s")
        if change.max() > 0:
            tracks.append(dict(name=name, t=t, value=change,
                               size=float(np.percentile(change, 95)) or 1.0))
    return dict(lines=lines, tracks=tracks)
