"""
fractal.py — local texture roughness lens.

Estimates local "natural roughness" of each region using multi-scale
intensity variation. Conceptually related to fractal dimension (rough
surfaces have higher D, smooth ones lower), but uses a simpler estimator
that works reliably at small window sizes and is much faster.

The method: compare fine-scale variation (pixel-to-pixel differences) to
coarse-scale variation (overall standard deviation of the patch). Natural
surfaces show similar variation at both scales (self-similar); man-made
smooth surfaces show very low fine-scale variation relative to their
overall range; pure noise shows high variation at both scales.

What it sonifies: every pixel becomes the local roughness of a window
around it, normalized to [0, 1]. Rough natural regions (foliage,
weathered surfaces, clouds) become bright; smooth artificial regions
(walls, pavement, clear sky) become dark.

Params (via --lens-params):
    window    sliding window size, odd. Default 7. Smaller = more detail
              but more noise. Try 5-15.
    boost     contrast multiplier for the final map. Default 1.5.
    clip_lo   lower percentile clip (0-100). Default 5.
    clip_hi   upper percentile clip (0-100). Default 95.
"""

import numpy as np
from scipy.ndimage import generic_filter, uniform_filter

NAME = "fractal"
DESCRIPTION = "local texture roughness — natural vs man-made surfaces"


def _local_roughness(patch):
    """Roughness measure for one square patch.

    Combines:
      - fine variation: mean absolute pixel-to-pixel difference
      - coarse variation: std of the whole patch
    Their ratio is a fractal-flavored measure: 1.0 means change is
    happening at the smallest scale (rough), near 0 means change happens
    only at larger scales (smooth gradient).
    """
    side = int(np.sqrt(patch.size))
    p = patch.reshape(side, side)

    coarse = float(np.std(p))
    if coarse < 1e-6:
        return 0.0   # uniform region — no texture

    # Fine-scale differences (high-frequency content)
    dh = np.abs(np.diff(p, axis=0)).mean()
    dv = np.abs(np.diff(p, axis=1)).mean()
    fine = (dh + dv) / 2.0

    # Roughness = how much of the variation is at the finest scale.
    # Multiply by coarse so totally-flat regions stay low.
    return float(fine * coarse)


def analyze(grid, params):
    window  = int(params.get("window", "7"))
    boost   = float(params.get("boost", "1.5"))
    clip_lo = float(params.get("clip_lo", "5"))
    clip_hi = float(params.get("clip_hi", "95"))

    if window % 2 == 0:
        window += 1
    if window < 3:
        window = 3

    # Run the per-window roughness estimator across the grid.
    roughness = generic_filter(
        grid.astype(np.float64),
        function=_local_roughness,
        size=window,
        mode="reflect",
    ).astype(np.float32)

    # Robust normalization: percentile-clip then stretch to [0,1].
    lo, hi = np.percentile(roughness, [clip_lo, clip_hi])
    if hi - lo > 1e-8:
        roughness = np.clip((roughness - lo) / (hi - lo), 0.0, 1.0)
    else:
        # Entire image has uniform roughness — emit a low uniform value
        # so the audio isn't dead silent (which would look like a bug).
        roughness = np.full_like(roughness, 0.15)

    return np.clip(roughness * boost, 0.0, 1.0)
