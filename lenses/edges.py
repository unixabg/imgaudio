"""
edges.py — biological edge detection lens.

Sonifies *contours* instead of raw brightness. Inspired by how mammalian
retinas use lateral inhibition (adjacent photoreceptors suppress each other)
to enhance edges. The result: smooth bright areas (sky, pavement) become
quiet, while boundaries (rooflines, vehicle outlines, the horizon) become
loud.

Two images that look similar in overall brightness but differ in shape
content will sound dramatically different through this lens.

Params (via --lens-params):
    threshold   suppress edge values below this (0.0-1.0). Default 0.1.
    boost       multiply edge magnitudes by this. Default 2.0.
"""

import numpy as np
from scipy.ndimage import sobel, gaussian_filter


NAME = "edges"
DESCRIPTION = "edge detection — sonifies image contours, not brightness"


def analyze(grid, params):
    threshold = float(params.get("threshold", "0.1"))
    boost     = float(params.get("boost", "2.0"))

    # Light blur first, to reduce single-pixel noise edges
    smoothed = gaussian_filter(grid, sigma=0.8)

    # Sobel edge magnitude: gradient along both axes, combined
    gx = sobel(smoothed, axis=1)
    gy = sobel(smoothed, axis=0)
    edges = np.hypot(gx, gy).astype(np.float32)

    # Normalize to [0, 1]
    peak = edges.max()
    if peak > 0:
        edges = edges / peak

    # Apply threshold (zero out faint edges to keep the audio clean)
    edges = np.where(edges < threshold, 0.0, edges)

    # Boost remaining edges, clip to [0, 1]
    return np.clip(edges * boost, 0.0, 1.0)
