"""
phyllotaxis.py — golden-angle spiral scan lens.

The default left-to-right raster scan that imgaudio.py uses is an artificial
convention. Nature arranges things differently: sunflower seeds, pinecones,
and pine needles are placed at the golden angle (137.5°) per step, producing
optimal non-overlapping coverage from the center outward.

This lens re-reads the image in that order. The output's column index becomes
"step number in the spiral"; the row index still maps to frequency. Each
spiral step contributes a Gaussian-smeared profile across rows, centered at
the y-position where the spiral was visiting. This produces a dense,
listenable representation that also survives recursive iteration.

Params (via --lens-params):
    steps_per_col   how many spiral steps per output column. Default 4.
    radius_power    1.0 = uniform radial speed, 0.5 = sqrt scaling (sunflower
                    seed density). Default 0.5.
    center_x        relative center x, 0.0-1.0. Default 0.5.
    center_y        relative center y, 0.0-1.0. Default 0.5.
    spread          Gaussian smearing across rows, in pixels. Default 8.
                    Higher = denser/smoother; lower = sparser/sharper.
"""

import numpy as np

NAME = "phyllotaxis"
DESCRIPTION = "golden-angle spiral scan — reads image center-out instead of L-to-R"

# Golden angle in radians: 360° / φ² ≈ 137.508°
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def analyze(grid, params):
    steps_per_col = int(params.get("steps_per_col", "4"))
    radius_power  = float(params.get("radius_power", "0.5"))
    center_x      = float(params.get("center_x", "0.5"))
    center_y      = float(params.get("center_y", "0.5"))
    spread        = float(params.get("spread", "8.0"))

    rows, cols = grid.shape
    total_steps = cols * steps_per_col

    cy = center_y * (rows - 1)
    cx = center_x * (cols - 1)
    max_radius = min(cy, rows - 1 - cy, cx, cols - 1 - cx)
    max_radius = max(max_radius, min(rows, cols) / 4.0)

    # Spiral positions: step n is at (radius, angle)
    n = np.arange(total_steps, dtype=np.float64)
    norm = n / total_steps
    radii = max_radius * np.power(norm, radius_power)
    thetas = n * GOLDEN_ANGLE

    ys_f = cy + radii * np.sin(thetas)
    xs_f = cx + radii * np.cos(thetas)

    ys = np.clip(np.round(ys_f).astype(int), 0, rows - 1)
    xs = np.clip(np.round(xs_f).astype(int), 0, cols - 1)

    # Sample brightness at each spiral point
    samples = grid[ys, xs].astype(np.float32)

    # For each step, build a Gaussian profile across rows centered at its y.
    # Vectorized: produce a (rows, total_steps) matrix where each column
    # is the Gaussian weight from one spiral step.
    row_idx = np.arange(rows, dtype=np.float32).reshape(-1, 1)
    ys_2d   = ys_f.astype(np.float32).reshape(1, -1)
    gauss   = np.exp(-((row_idx - ys_2d) ** 2) / (2.0 * spread * spread))

    # Multiply each column by its sample value
    contributions = gauss * samples.reshape(1, -1)   # (rows, total_steps)

    # Reduce groups of steps_per_col adjacent steps into a single output column.
    # Use max so multiple overlapping bumps don't wash each other out.
    contributions_3d = contributions.reshape(rows, cols, steps_per_col)
    out = contributions_3d.max(axis=2)

    # Normalize to [0, 1]
    peak = out.max()
    if peak > 0:
        out = out / peak
    return out.astype(np.float32)
