"""
_example.py — template for new lenses. Copy this file, rename it (drop the
leading underscore), and modify. The loader skips files starting with `_`.

A lens transforms a 2D brightness grid into a 2D brightness grid. The
transformation should surface some hidden structure — edges, textures,
paths, statistical features — so the resulting audio reflects that
structure rather than raw brightness.
"""

import numpy as np


# Required: short identifier used by --lens. Use lowercase, no spaces.
NAME = "example"

# Required: one-line description shown in --help.
DESCRIPTION = "template lens that does nothing — copy this to start a new one"


def analyze(grid, params):
    """Transform a brightness grid.

    Args:
        grid:   (rows, cols) float32 numpy array, values in [0.0, 1.0].
                Already preprocessed if --auto-prep was used.
        params: dict of str -> str, parsed from --lens-params on the CLI.

    Returns:
        (rows, cols) float32 numpy array, values in [0.0, 1.0].
        Same shape as input.

    Convention for reading params (cast to the type you want, with a default):
        threshold = float(params.get("threshold", "0.5"))
        n_passes  = int(params.get("n_passes", "3"))
        verbose   = params.get("verbose", "false").lower() == "true"
    """
    # This template just returns the input unchanged.
    return grid.astype(np.float32)
