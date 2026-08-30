"""
spectral.py — 2D Fourier transform lens.

Sonifies the image's *spatial frequency* content rather than its brightness
or its edges. A 2D FFT decomposes a picture into repeating patterns: low
frequencies near the center are broad gradients, high frequencies toward the
edges are fine detail, and a bright streak at angle θ means strong periodic
structure running perpendicular to θ.

This asks a different question from the other lenses. `edges` finds *where*
boundaries are; this finds *how often* patterns repeat and *in what
direction*, regardless of where in the frame they sit. Two photos of the same
building from different positions look quite different to `edges` but nearly
identical here, because the repeating structure is unchanged.

Man-made scenes are full of periodic structure — brick courses, siding,
fence pickets, window mullions, shingles — and each shows as a distinct spike.
Vegetation, clouds, and water have broad diffuse spectra instead. The
resulting audio is a fingerprint of the scene's periodicity.

Because the FFT of a real image is symmetric about the center, the output
grid is symmetric too, which makes the synthesized audio unusually
consonant — every partial has a mirror partner.

Params (via --lens-params):
    mode      "magnitude" (default) full centered spectrum;
              "quadrant" one quarter only, since the rest is redundant;
              "radial"   average by distance from center — scale content
                         with orientation discarded;
              "angular"  average by angle — orientation content with scale
                         discarded.
    boost     contrast multiplier. Default 1.0.
    clip_lo   lower percentile for normalization (0-100). Default 2.
    clip_hi   upper percentile. Default 99.5.
    window    "hann" (default) tapers the image edges before transforming,
              removing the cross-shaped artifact caused by the frame's own
              hard borders; "none" leaves it in.
"""

import numpy as np

NAME = "spectral"
DESCRIPTION = "2D FFT — sonifies spatial frequency and orientation"


def _window_2d(rows, cols):
    """Separable Hann window. Without this, the image's own rectangular
    border acts as a step edge and dominates the spectrum with a bright
    horizontal/vertical cross."""
    return np.outer(np.hanning(rows), np.hanning(cols)).astype(np.float32)


def _radial_profile(mag):
    """Average magnitude by distance from center, then broadcast back to a
    full grid. Keeps scale information, discards orientation."""
    rows, cols = mag.shape
    cy, cx = rows / 2.0, cols / 2.0
    y, x = np.ogrid[:rows, :cols]
    r = np.hypot(y - cy, x - cx)
    r_int = r.astype(int)
    n_bins = r_int.max() + 1
    total = np.bincount(r_int.ravel(), weights=mag.ravel(), minlength=n_bins)
    count = np.bincount(r_int.ravel(), minlength=n_bins)
    profile = total / np.maximum(count, 1)
    return profile[r_int].astype(np.float32)


def _angular_profile(mag, n_bins=180):
    """Average magnitude by angle, then broadcast back. Keeps orientation
    information, discards scale."""
    rows, cols = mag.shape
    cy, cx = rows / 2.0, cols / 2.0
    y, x = np.ogrid[:rows, :cols]
    theta = np.arctan2(y - cy, x - cx) % np.pi          # 0..pi, fold symmetry
    idx = np.clip((theta / np.pi * n_bins).astype(int), 0, n_bins - 1)
    total = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=n_bins)
    count = np.bincount(idx.ravel(), minlength=n_bins)
    profile = total / np.maximum(count, 1)
    return profile[idx].astype(np.float32)


def analyze(grid, params):
    mode    = params.get("mode", "magnitude")
    boost   = float(params.get("boost", "1.0"))
    clip_lo = float(params.get("clip_lo", "2"))
    clip_hi = float(params.get("clip_hi", "99.5"))
    window  = params.get("window", "hann")

    rows, cols = grid.shape
    g = grid.astype(np.float64)

    # Remove DC before transforming — otherwise the mean brightness dominates
    # the center bin by orders of magnitude and swamps everything else.
    g = g - g.mean()

    if window == "hann":
        g = g * _window_2d(rows, cols)

    # 2D FFT, shifted so zero frequency sits at the center of the image
    spec = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(spec).astype(np.float32)

    # Log compression — spectral magnitudes span many orders of magnitude
    mag = np.log1p(mag)

    if mode == "radial":
        out = _radial_profile(mag)
    elif mode == "angular":
        out = _angular_profile(mag)
    elif mode == "quadrant":
        # The spectrum of a real image is conjugate-symmetric, so one
        # quadrant holds all the information. Tile it back to full size.
        q = mag[rows // 2:, cols // 2:]
        out = np.asarray(
            np.kron(q, np.ones((2, 2), dtype=np.float32))
        )[:rows, :cols].astype(np.float32)
        if out.shape != (rows, cols):          # odd dimensions
            padded = np.zeros((rows, cols), dtype=np.float32)
            h, w = out.shape
            padded[:h, :w] = out
            out = padded
    else:                                       # "magnitude"
        out = mag

    # Robust normalization
    lo, hi = np.percentile(out, [clip_lo, clip_hi])
    if hi - lo > 1e-9:
        out = np.clip((out - lo) / (hi - lo), 0.0, 1.0)
    else:
        out = np.zeros_like(out)

    return np.clip(out * boost, 0.0, 1.0).astype(np.float32)
