#!/usr/bin/env python3
"""
anomaly.py — recurrence plots and anomaly detection for audio.

A recurrence plot compares every moment in a signal to every other moment,
plotting brightness where two moments are similar. Structure invisible in a
waveform or spectrogram becomes legible:

    diagonal lines      repeated passages (the signal revisits a state)
    solid blocks        sustained regimes (a long stretch of "same thing")
    block boundaries    regime changes (something shifted)
    dark rows/columns   anomalies (a moment unlike any other moment)
    checkerboard        periodicity (day/night cycles, seasonal rhythm)

For a year of frame-audio, the whole year becomes one square image where the
rhythms and the ruptures are both visible at once.

Subcommands:
    recurrence IN.wav OUT.png     build the recurrence plot
    discords   IN.wav             rank the most anomalous moments

Examples:
    python3 anomaly.py recurrence year.wav year_rp.png
    python3 anomaly.py recurrence year.wav rp.png --size 1500 --mode binary
    python3 anomaly.py recurrence year.wav rp.png --discords 20
    python3 anomaly.py discords year.wav --top 20

Dependencies: numpy, scipy, Pillow
"""

import argparse
import sys

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import stft


# ---------------------------------------------------------------------------
# Feature extraction — turn audio into a sequence of state vectors
# ---------------------------------------------------------------------------
def audio_to_features(path, n_frames, n_bands=64, f_lo=60.0, f_hi=9000.0):
    """Load audio and reduce it to `n_frames` log-band spectral vectors.

    Each frame is a vector describing "what the sound was like" at that
    moment. Two frames are similar if their vectors are close. Log-spaced
    bands (rather than raw FFT bins) match pitch perception and keep the
    vectors compact.

    Returns: ((n_frames, n_bands) float32, L2-normalized rows), duration.
    """
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32767.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483647.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / sr
    print(f"loaded {path}: {duration:.1f}s at {sr} Hz")

    nperseg = 2048
    target_stft_frames = max(n_frames * 2, 512)
    hop = max(1, (len(audio) - nperseg) // target_stft_frames)
    f, _, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=nperseg - hop,
                   window="hann", boundary=None, padded=False)
    mag = np.abs(Z)

    # Collapse FFT bins into log-spaced bands
    edges = np.geomspace(f_lo, min(f_hi, sr / 2 - 1), n_bands + 1)
    bands = np.zeros((n_bands, mag.shape[1]), dtype=np.float32)
    for b in range(n_bands):
        sel = (f >= edges[b]) & (f < edges[b + 1])
        if np.any(sel):
            bands[b] = mag[sel].mean(axis=0)
        elif b > 0:
            bands[b] = bands[b - 1]

    bands = np.log1p(bands * 100.0)

    # Pool along time down to exactly n_frames
    t_frames = bands.shape[1]
    if t_frames < n_frames:
        print(f"warning: only {t_frames} analysis frames available; "
              f"reducing --size to {t_frames}", file=sys.stderr)
        n_frames = t_frames
    idx = np.linspace(0, t_frames, n_frames + 1).astype(int)
    feats = np.zeros((n_frames, n_bands), dtype=np.float32)
    for i in range(n_frames):
        a, b = idx[i], max(idx[i + 1], idx[i] + 1)
        feats[i] = bands[:, a:b].mean(axis=1)

    # L2-normalize so comparison is about spectral *shape*, not loudness.
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return feats / norms, duration


# ---------------------------------------------------------------------------
# Recurrence matrix
# ---------------------------------------------------------------------------
def recurrence_matrix(feats):
    """Pairwise distance between every pair of frames.

    ||a-b||² = ||a||² + ||b||² - 2·a·b, and both norms are 1 after L2
    normalization, so this reduces to a single matrix multiply.
    """
    n = feats.shape[0]
    print(f"computing {n}×{n} recurrence matrix ({n * n / 1e6:.1f}M cells)...")
    gram = feats @ feats.T
    return np.sqrt(np.maximum(2.0 - 2.0 * gram, 0.0)).astype(np.float32)


def render(dist, mode="continuous", percentile=15.0, invert=False):
    """Turn a distance matrix into a displayable 8-bit image."""
    if mode == "binary":
        thresh = np.percentile(dist, percentile)
        img = (dist <= thresh).astype(np.float32)
        print(f"  binary threshold at {percentile}th percentile "
              f"(distance {thresh:.4f}); {100 * img.mean():.1f}% recurrent")
    else:
        lo, hi = np.percentile(dist, [2, 98])
        img = np.clip((dist - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        img = 1.0 - img          # bright = similar
    if invert:
        img = 1.0 - img
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Discords — the moments least like any other moment
# ---------------------------------------------------------------------------
def find_discords(dist, duration, top=10, exclusion=0.02):
    """Rank frames by how unlike everything else they are.

    For each frame, find the distance to its nearest neighbour, excluding
    itself and its immediate temporal neighbours (which are trivially
    similar). Frames whose *nearest* neighbour is still far away are the
    anomalies — the matrix-profile "discord" idea.
    """
    n = dist.shape[0]
    band = max(1, int(n * exclusion))
    d = dist.copy()
    for offset in range(-band, band + 1):
        idx = np.arange(max(0, -offset), min(n, n - offset))
        d[idx, idx + offset] = np.inf

    nn_dist = d.min(axis=1)
    order = np.argsort(nn_dist)[::-1][:top]
    sec_per_frame = duration / n
    return [(int(i), float(i * sec_per_frame), float(nn_dist[i]))
            for i in order]


def _print_discords(dist, duration, top):
    print(f"\ntop {top} anomalous moments:")
    for rank, (i, t, d) in enumerate(find_discords(dist, duration, top=top), 1):
        mm, ss = divmod(t, 60)
        hh, mm = divmod(mm, 60)
        print(f"  {rank:2d}. {int(hh):02d}:{int(mm):02d}:{ss:05.2f}  "
              f"(frame {i}, novelty {d:.4f})")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_recurrence(args):
    feats, duration = audio_to_features(args.input, args.size,
                                        n_bands=args.bands)
    dist = recurrence_matrix(feats)
    img = render(dist, mode=args.mode, percentile=args.percentile,
                 invert=args.invert)
    Image.fromarray(img, mode="L").save(args.output)
    print(f"wrote {args.output}  ({img.shape[1]}×{img.shape[0]}, "
          f"{duration / img.shape[0]:.2f}s per pixel)")
    if args.discords:
        _print_discords(dist, duration, args.discords)


def cmd_discords(args):
    feats, duration = audio_to_features(args.input, args.size,
                                        n_bands=args.bands)
    dist = recurrence_matrix(feats)
    _print_discords(dist, duration, args.top)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size",  type=int, default=1000,
                   help="frames per axis; output is size×size pixels")
    p.add_argument("--bands", type=int, default=64,
                   help="log-spaced spectral bands per frame vector")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("recurrence", help="build a recurrence plot")
    pr.add_argument("input"); pr.add_argument("output")
    pr.add_argument("--mode", choices=["continuous", "binary"],
                    default="continuous")
    pr.add_argument("--percentile", type=float, default=15.0,
                    help="binary mode: %% of closest pairs marked recurrent")
    pr.add_argument("--invert", action="store_true", help="flip black/white")
    pr.add_argument("--discords", type=int, default=0,
                    help="also print the N most anomalous moments")

    pd = sub.add_parser("discords", help="rank the most anomalous moments")
    pd.add_argument("input")
    pd.add_argument("--top", type=int, default=10)

    return p


def main():
    args = build_parser().parse_args()
    {"recurrence": cmd_recurrence, "discords": cmd_discords}[args.cmd](args)


if __name__ == "__main__":
    main()
