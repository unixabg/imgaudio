#!/usr/bin/env python3
"""
imgaudio.py — image ↔ audio in a single script.

Encode an image as audio whose spectrogram visually reconstructs the image,
and decode that audio back to an image. The "Photosounder" approach: each
image row maps to a frequency (log-spaced), each column maps to a time slice,
pixel brightness controls oscillator amplitude. Synthesis uses overlap-add
with Hann windows and continuous phase per oscillator (no clicks).

Subcommands:
    encode IMAGE AUDIO      image → audio
    decode AUDIO IMAGE      audio → image (via STFT)
    roundtrip IMAGE         encode + decode + comparison plot

Examples:
    python3 imgaudio.py encode photo.jpg photo.wav
    python3 imgaudio.py decode photo.wav recovered.png
    python3 imgaudio.py roundtrip photo.jpg
    python3 imgaudio.py encode photo.jpg photo.wav --rows 400 --cols 800 --col-sec 0.03
    python3 imgaudio.py --lens edges encode photo.jpg photo.wav

Dependencies: numpy, scipy, Pillow, matplotlib (matplotlib only for roundtrip)
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import get_window, stft


# ---------------------------------------------------------------------------
# Lens registry — auto-loads every .py file in ./lenses (except _example, __init__)
# ---------------------------------------------------------------------------
def _load_lenses():
    """Discover lens modules in ./lenses next to this script.

    A lens is a Python file exporting:
        NAME        : str   — short identifier used by --lens
        DESCRIPTION : str   — one-line description shown in help
        analyze(grid, params) -> grid
                      Takes a (rows, cols) numpy array in [0,1] and a dict
                      of string parameters; returns a (rows, cols) array in [0,1].

    See lenses/README.md and lenses/_example.py for the contract.
    """
    here = Path(__file__).resolve().parent
    lens_dir = here / "lenses"
    registry = {}
    if not lens_dir.is_dir():
        return registry
    sys.path.insert(0, str(here))
    for path in sorted(lens_dir.glob("*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue
        modname = f"lenses.{path.stem}"
        try:
            mod = importlib.import_module(modname)
            name = getattr(mod, "NAME", path.stem)
            registry[name] = mod
        except Exception as e:
            print(f"warning: failed to load lens {path.name}: {e}",
                  file=sys.stderr)
    return registry


_LENSES = _load_lenses()


def _parse_lens_params(s):
    """Parse 'key=val,key=val' into a dict. Values stay strings; lenses cast."""
    if not s:
        return {}
    out = {}
    for pair in s.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# Adaptive preprocessing — handles wildly varying exposure across frames
# ---------------------------------------------------------------------------
def _auto_prep(grid):
    """Adaptive per-image preprocessing for mixed-exposure datasets.

    1. Percentile-clip 1%/99% then stretch to [0, 1]. This expands the actual
       used range to fill the histogram regardless of how bright or dim the
       source is.
    2. Inspect the median brightness AFTER stretching. If it's bright
       (median > 0.5), invert so the structural content (buildings, signs,
       silhouettes) becomes the bright "loud" pixels for the encoder.
    3. Apply a mild gamma to firm up mid-tone contrast.

    Returns: (prepped_grid, dict_of_decisions)
    """
    lo, hi = np.percentile(grid, [1, 99])
    if hi > lo:
        grid = np.clip((grid - lo) / (hi - lo), 0.0, 1.0)

    median_after = float(np.median(grid))
    inverted = median_after > 0.5
    if inverted:
        grid = 1.0 - grid

    grid = grid ** 1.3
    return grid, {"clip_lo": float(lo), "clip_hi": float(hi),
                  "median_after": median_after, "inverted": inverted}


# ---------------------------------------------------------------------------
# Defaults — encode and decode must use the same values to round-trip cleanly
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    sr=22050,         # sample rate (Hz)
    rows=200,         # frequency bins (image rows)
    cols=400,         # time steps (image cols)
    f_lo=80.0,        # lowest freq (Hz)
    f_hi=8000.0,      # highest freq (Hz)
    col_sec=0.05,     # seconds per column
    gamma=1.7,        # brightness gamma (encoder side)
    threshold=0.02,   # ignore pixels below this brightness when synthesizing
    auto_prep=False,  # apply adaptive preprocessing
)


# ---------------------------------------------------------------------------
# Colour sidecar — chroma stored beside the audio, reapplied on decode
# ---------------------------------------------------------------------------
# The audio carries luminance only. Human vision has far lower colour
# resolution than luminance resolution (which is why JPEG and every video
# codec subsample chroma), so the colour information compresses to a few
# hundred bytes with almost no visible loss. Stored as a small PNG beside the
# WAV and reapplied at decode as best-effort recolouring.
#
# Measured against the original: 32x18 gives RMSE 0.025 at ~450 bytes,
# 16x9 gives 0.029 at ~193 bytes, 8x5 gives 0.032 at ~119 bytes.

def _png_meta(inverted, w, h):
    from PIL import PngImagePlugin
    meta = PngImagePlugin.PngInfo()
    meta.add_text("imgaudio_inverted", "1" if inverted else "0")
    meta.add_text("imgaudio_size", f"{w}x{h}")
    return meta


def chroma_path(audio_path):
    """Sidecar path for a given audio file: foo.wav -> foo.chroma.png"""
    return Path(audio_path).with_suffix(".chroma.png")


def write_chroma(image_path, sidecar_path, width=32, inverted=False):
    """Save a tiny YCbCr chroma plane next to the audio."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    height = max(1, round(width * h / w))
    small = img.convert("YCbCr").resize((width, height), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.uint8).copy()
    arr[..., 0] = 128                     # drop luma; the audio carries it
    Image.fromarray(arr, "YCbCr").convert("RGB").save(
        sidecar_path, "PNG", pnginfo=_png_meta(inverted, w, h))
    size = Path(sidecar_path).stat().st_size
    print(f"  chroma sidecar: {sidecar_path} ({width}x{height}, {size} bytes"
          f"{', inverted' if inverted else ''})")


def apply_chroma(gray_arr, sidecar_path):
    """Recolour a decoded grayscale array using a chroma sidecar.

    Returns an RGB array. Best-effort: chroma is upsampled from a very small
    plane, so colour boundaries are soft. Luminance detail comes entirely
    from the audio.
    """
    side = Image.open(sidecar_path)
    inverted = side.info.get("imgaudio_inverted", "0") == "1"
    h, w = gray_arr.shape

    luma = gray_arr
    if inverted:
        # auto-prep flipped the image before encoding; flip back so the
        # recovered luminance matches the orientation the chroma was taken in
        luma = 255 - luma

    chroma = np.asarray(
        side.convert("YCbCr").resize((w, h), Image.BICUBIC), dtype=np.uint8)
    merged = np.stack([luma.astype(np.uint8),
                       chroma[..., 1], chroma[..., 2]], axis=-1)
    return np.asarray(Image.fromarray(merged, "YCbCr").convert("RGB"))


# ---------------------------------------------------------------------------
# Frequency grid — which pitch each image row stands for
# ---------------------------------------------------------------------------
# Encode, decode and the palette must agree on this, or pictures come back
# stretched. "log" (the default) gives every octave the same height, the way
# we hear pitch. "linear" gives every Hz the same height, the way most
# spectrogram tools draw, so it suits images drawn with those tools.

def freq_grid(f_lo, f_hi, rows, scale="log"):
    """Frequency of each row, lowest first."""
    if scale == "linear":
        return np.linspace(f_lo, f_hi, rows)
    return np.geomspace(f_lo, f_hi, rows)


# ---------------------------------------------------------------------------
# Octave light — one shared palette for every sound
# ---------------------------------------------------------------------------
# Visible light spans almost exactly one octave: deep red near 400 THz to
# violet near 790 THz. Raising any pitch by whole octaves lands it somewhere
# in that octave of light, so every pitch gets a colour that depends on
# nothing but the pitch, and the same note has the same colour in every
# octave and in every recording. Brightness stays loudness.
#
# The anchor is physical (deep red where light's octave begins), but the
# twelve notes are spaced evenly around the colour wheel in rainbow order.
# Pure wavelength colours would put F#, G and G# in one indistinguishable red.

OCTAVE_LIGHT_START = 400e12   # Hz, deep red


def octave_rgb(freq_hz):
    """RGB in [0, 1] for each frequency: red -> yellow -> green -> blue ->
    violet as the pitch rises through an octave, repeating every octave."""
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    light = freq_hz * 2.0 ** np.ceil(np.log2(OCTAVE_LIGHT_START / freq_hz))
    p = np.log2(light / OCTAVE_LIGHT_START) % 1.0      # position in the octave
    h = p * 5.0                                          # hue 0-300 deg, in sixths
    x = 1.0 - np.abs(h % 2.0 - 1.0)
    one, zero = np.ones_like(h), np.zeros_like(h)
    seg = np.floor(h).astype(int) % 6
    r = np.choose(seg, [one, x, zero, zero, x, one])
    g = np.choose(seg, [x, one, one, x, zero, zero])
    b = np.choose(seg, [zero, zero, x, one, one, x])
    return np.stack([r, g, b], axis=-1)


def apply_octave(gray_arr, f_lo, f_hi, scale="log"):
    """Colour a decoded grayscale picture by pitch. Row 0 is the highest
    frequency, matching decode's image orientation."""
    rows = gray_arr.shape[0]
    rgb = octave_rgb(freq_grid(f_lo, f_hi, rows, scale)[::-1])[:, None, :]
    return (rgb * gray_arr[..., None]).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Annotation — rulers on a decoded picture, for reading it precisely
# ---------------------------------------------------------------------------
# A decoded picture shows the shape of a sound; the rulers say exactly when
# and at what pitch. The annotated copy is for reading only: it is written
# beside the plain picture, which stays untouched so the round trip still works.

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _font(size):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)       # Pillow 10.1+
    except TypeError:
        return ImageFont.load_default()


def _nice_step(span, target):
    """A round step (1, 2 or 5 x 10^n) giving about `target` ticks over span."""
    raw = span / max(target, 1)
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def annotate(picture, out_path, *, t_first, t_last, f_lo, f_hi, scale="log",
             octave_key=False, title=""):
    """Write a copy of a decoded picture with a time ruler (seconds, bottom),
    a frequency ruler (Hz, left) and note names (right; every C, plus A4).

    picture       uint8 array, (rows, cols) grey or (rows, cols, 3) colour;
                  row 0 is the highest frequency, as decode writes it
    t_first/last  seconds at the centres of the first and last columns
    """
    from PIL import ImageDraw
    pic = Image.fromarray(picture).convert("RGB")
    cols, rows = pic.size
    k = max(1, int(np.ceil(900 / cols)))                 # enlarge small pictures
    pic = pic.resize((cols * k, rows * k), Image.NEAREST)
    W, H = pic.size
    L, R, T, B = 74, 78, 34, 46 + (34 if octave_key else 0)
    out = Image.new("RGB", (L + W + R, T + H + B), "white")
    out.paste(pic, (L, T))
    d = ImageDraw.Draw(out)
    f_small, f_label = _font(13), _font(14)
    ink, faint = (20, 20, 20), (120, 120, 120)

    freqs = freq_grid(f_lo, f_hi, rows, scale)           # row 0 = lowest here
    def y_of(freq):
        """Pixel row of a frequency: centre of the row it falls in."""
        if scale == "linear":
            i = (freq - f_lo) / (f_hi - f_lo) * (rows - 1)
        else:
            i = np.log(freq / f_lo) / np.log(f_hi / f_lo) * (rows - 1)
        return T + (rows - 1 - i + 0.5) * k

    def x_of(t):
        """Pixel column of a time: decode's columns are evenly spaced from
        t_first to t_last, each centred on its analysis window."""
        j = (t - t_first) / max(t_last - t_first, 1e-9) * (cols - 1)
        return L + (j + 0.5) * k

    d.rectangle((L - 1, T - 1, L + W, T + H), outline=ink)

    # Frequency ruler (left)
    if scale == "linear":
        step = _nice_step(f_hi - f_lo, 8)
        ticks = np.arange(np.ceil(f_lo / step) * step, f_hi + 1e-9, step)
    else:
        ticks = [m * 10 ** e for e in range(1, 6) for m in (1, 2, 5)]
    for f in ticks:
        if f_lo <= f <= f_hi:
            y = y_of(f)
            d.line((L - 6, y, L - 1, y), fill=ink)
            label = f"{f / 1000:g}k" if f >= 1000 else f"{f:g}"
            d.text((L - 9, y), label, fill=ink, font=f_small, anchor="rm")
    d.text((10, T - 20), "Hz", fill=faint, font=f_small)

    # Note names (right): every C, and A4 = 440 Hz for reference
    last_y = None
    marks = [(12 * (o + 1), f"C{o}") for o in range(0, 10)] + [(69, "A4")]
    for midi, name in sorted(marks, key=lambda m: -m[0]):
        f = 440.0 * 2 ** ((midi - 69) / 12)
        if not f_lo <= f <= f_hi:
            continue
        y = y_of(f)
        if last_y is not None and abs(y - last_y) < 14:
            continue
        d.line((L + W, y, L + W + 6, y), fill=ink)
        d.text((L + W + 9, y), name, fill=ink, font=f_small, anchor="lm")
        last_y = y
    d.text((L + W + 9, T - 20), "note", fill=faint, font=f_small)

    # Time ruler (bottom)
    span = t_last - t_first
    step = _nice_step(span, 10)
    half_col = span / max(cols - 1, 1) / 2               # the picture's edge
    for t in np.arange(np.ceil((t_first - half_col) / step) * step, t_last + half_col + 1e-9, step):
        x = x_of(t)
        d.line((x, T + H, x, T + H + 6), fill=ink)
        d.text((x, T + H + 9), f"{round(t, 6) + 0:g}", fill=ink, font=f_small, anchor="mt")
    d.text((L + W + 9, T + H + 9), "s", fill=faint, font=f_small)

    # Title
    scale_word = "linear" if scale == "linear" else "log"
    d.text((L, 10), title or "", fill=ink, font=f_label)
    d.text((L + W, 10), f"{f_lo:g}–{f_hi:g} Hz, {scale_word} scale, top = high",
           fill=faint, font=f_small, anchor="ra")

    # Octave light key: the colour of each note, the same in every octave
    if octave_key:
        y0 = T + H + 40
        d.text((L, y0 + 8), "colour = note:", fill=ink, font=f_small, anchor="lm")
        x = L + 110
        rgb = octave_rgb(261.6256 * 2.0 ** (np.arange(12) / 12))
        for name, col in zip(_NOTE_NAMES, rgb):
            c = tuple(int(round(v * 255)) for v in col)
            d.rectangle((x, y0, x + 26, y0 + 14), fill=c)
            d.text((x + 13, y0 + 17), name, fill=ink, font=_font(11), anchor="mt")
            x += 32
    out.save(out_path)
    return out_path


def annotated_path(image_path):
    """look.png -> look.annotated.png"""
    p = Path(image_path)
    return p.with_name(p.stem + ".annotated" + p.suffix)


# ---------------------------------------------------------------------------
# Text annotation — the same sound described in words
# ---------------------------------------------------------------------------
# A picture costs a reader far more to take in than a few lines of text that
# say the same thing. This finds the notes in the sound and writes them out:
# when each starts, its name, its pitch in Hz and how far that is from equal
# temperament, plus one line on what kind of sound it is. It reads the audio
# itself, with longer windows than the picture uses, so its pitches are finer
# than the picture's bands (a few cents rather than about a semitone low down).
#
# How the notes are found:
#   1. Every 23 ms, list the peaks of the spectrum (a 0.19 s window; 0.37 s
#      below 160 Hz, where neighbouring semitones are only a few Hz apart).
#   2. In each frame pick fundamentals one at a time: the peak whose overtone
#      series (x1, x2, x3 ...) is strongest, preferring a fundamental to one of
#      its own overtones; then take that series away and look again. An
#      overtone shared with another note keeps whatever is left over, so the
#      upper note of a fifth is not lost in the lower note's overtones.
#   3. Join each pitch from frame to frame into a track.
#   4. A track starts where its own pitch first rises, timed with a short
#      window; a track that rises again sharply at an onset is a new note at
#      the same pitch (struck again).
#   5. Drop what is left of overtones, quiet companions and duplicates.

FUND_LO = 30.0        # Hz: lowest fundamental looked for, whatever the picture shows


def _peaks(col, freqs, floor):
    """Peaks of one spectrum column: (Hz, amplitude), refined between bins."""
    db = 20 * np.log10(col + 1e-12)
    i = np.where((col[1:-1] > col[:-2]) & (col[1:-1] >= col[2:])
                 & (col[1:-1] > floor))[0] + 1
    a, b, c = db[i - 1], db[i], db[i + 1]
    den = a - 2 * b + c
    p = np.where(den != 0, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    return freqs[i] + p * (freqs[1] - freqs[0]), 10 ** ((b - 0.25 * (a - c) * p) / 20)


def _fundamentals(pf, pa, f_lo, f_hi, rel=0.06, max_notes=6, nh=40, n_score=12,
                  max_cands=30):
    """Fundamentals in one frame's peaks, strongest first: [(Hz, strength)]."""
    order = np.argsort(pf)
    pf, amp = pf[order], pa[order].copy()
    lpf = np.log2(pf)
    inr = np.where((pf >= f_lo) & (pf <= f_hi))[0]
    if not len(inr):
        return []
    cand = inr[np.argsort(-amp[inr])[:max_cands]]
    c = pf[cand]
    h = np.arange(1, nh + 1)
    target = np.log2(c[:, None] * h[None, :])
    pos = np.clip(np.searchsorted(lpf, target), 1, len(lpf) - 1)
    near = np.where(np.abs(lpf[pos - 1] - target) < np.abs(lpf[pos] - target), pos - 1, pos)
    # within ~36 cents, a little wider for stiff strings, never reaching the next overtone
    ok = np.abs(lpf[near] - target) < np.minimum(0.03 + 0.004 * h, 0.6 / h)
    w = ok / np.sqrt(h) * (h <= n_score)          # judge by the first dozen, remove all
    alive = np.ones(len(c), bool)
    out, first = [], None
    for _ in range(max_notes):
        sc = (amp[near] * w).sum(axis=1) * alive * (amp[cand] > 0)
        bi = int(np.argmax(sc))
        best_s = sc[bi]
        if best_s <= 0:
            break
        # an overtone can outscore its own fundamental: prefer the fundamental
        # if it has strength of its own, not just the overtone's
        for ci in np.argsort(c):
            if c[ci] >= c[bi] or sc[ci] < 0.4 * best_s:
                continue
            r = c[bi] / c[ci]
            k = int(round(r))
            if 2 <= k <= 4 and abs(r - k) < 0.02:
                if (amp[near[ci]] * w[ci] * (h % k != 0)).sum() >= 0.25 * best_s:
                    bi, best_s = ci, sc[ci]
                    break
        first = best_s if first is None else first
        if best_s < rel * first:
            break
        out.append((float(c[bi]), float(best_s)))
        alive[bi] = False
        # take its overtones away; a shared one keeps what its neighbours
        # in the series don't explain
        ks = near[bi]
        a = amp[ks].copy()
        for i in np.where(ok[bi])[0]:
            if i == 0:
                amp[ks[0]] = 0.0
                continue
            nb = [a[x] for x in (i - 1, i + 1) if 0 <= x < nh and ok[bi, x]]
            expect = max(nb) * 3.0 if nb else a[i]
            amp[ks[i]] = a[i] - expect if a[i] > expect else 0.0
    return out


def _mag_stft(x, sr, nperseg, hop, nfft, fmax=None, fmin=0.0):
    """Magnitude spectrogram, as scipy's stft would give it (Hann window,
    no edge padding), but worked out a block of frames at a time and kept
    only between fmin and fmax, so a long recording fits in memory.
    Returns (freqs, frame centre times, magnitudes as float32)."""
    win = get_window("hann", nperseg)
    scale = 1.0 / win.sum()
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    rows = np.where((freqs >= fmin) & (freqs <= (fmax if fmax else freqs[-1])))[0]
    lo, hi = rows[0], rows[-1] + 1
    n = max(0, (len(x) - nperseg) // hop + 1)
    out = np.empty((hi - lo, n), dtype=np.float32)
    idx = np.arange(nperseg)
    for a in range(0, n, 512):
        b = min(n, a + 512)
        seg = x[(np.arange(a, b) * hop)[:, None] + idx[None, :]] * win
        out[:, a:b] = (np.abs(np.fft.rfft(seg, nfft, axis=1))[:, lo:hi] * scale).T
    return freqs[lo:hi], (np.arange(n) * hop + nperseg / 2) / sr, out


def find_notes(audio, sr, f_hi=8000.0):
    """The notes in a sound: a list of dicts with t0 (start, s), t1 (last
    heard, s), midi (fractional), spread (cents), and loudness and timing
    details used to describe the sound's character."""
    x = np.asarray(audio, dtype=np.float64)
    f_hi = min(f_hi, sr * 0.45)
    N = 1 << int(np.ceil(np.log2(0.17 * sr)))     # 4096 at 22.05 kHz
    hop = N // 8
    pad = N                                       # silence each side: edges get whole windows
    x = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    top = min(sr / 2, 11025.0)                    # spectra kept up to here
    f, t, M = _mag_stft(x, sr, N, hop, 2 * N, top)
    if M.size == 0 or M.max() <= 0:
        return [], None
    tc = t - pad / sr
    split = 160.0                                 # below this, use the long window
    f2, _, M2 = _mag_stft(np.concatenate([np.zeros(N // 2), x]), sr, 2 * N, hop,
                          4 * N, split * 1.1)
    gfloor = M.max() * 10 ** (-55 / 20)

    frames = []
    for j in range(M.shape[1]):
        col = M[:, j]
        floor = max(gfloor, col.max() * 10 ** (-45 / 20))
        pf, pa = _peaks(col, f, floor)
        if j < M2.shape[1]:
            qf, qa = _peaks(M2[:, j], f2, floor)
            hi = pf >= split
            pf = np.concatenate([qf[qf < split], pf[hi]])
            pa = np.concatenate([qa[qf < split], pa[hi]])
        keep = pf > 0
        frames.append(_fundamentals(pf[keep], pa[keep], FUND_LO, f_hi) if keep.any() else [])

    # join frames into tracks; a pitch may drop out for a few frames
    tracks, active = [], []
    for j, fl in enumerate(frames):
        used = set()
        for f0, s in sorted(fl, key=lambda z: -z[1]):
            m = 69 + 12 * np.log2(f0 / 440.0)
            cand = [tr for tr in active if id(tr) not in used
                    and abs(np.median(tr["m"][-6:]) - m) < 0.45]
            if cand:
                tr = min(cand, key=lambda tr: (j - tr["last"],
                                               abs(np.median(tr["m"][-6:]) - m)))
            else:
                tr = dict(j=[], m=[], s=[], last=j)
                tracks.append(tr)
                active.append(tr)
            tr["j"].append(j); tr["m"].append(m); tr["s"].append(s); tr["last"] = j
            used.add(id(tr))
        active = [tr for tr in active if j - tr["last"] <= 6]

    # onsets of any note, from a short window
    Ns = 512
    _, ts, Ms = _mag_stft(x, sr, Ns, 128, Ns)
    bb = np.maximum(np.diff(np.log(Ms + 1e-4), axis=1), 0).sum(axis=0)
    del Ms
    tb = ts[1:] - pad / sr
    thr = np.median(bb) + 3 * np.std(bb)
    onsets = tb[[i for i in range(1, len(bb) - 1)
                 if bb[i] > thr and bb[i] >= bb[i - 1] and bb[i] >= bb[i + 1]]]

    # each pitch's own loudness, from a medium window, to time its onsets
    Nm, lag = 2048 * N // 4096, 8
    fm, tm, Mm = _mag_stft(x, sr, Nm, Nm // 16, 2 * Nm, top)
    tm = tm - pad / sr

    def rise_at_pitch(f0):
        rows = []
        for hh in (1, 2):
            sel = np.where(np.abs(np.log2(fm / (f0 * hh) + 1e-12)) < 0.025)[0]
            if not len(sel):
                sel = [int(np.argmin(np.abs(fm - f0 * hh)))]
            rows.append(Mm[sel].max(axis=0))
        e = np.log(rows[0] + 0.5 * rows[1] + 1e-4)
        r = np.zeros_like(e)
        r[lag:-lag] = e[2 * lag:] - e[:-2 * lag]  # a rise across half a window, not a wobble
        return np.maximum(r, 0), e

    def snap(t0):
        if len(onsets):
            k = int(np.argmin(np.abs(onsets - t0)))
            if abs(onsets[k] - t0) < 0.06:
                return float(onsets[k])
        return float(t0)

    lead = 0.6 * N / sr
    notes = []
    for tr in tracks:
        js, ms, ss = map(np.asarray, (tr["j"], tr["m"], tr["s"]))
        if len(js) < 3:
            continue
        f0 = 440 * 2 ** ((np.median(ms) - 69) / 12)
        rise, level = rise_at_pitch(f0)
        tf = tc[js]

        def start_in(a, b):
            sel = np.where((tm >= a) & (tm <= b))[0]
            if not len(sel):
                return a, 0.0
            i = sel[int(np.argmax(rise[sel]))]
            top = rise[i]
            while i > sel[0] and rise[i - 1] >= 0.5 * top:     # back to where it began
                i -= 1
            return tm[min(i + lag, len(tm) - 1)], top

        # the first frame to hear the note may be centred up to half a window early
        half = (N if f0 < split else N / 2) / sr
        t_start, _ = start_in(tf[0] - lead, tf[0] + half)

        def swells(t0):
            """The note itself gets louder, not a neighbour leaking in."""
            after = ss[(tf >= t0) & (tf <= t0 + 0.3)]
            before = ss[(tf >= max(t0 - 0.3, t_start + 0.3)) & (tf < t0)]
            return len(after) and len(before) and after.max() > 1.4 * before.min()

        starts = [t_start]
        for i in np.where((tm > t_start + 0.12) & (tm < tf[-1]))[0]:
            if (rise[i] >= 0.45 and rise[i] == rise[max(0, i - 6):i + 7].max()
                    and len(onsets) and np.min(np.abs(onsets - tm[i])) < 0.04
                    and tm[i] - starts[-1] > 0.12 and swells(tm[i])):
                starts.append(tm[i])                    # struck again
        starts.append(np.inf)
        for a, b in zip(starts[:-1], starts[1:]):
            k = np.where((tf >= a - lead) & (tf < b - lead * 0.5))[0]
            if len(k) < 2:
                continue
            seg_s, seg_m = ss[k], ms[k]
            loud = seg_s >= seg_s.max() * 0.3
            mm = seg_m[loud]
            ipk = int(np.argmax(seg_s))
            t0 = max(0.0, snap(a))
            notes.append(dict(
                t0=t0, t1=float(tf[k[-1]]), midi=float(np.median(mm)),
                spread=float(np.percentile(mm, 75) - np.percentile(mm, 25)) * 50,
                peak=float(seg_s.max()), tpk=float(tf[k[ipk]]), jpk=int(js[k[ipk]])))

    notes = [n for n in notes if n["t1"] >= n["t0"] + 0.06]    # crumbs of a ringing note
    if notes:
        top = max(n["peak"] for n in notes)

        def loudest_at(n):
            return max(o["peak"] for o in notes
                       if o["t0"] - 0.1 <= n["tpk"] <= o["t1"] + 1e-9)
        notes = [n for n in notes
                 if n["peak"] > top * 0.03 and n["peak"] > 0.15 * loudest_at(n)]

    def overtone_of_another(n):
        for o in notes:
            r = 2 ** ((n["midi"] - o["midi"]) / 12)
            k = round(r)
            if (o is not n and 2 <= k <= 8 and abs(1200 * np.log2(r / k)) < 40
                    and o["t0"] - 0.1 < n["t0"] < o["t1"]
                    and n["peak"] < (0.8 if k == 2 else 0.5) * o["peak"]):
                return True
        return False
    notes = [n for n in notes if not overtone_of_another(n)]
    # one note found twice, or a faint shadow a semitone beside a note that
    # starts with it (a tremolo's sidebands): keep the louder
    kept = []
    for n in sorted(notes, key=lambda n: -n["peak"]):
        if not any(abs(n["t0"] - o["t0"]) < 0.06 and
                   (abs(n["midi"] - o["midi"]) < 0.4 or
                    (abs(n["midi"] - o["midi"]) < 1.5 and n["peak"] < 0.6 * o["peak"]))
                   for o in kept):
            kept.append(n)
    kept.sort(key=lambda n: (round(n["t0"], 2), n["midi"]))

    # attack and decay, from the finer-timed level at each note's pitch, up to
    # the next note's start (where a held note may stop) or one second
    starts = np.array([n["t0"] for n in kept])
    for n in kept:
        n["attack"] = n["decay"] = None
        f0 = 440 * 2 ** ((n["midi"] - 69) / 12)
        # several overtones together, so the beating of a chorused tone evens out
        hs = [hh for hh in range(1, 7) if f0 * hh < fm[-1]]
        level = np.log(sum(Mm[np.abs(np.log2(fm / (f0 * hh) + 1e-12)) < 0.025].max(axis=0)
                           if (np.abs(np.log2(fm / (f0 * hh) + 1e-12)) < 0.025).any()
                           else Mm[int(np.argmin(np.abs(fm - f0 * hh)))]
                           for hh in hs) + 1e-4)
        nxt = starts[starts > n["t0"] + 0.1]
        # a last note may be let go at any moment, so judge its decay only if alone
        end = min(nxt[0] - 0.1 if len(nxt) else (np.inf if len(kept) == 1 else n["t0"]),
                  n["t0"] + 1.0, n["t1"] + 0.1)
        end_att = min(nxt[0] if len(nxt) else np.inf, n["t0"] + 1.0, n["t1"] + 0.1)
        wa = np.where((tm >= n["t0"]) & (tm <= end_att))[0]
        if len(wa) > 3:
            n["attack"] = float(tm[wa[int(np.argmax(level[wa]))]] - n["t0"])
        w = np.where((tm >= n["t0"]) & (tm <= end))[0]
        if len(w) > 3:
            ip = w[int(np.argmax(level[w]))]
            d = w[w >= ip][: int(0.5 * sr / (Nm // 16))]
            if len(d) and tm[d[-1]] - tm[d[0]] >= 0.25:     # long enough to judge
                n["decay"] = float(np.polyfit(tm[d], level[d], 1)[0] * 20 / np.log(10))
    return kept, dict(M=M, f=f, tc=tc)


def _character(notes, spec):
    """One line on what kind of sound it is: how notes begin and end, how
    many overtones they carry, how much noise lies between them, and how
    steady the pitch is."""
    if not notes:
        return "no notes found"
    M, f, tc = spec["M"], spec["f"], spec["tc"]
    parts = []

    # how notes begin and what they do after their peak
    attacks = [n["attack"] for n in notes if n["attack"] is not None]
    decays = [n["decay"] for n in notes if n["decay"] is not None]
    att = float(np.median(attacks)) if attacks else None
    dec = float(np.median(decays)) if decays else None
    if att is not None and dec is not None:
        if att < 0.1 and dec < -12:
            parts.append("struck or plucked")
        elif att < 0.1 and dec < -6:
            parts.append("struck or plucked, ringing on")
        elif dec > -4:
            parts.append("held, like a blown or bowed note")
    if att is not None:
        word = "sharp" if att < 0.1 else "soft" if att < 0.25 else "slow"
        parts.append(f"{word} attack ({att:.2f} s to full loudness)")
    if dec is not None:
        word = ("die away quickly" if dec < -12 else "fade slowly" if dec < -4
                else "stay level")
        parts.append(f"after it notes {word} ({dec:.0f} dB per second)")

    # brightness: overtones within 30 dB of the strongest, at each note's peak
    counts = []
    for n in notes:
        col = M[:, n["jpk"]]
        f0 = 440 * 2 ** ((n["midi"] - 69) / 12)
        amps = []
        for h in range(1, 21):
            if f0 * h >= f[-1]:
                break
            sel = np.abs(np.log2((f + 1e-9) / (f0 * h))) < 0.02
            amps.append(col[sel].max() if sel.any() else 0.0)
        amps = np.asarray(amps)
        if amps.size and amps.max() > 0:
            counts.append(int((amps >= amps.max() * 10 ** (-30 / 20)).sum()))
    if counts:
        c = int(np.median(counts))
        word = "dark" if c <= 3 else "bright" if c >= 9 else "medium"
        parts.append(f"{word} tone ({c} overtones within 30 dB)")

    # noise: share of energy not on any sounding note's overtones
    df = f[1] - f[0]
    on = off = 0.0
    for j in range(0, M.shape[1], 2):
        sounding = [n for n in notes if n["t0"] <= tc[j] <= n["t1"]]
        if not sounding:
            continue
        p = M[:, j] ** 2
        mask = np.zeros(len(f), bool)
        for n in sounding:
            f0 = 440 * 2 ** ((n["midi"] - 69) / 12)
            for h in range(1, 41):
                fh = f0 * h
                if fh >= f[-1]:
                    break
                mask |= np.abs(f - fh) <= max(0.015 * fh, 3 * df)
        band = (f >= FUND_LO)
        on += p[mask & band].sum()
        off += p[~mask & band].sum()
    if on + off > 0:
        noise = 100 * off / (on + off)
        word = "clean" if noise < 5 else "some noise" if noise < 20 else "noisy"
        parts.append(f"{word} ({noise:.0f}% of the energy lies between the notes' overtones)")
        if noise >= 40:
            parts.append("much of it may not be notes at all, so read the list with care")

    # steadiness of pitch
    wob = float(np.median([n["spread"] for n in notes]))
    parts.append("steady pitch" if wob < 3 else f"pitch wavers about ±{wob:.0f} cents (vibrato or drift)")
    return "; ".join(parts)


def annotate_text(audio, sr, out_path, *, name, f_lo, f_hi, scale="log",
                  t_first=None, t_last=None, pictures=()):
    """Write NAME.annotated.txt: the notes of a sound in plain text."""
    notes, spec = find_notes(audio, sr)
    dur = len(audio) / sr
    lines = [f"{name}: notes found in the sound by imgaudio.py --annotate-text"]
    lines.append(f"sound: {dur:.2f} s at {sr} Hz; {len(notes)} notes; "
                 + _character(notes, spec))
    if pictures:
        scale_txt = ("log scale, each octave the same height" if scale != "linear"
                     else "linear scale, each Hz the same height")
        lines.append(
            f"picture: {' and '.join(pictures)} {'show' if len(pictures) > 1 else 'shows'} "
            f"the same sound: time runs left to "
            f"right from {t_first:.2f} to {t_last:.2f} s; pitch runs bottom to top from "
            f"{f_lo:g} to {f_hi:g} Hz ({scale_txt}). A note is a stack of lines; its "
            "pitch is the lowest line of the stack, read at the middle of its band.")
    lines.append("how to read: one line per note, in order of starting time; notes that "
                 "start together are a chord. Overtones are not listed. Cents are "
                 "hundredths of a semitone away from the named equal-tempered note "
                 "(+ sharp, - flat), with how sure the pitch is (±).")
    lines.append("")
    lines.append("start s   lasts s  note   Hz        cents")
    for n in notes:
        m = n["midi"]
        k = int(round(m))
        hz = 440 * 2 ** ((m - 69) / 12)
        cents = int(round(100 * (m - k))) + 0          # no "-0"
        sure = max(2, int(round(n["spread"])))
        where = ("  (below the picture)" if hz < f_lo else
                 "  (above the picture)" if hz > f_hi else "")
        lines.append(f"{n['t0']:7.2f}   {n['t1'] - n['t0']:6.2f}  "
                     f"{_NOTE_NAMES[k % 12] + str(k // 12 - 1):5s}  {hz:8.1f}  "
                     f"{cents:+4d} ±{sure}{where}")
    if not notes:
        lines.append("(none)")
    Path(out_path).write_text("\n".join(lines) + "\n")
    return out_path


def annotated_text_path(image_path):
    """look.png -> look.annotated.txt"""
    p = Path(image_path)
    return p.with_name(p.stem + ".annotated.txt")


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
def encode(image_path, audio_path, *, sr, rows, cols, f_lo, f_hi, col_sec,
           gamma, threshold, auto_prep=False, no_normalize=False,
           lossless=False, lens="raw", lens_params=None,
           color=False, color_width=32, freq_scale="log", **_):
    """Synthesize audio whose spectrogram is the image, or raw-byte encode."""
    if lossless:
        # Raw byte mode: wrap the image file's bytes as 8-bit unsigned PCM.
        # Audio sounds like noise; recovery is byte-identical.
        with open(image_path, "rb") as fp:
            data = fp.read()
        samples = np.frombuffer(data, dtype=np.uint8)
        wavfile.write(audio_path, sr, samples)
        print(f"encoded (lossless): {audio_path}  "
              f"({len(samples)/sr:.3f}s, {len(samples)} bytes)")
        return

    img = Image.open(image_path).convert("L")
    img = img.resize((cols, rows), Image.LANCZOS)
    grid = np.asarray(img, dtype=np.float32) / 255.0
    grid = grid[::-1]                     # flip: top of image = high freq

    if auto_prep:
        grid, info = _auto_prep(grid)
        print(f"  auto-prep: clip=[{info['clip_lo']:.2f},{info['clip_hi']:.2f}] "
              f"median={info['median_after']:.2f} inverted={info['inverted']}")

    if color:
        write_chroma(image_path, str(chroma_path(audio_path)),
                     width=color_width,
                     inverted=bool(auto_prep and info["inverted"]))

    # --- Apply lens (biological pattern-finder), if one is selected ---------
    if lens and lens != "raw":
        lens_mod = _LENSES.get(lens)
        if lens_mod is None:
            raise SystemExit(f"unknown lens '{lens}'. "
                             f"available: {sorted(_LENSES.keys())}")
        print(f"  lens: {lens} ({getattr(lens_mod, 'DESCRIPTION', '')})")
        grid = lens_mod.analyze(grid, lens_params or {})
        grid = np.clip(grid, 0.0, 1.0).astype(np.float32)

    grid = grid ** gamma                  # gamma curve for cleaner read-back

    if f_hi >= sr / 2:
        raise SystemExit(f"--f-hi {f_hi:.0f} Hz needs a sample rate above "
                         f"{2 * f_hi:.0f} Hz; try --sr 44100")
    freqs = freq_grid(f_lo, f_hi, rows, freq_scale)
    samples_per_col = int(col_sec * sr)
    frame_len = samples_per_col * 2       # overlap-add: frames overlap 50%
    hop = samples_per_col
    window = np.hanning(frame_len).astype(np.float32)

    total = hop * cols + frame_len
    out = np.zeros(total, dtype=np.float32)

    # Per-oscillator phase, advanced every hop so sine waves stay continuous
    phase = np.zeros(rows, dtype=np.float64)
    t_frame = np.arange(frame_len) / sr

    for col in range(cols):
        amps = grid[:, col]
        active = amps > threshold

        if np.any(active):
            a = amps[active][:, None]
            f = freqs[active][:, None]
            ph = phase[active][:, None]
            sines = a * np.sin(2 * np.pi * f * t_frame + ph).astype(np.float32)
            frame = sines.sum(axis=0) * window
            start = col * hop
            out[start:start + frame_len] += frame

        phase += 2 * np.pi * freqs * (hop / sr)
        phase %= 2 * np.pi

    # Normalize to -3 dB peak (per-file) OR use deterministic scaling
    # (per-file normalization makes a/b diffs imperfect; --no-normalize fixes this)
    if no_normalize:
        # Fixed scaling: divide by sqrt(rows) ≈ theoretical max for orthogonal sines
        out = out / np.sqrt(rows) * 0.5
        out = np.clip(out, -0.95, 0.95)
    else:
        peak = float(np.max(np.abs(out)))
        if peak > 0:
            out = out / peak * 0.707
    fade = int(0.1 * sr)
    out[:fade]  *= np.linspace(0, 1, fade, dtype=np.float32)
    out[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

    wavfile.write(audio_path, sr, (out * 32767).astype(np.int16))
    print(f"encoded: {audio_path}  ({len(out)/sr:.2f}s, "
          f"{rows} bins × {cols} steps, {f_lo:.0f}–{f_hi:.0f} Hz {freq_scale})")

# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def decode(audio_path, image_path, *, sr, rows, cols, f_lo, f_hi,
           lossless=False, color=False, lens="raw", lens_params=None,
           palette="grey", freq_scale="log", annotate_out=False,
           annotate_text_out=False, **_):
    """Recover the image from the audio via STFT magnitude, or raw-byte decode."""
    if lossless:
        # Raw byte mode: WAV samples ARE the image bytes. Strip header, write file.
        _, samples = wavfile.read(audio_path)
        if samples.dtype != np.uint8:
            samples = samples.astype(np.uint8)
        with open(image_path, "wb") as fp:
            fp.write(samples.tobytes())
        # Verify by inspecting the magic bytes
        magic = samples[:3].tobytes().hex()
        valid = magic == "ffd8ff"
        print(f"decoded (lossless): {image_path}  ({len(samples)} bytes, "
              f"magic={magic} {'✓ JPEG' if valid else '⚠ NOT a JPEG'})")
        return

    sr_in, audio = wavfile.read(audio_path)
    if sr_in != sr:
        print(f"warning: file sample rate {sr_in} != configured {sr}",
              file=sys.stderr)
        sr = sr_in
    audio = audio.astype(np.float32)
    if audio.dtype == np.int16 or audio.max() > 1.5:
        audio /= np.iinfo(np.int16).max
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # STFT — aim for roughly `cols` time frames across the whole signal
    nperseg = 1024
    hop = max(1, (len(audio) - nperseg) // cols)
    f, _, Z = stft(audio, fs=sr, nperseg=nperseg,
                   noverlap=nperseg - hop, window="hann",
                   boundary=None, padded=False)
    mag = np.abs(Z)

    # Resample STFT (linear freq) onto the encoder's log-frequency grid
    target_freqs = freq_grid(f_lo, f_hi, rows, freq_scale)
    img_arr = np.zeros((rows, mag.shape[1]), dtype=np.float32)
    for i, tf in enumerate(target_freqs):
        idx = int(np.argmin(np.abs(f - tf)))
        img_arr[i] = mag[idx]

    # Resize to the canonical (rows, cols) grid
    if img_arr.shape[1] != cols:
        img_arr = np.asarray(
            Image.fromarray(img_arr).resize((cols, rows), Image.LANCZOS)
        )
        # Lanczos overshoots below zero around sharp tones; a magnitude under
        # -1/50 makes log1p below return NaN and blanks the whole picture.
        img_arr = np.maximum(img_arr, 0.0)

    # Log compression + normalize, then flip back to image orientation
    img_arr = np.log1p(img_arr * 50)
    img_arr -= img_arr.min()
    if img_arr.max() > 0:
        img_arr /= img_arr.max()

    # --- Apply lens to the audio's spectrogram, if one is selected ----------
    if lens and lens != "raw":
        lens_mod = _LENSES.get(lens)
        if lens_mod is None:
            raise SystemExit(f"unknown lens '{lens}'. "
                             f"available: {sorted(_LENSES.keys())}")
        print(f"  lens: {lens} ({getattr(lens_mod, 'DESCRIPTION', '')}) "
              f"[applied to spectrogram]")
        img_arr = lens_mod.analyze(img_arr.astype(np.float32),
                                   lens_params or {})
        img_arr = np.clip(img_arr, 0.0, 1.0)

    # Convert to uint8, flip to image orientation
    img_arr = (img_arr * 255).astype(np.uint8)[::-1]

    sidecar = chroma_path(audio_path)
    if color and sidecar.exists():
        rgb = apply_chroma(img_arr, str(sidecar))
        Image.fromarray(rgb, "RGB").save(image_path)
        print(f"decoded: {image_path}  ({cols}×{rows}, recoloured from "
              f"{sidecar.name})")
    elif palette == "octave":
        if color:
            print(f"  no chroma sidecar at {sidecar} — using octave light",
                  file=sys.stderr)
        Image.fromarray(apply_octave(img_arr, f_lo, f_hi, freq_scale), "RGB").save(image_path)
        print(f"decoded: {image_path}  ({cols}×{rows}, octave light palette)")
    else:
        if color:
            print(f"  no chroma sidecar at {sidecar} — writing grayscale",
                  file=sys.stderr)
        Image.fromarray(img_arr, mode="L").save(image_path)
        print(f"decoded: {image_path}  ({cols}×{rows})")

    if annotate_out:
        shown = np.asarray(Image.open(image_path))
        n = len(audio)
        out = annotate(shown, annotated_path(image_path),
                       t_first=(nperseg / 2) / sr, t_last=(n - nperseg / 2) / sr,
                       f_lo=f_lo, f_hi=f_hi, scale=freq_scale,
                       octave_key=(palette == "octave" and shown.ndim == 3
                                   and not (color and sidecar.exists())),
                       title=Path(audio_path).name)
        print(f"annotated: {out}  (rulers for reading; re-encode the plain picture)")

    if annotate_text_out:
        pics = [Path(image_path).name]
        if annotate_out:
            pics.append(annotated_path(image_path).name)
        out = annotate_text(audio, sr, annotated_text_path(image_path),
                            name=Path(audio_path).name, f_lo=f_lo, f_hi=f_hi,
                            scale=freq_scale, t_first=(nperseg / 2) / sr,
                            t_last=(len(audio) - nperseg / 2) / sr, pictures=pics)
        print(f"annotated: {out}  (the notes in words)")


# ---------------------------------------------------------------------------
# Round-trip convenience: encode, decode, save a comparison plot
# ---------------------------------------------------------------------------
def roundtrip(image_path, *, rows, cols, out_dir=None, **kw):
    """Run encode + decode, save audio + recovered image + 3-panel plot."""
    img = Path(image_path)
    # By default outputs go next to the input image; --out-dir places them
    # in the given directory (created if needed) using just the filename stem.
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = out / img.stem
    else:
        stem = img.with_suffix("")

    audio_path = f"{stem}_audio.wav"
    plot_path  = f"{stem}_roundtrip.png"

    # In lossless mode the recovered file must keep the original extension
    # (otherwise the byte stream — a JPEG — would be saved with the wrong name)
    if kw.get("lossless"):
        recovered_path = f"{stem}_recovered{img.suffix}"
    else:
        recovered_path = f"{stem}_recovered.png"

    encode(image_path, audio_path, rows=rows, cols=cols, **kw)
    decode(audio_path, recovered_path, rows=rows, cols=cols, **kw)

    if kw.get("lossless"):
        # Verify byte-identical recovery
        import hashlib
        def md5(p):
            with open(p, "rb") as fp:
                return hashlib.md5(fp.read()).hexdigest()
        orig_hash = md5(image_path)
        rec_hash  = md5(recovered_path)
        match = orig_hash == rec_hash
        print(f"verification: {'✓' if match else '✗'} "
              f"md5(original)={orig_hash[:16]}... "
              f"md5(recovered)={rec_hash[:16]}...")
        if not match:
            print("WARNING: files differ!")
        return  # no spectrogram comparison plot in lossless mode

    # Plot original / spectrogram-of-audio / recovered, side by side
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping comparison plot.")
        return

    sr, audio = wavfile.read(audio_path)
    audio = audio.astype(np.float32) / 32767

    orig = np.asarray(
        Image.open(image_path).convert("L").resize((cols, rows), Image.LANCZOS)
    )
    rec = np.asarray(Image.open(recovered_path))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].imshow(orig, cmap="gray", aspect="auto")
    axes[0].set_title(f"1. original (downsampled to {rows}×{cols} grid)")
    axes[0].axis("off")

    axes[1].specgram(audio, NFFT=1024, Fs=sr, noverlap=512, cmap="magma")
    axes[1].set_yscale("log")
    axes[1].set_ylim(kw["f_lo"], kw["f_hi"])
    axes[1].set_title(f"2. spectrogram of generated audio "
                      f"({len(audio)/sr:.1f}s)")
    axes[1].set_ylabel("Hz"); axes[1].set_xlabel("time (s)")

    axes[2].imshow(rec, cmap="gray", aspect="auto")
    axes[2].set_title("3. image recovered from audio via STFT")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=110, bbox_inches="tight")
    print(f"plot:    {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="imgaudio.py",
        description="Image ↔ audio via spectrogram synthesis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Shared params (apply to all subcommands)
    p.add_argument("--sr",       type=int,   default=DEFAULTS["sr"],
                   help="sample rate of the audio, Hz. Pitches above half of "
                        "it cannot exist, so use 44100 with --f-hi above 11000")
    p.add_argument("--rows",     type=int,   default=DEFAULTS["rows"],
                   help="picture height: pitch steps between --f-lo and --f-hi")
    p.add_argument("--cols",     type=int,   default=DEFAULTS["cols"],
                   help="picture width: time steps")
    p.add_argument("--f-lo",     type=float, default=DEFAULTS["f_lo"],
                   help="pitch of the bottom row, Hz")
    p.add_argument("--f-hi",     type=float, default=DEFAULTS["f_hi"],
                   help="pitch of the top row, Hz (encode requires it below --sr/2)")
    p.add_argument("--col-sec",  type=float, default=DEFAULTS["col_sec"],
                   help="encode: seconds per column; the sound lasts cols x this")
    p.add_argument("--gamma",    type=float, default=DEFAULTS["gamma"],
                   help="contrast curve on brightness; above 1 darkens the "
                        "midtones and quietens faint detail")
    p.add_argument("--threshold",type=float, default=DEFAULTS["threshold"],
                   help="brightness (0-1) below which a pixel is silent")
    p.add_argument("--auto-prep", action="store_true",
                   help="Adaptive preprocessing: percentile-clip, invert if "
                        "mostly bright, mild gamma. Useful for batches of "
                        "frames with varying exposure (day/night, etc).")
    p.add_argument("--no-normalize", action="store_true",
                   help="Use deterministic scaling instead of per-file peak "
                        "normalization. Needed for clean null-test diffs "
                        "between two encoded files.")
    p.add_argument("--lossless", action="store_true",
                   help="Raw-byte mode: encode the image file's bytes "
                        "directly as 8-bit PCM. Audio sounds like noise but "
                        "decoding recovers the byte-identical original. "
                        "Other spectrogram parameters are ignored.")

    lens_help = "Apply a biological-pattern lens to the image before synthesis. "
    if _LENSES:
        lens_help += "Available: " + ", ".join(
            f"{n} ({getattr(m, 'DESCRIPTION', '')})"
            for n, m in sorted(_LENSES.items())) + "."
    else:
        lens_help += "(No lenses found in lenses/ directory.)"
    p.add_argument("--color", action="store_true",
                   help="write a small chroma sidecar on encode and reapply "
                        "it on decode. The audio still carries luminance "
                        "only; colour rides alongside in ~450 bytes")
    p.add_argument("--color-width", type=int, default=32,
                   help="chroma sidecar width in pixels; height follows the "
                        "source aspect. 32 is near-lossless to the eye")
    p.add_argument("--freq-scale", choices=["log", "linear"], default="log",
                   help="how rows map to pitch: log gives every octave the same "
                        "height (how we hear); linear gives every Hz the same "
                        "height (how most spectrogram tools draw). Encode and "
                        "decode must use the same scale")
    p.add_argument("--annotate", action="store_true",
                   help="decode only: also write NAME.annotated.png with a time "
                        "ruler, a frequency ruler and note names, for reading the "
                        "picture precisely. The plain picture is unchanged")
    p.add_argument("--annotate-text", action="store_true",
                   help="decode only: also write NAME.annotated.txt listing the "
                        "notes in the sound (start, name, Hz, cents) and what "
                        "kind of sound it is, in a few lines of plain text. "
                        "Cheaper for a reader than the picture; works with or "
                        "without --annotate")
    p.add_argument("--palette", choices=["grey", "octave"], default="grey",
                   help="decode only: 'octave' colours each pitch by raising "
                        "it by octaves into visible light, so every sound "
                        "shares one palette. A --color sidecar takes priority. "
                        "Encode reads brightness, so re-encode the grey picture")
    p.add_argument("--lens", default="raw", help=lens_help)
    p.add_argument("--lens-params", default="",
                   help="Comma-separated key=value pairs passed to the lens, "
                        "e.g. --lens-params threshold=0.3,blur=2")

    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="image → audio")
    pe.add_argument("image", help="picture to encode (any format PIL reads)")
    pe.add_argument("audio", help="WAV to write")

    pd = sub.add_parser("decode", help="audio → image")
    pd.add_argument("audio", help="WAV to read (from encode, or any recording)")
    pd.add_argument("image", help="PNG to write; --annotate adds NAME.annotated.png, "
                                  "--annotate-text adds NAME.annotated.txt")

    pr = sub.add_parser("roundtrip", help="encode + decode + plot")
    pr.add_argument("image")
    pr.add_argument("--out-dir", default=None,
                    help="Directory for output files (created if needed). "
                         "Defaults to the same directory as the input image.")

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Hyphenated CLI flags become underscored attributes
    kw = dict(sr=args.sr, rows=args.rows, cols=args.cols,
              f_lo=args.f_lo, f_hi=args.f_hi, col_sec=args.col_sec,
              gamma=args.gamma, threshold=args.threshold,
              auto_prep=args.auto_prep, no_normalize=args.no_normalize,
              lossless=args.lossless,
              lens=args.lens, lens_params=_parse_lens_params(args.lens_params),
              color=args.color, color_width=args.color_width,
              palette=args.palette, freq_scale=args.freq_scale,
              annotate_out=args.annotate, annotate_text_out=args.annotate_text)

    if args.cmd == "encode":
        encode(args.image, args.audio, **kw)
    elif args.cmd == "decode":
        decode(args.audio, args.image, **kw)
    elif args.cmd == "roundtrip":
        roundtrip(args.image, out_dir=args.out_dir, **kw)


if __name__ == "__main__":
    main()
