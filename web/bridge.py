"""bridge.py — glue between the web page and the unmodified CLI scripts.

Runs inside Pyodide. It calls imgaudio.main() and notate.main() with the same
arguments you would type on the command line, so the browser and the CLI
always share one implementation.
"""

import contextlib
import io
import json
import sys

import numpy as np
from PIL import Image
from scipy.signal import _spectral_py

import imgaudio
import notate

WORK = "/work"


# --- browser fix: STFT framing on 32-bit WebAssembly ------------------------
# scipy builds every window as a view of shape (n_samples, nperseg) and keeps
# every hop-th row. The view costs no memory, but numpy still checks its
# nominal size, and on wasm32 a 20 s clip at nperseg=4096 exceeds 2 GiB, so
# notate.py fails with "array is too big". Building the view with the hop
# already in the strides gives identical numbers with a nominal size that fits.
def _fft_helper_wasm(x, win, detrend_func, nperseg, noverlap, nfft, sides):
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        step = nperseg - noverlap
        n = (x.shape[-1] - nperseg) // step + 1
        result = np.lib.stride_tricks.as_strided(
            x, shape=x.shape[:-1] + (n, nperseg),
            strides=x.strides[:-1] + (step * x.strides[-1], x.strides[-1]),
            writeable=False)
    result = detrend_func(result)
    result = win * result
    if sides == "twosided":
        return _spectral_py.sp_fft.fft(result, n=nfft)
    return _spectral_py.sp_fft.rfft(result.real, n=nfft)


_spectral_py._fft_helper = _fft_helper_wasm


# Lenses that keep each picture row at its own pitch, so colouring rows by
# pitch still means something. spectral and phyllotaxis rearrange the picture.
PITCH_LENSES = {"raw", "edges", "fractal", "reveal"}


def catalog():
    """Lenses, scales and palette legend, read from the scripts themselves."""
    lenses = [{"name": "raw", "desc": "plain brightness, the honest spectrogram"}]
    for name, mod in sorted(imgaudio._LENSES.items()):
        lenses.append({"name": name, "desc": getattr(mod, "DESCRIPTION", "")})
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    rgb = imgaudio.octave_rgb(261.6256 * 2.0 ** (np.arange(12) / 12))
    legend = [{"note": n, "rgb": "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in col)}
              for n, col in zip(names, rgb)]
    return json.dumps({
        "lenses": lenses,
        "pitch_lenses": sorted(PITCH_LENSES),
        "octave_legend": legend,
        "scales": {"10": list(notate.SCALES), "60": list(notate.RATIO_SCALES)},
    })


# Which part of the sound a run reads. The page clips recordings to the
# sample rate given here, so "full" keeps 44.1 kHz for detail up to 20 kHz.
RANGES = {
    "standard": dict(f_lo=80, f_hi=8000, sr=22050),
    "low":      dict(f_lo=40, f_hi=4000, sr=22050),   # README: squashed at the bottom
    "full":     dict(f_lo=40, f_hi=20000, sr=44100),  # hidden images often sit up high
}


def _band(o):
    """Frequency range and scale for a run, as a dict plus imgaudio flags."""
    b = dict(RANGES.get(o.get("range", "standard"), RANGES["standard"]))
    b["scale"] = "linear" if o.get("fscale") == "linear" else "log"
    b["args"] = ["--f-lo", str(b["f_lo"]), "--f-hi", str(b["f_hi"]),
                 "--freq-scale", b["scale"]]
    return b


PHOTO_BAND = dict(f_lo=80, f_hi=8000, sr=22050, scale="log", args=[])


def _octave(src, dst, band):
    """Save a pitch-coloured copy of a grey decoded picture (imgaudio's
    octave light palette). The grey original stays for re-encoding."""
    grey = np.asarray(Image.open(src).convert("L"))
    rgb = imgaudio.apply_octave(grey, band["f_lo"], band["f_hi"], band["scale"])
    Image.fromarray(rgb, "RGB").save(dst)


def _run(fn, argv):
    """Run a CLI entry point, capturing its output; raise on a non-zero exit."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            fn(argv)
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"{e.code}\n{buf.getvalue()}")
    return buf.getvalue()


def _notate_main(argv):
    sys.argv = ["notate.py"] + argv      # notate.main() reads sys.argv
    notate.main()


def _grid(o):
    """--rows/--cols (and --lens-params) shared by every step of one run."""
    rows, cols = int(o["rows"]), int(o["cols"])
    lens_params = ""
    if o["lens"] == "spectral":
        # README: quadrant breaks the mirror symmetry; the spectrum is square.
        lens_params, cols = "mode=quadrant", rows
    return ["--rows", str(rows), "--cols", str(cols)], lens_params


def _music(o, src, grid, band=PHOTO_BAND):
    """src -> melody.wav + .mid (with --verify), then melody.wav -> melody.png."""
    log = _run(_notate_main,
               ["notes", src, f"{WORK}/melody.wav",
                "--midi", f"{WORK}/melody.mid",
                "--base", str(o["base"]), "--scale", o["scale"], "--root", o["root"],
                "--bpm", str(o["bpm"]), "--grid", str(o["grid"]),
                "--voices", str(o["voices"]), "--program", str(o["program"]),
                "--dry-mix", str(o["dry_mix"]), "--f-lo", str(min(80, band["f_lo"])),
                "--verify"])

    # Read a picture back out of the music too. Greyscale on purpose: the
    # colour sidecar would paint the photo onto whatever gets decoded, even
    # notes that carry none of it. Only --dry-mix brings the source back.
    log += _run(imgaudio.main,
                [*grid, *band["args"], "--sr", "44100", "decode",
                 f"{WORK}/melody.wav", f"{WORK}/melody.png"])
    _octave(f"{WORK}/melody.png", f"{WORK}/melody-octave.png", band)
    return log


def make(opts_json):
    """Start from a photo: photo -> sound.wav -> picture.png, then the music."""
    o = json.loads(opts_json)
    grid, lens_params = _grid(o)
    color = ["--color"] if o["color"] else []

    log = _run(imgaudio.main,
               ["--auto-prep", *grid, "--lens", o["lens"],
                "--lens-params", lens_params, *color,
                "encode", f"{WORK}/photo.jpg", f"{WORK}/sound.wav"])
    log += _run(imgaudio.main,
                [*grid, *color, "decode", f"{WORK}/sound.wav", f"{WORK}/picture.png"])
    return log + _music(o, f"{WORK}/sound.wav", grid)


def make_from_sound(opts_json):
    """Start from a recording (README: "Starting from audio instead"):
    input.wav -> picture.png, picture.png -> resound.wav (the round trip),
    then the music from the recording itself."""
    o = json.loads(opts_json)
    grid, lens_params = _grid(o)
    band = _band(o)
    sr = ["--sr", str(band["sr"])]

    log = _run(imgaudio.main,
               [*grid, *band["args"], *sr, "--lens", o["lens"], "--lens-params", lens_params,
                "decode", f"{WORK}/input.wav", f"{WORK}/picture.png"])
    if o["lens"] in PITCH_LENSES:
        _octave(f"{WORK}/picture.png", f"{WORK}/picture-octave.png", band)
    # Play the picture back as sound. No --auto-prep: a decoded picture is
    # already a spectrogram, so it must not be clipped or inverted like a photo.
    log += _run(imgaudio.main,
                [*grid, *band["args"], *sr, "encode", f"{WORK}/picture.png", f"{WORK}/resound.wav"])
    return log + _music(o, f"{WORK}/input.wav", grid, band)
