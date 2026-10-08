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


def catalog():
    """Lenses and scales the page can offer, read from the scripts themselves."""
    lenses = [{"name": "raw", "desc": "plain brightness, the honest spectrogram"}]
    for name, mod in sorted(imgaudio._LENSES.items()):
        lenses.append({"name": name, "desc": getattr(mod, "DESCRIPTION", "")})
    return json.dumps({
        "lenses": lenses,
        "scales": {"10": list(notate.SCALES), "60": list(notate.RATIO_SCALES)},
    })


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


def make(opts_json):
    """photo -> sound.wav -> picture.png, sound.wav -> melody.wav + .mid,
    then melody.wav -> melody.png"""
    o = json.loads(opts_json)
    rows, cols = int(o["rows"]), int(o["cols"])
    lens_params = ""
    if o["lens"] == "spectral":
        # README: quadrant breaks the mirror symmetry; the spectrum is square.
        lens_params, cols = "mode=quadrant", rows

    grid = ["--rows", str(rows), "--cols", str(cols)]
    color = ["--color"] if o["color"] else []

    log = _run(imgaudio.main,
               ["--auto-prep", *grid, "--lens", o["lens"],
                "--lens-params", lens_params, *color,
                "encode", f"{WORK}/photo.jpg", f"{WORK}/sound.wav"])

    log += _run(imgaudio.main,
                [*grid, *color, "decode", f"{WORK}/sound.wav", f"{WORK}/picture.png"])

    base = str(o["base"])
    log += _run(_notate_main,
                ["notes", f"{WORK}/sound.wav", f"{WORK}/melody.wav",
                 "--midi", f"{WORK}/melody.mid",
                 "--base", base, "--scale", o["scale"], "--root", o["root"],
                 "--bpm", str(o["bpm"]), "--grid", str(o["grid"]),
                 "--voices", str(o["voices"]), "--program", str(o["program"]),
                 "--dry-mix", str(o["dry_mix"]), "--verify"])

    # Read a picture back out of the music too. Greyscale on purpose: the
    # colour sidecar would paint the photo onto whatever gets decoded, even
    # notes that carry none of it. Only --dry-mix brings the photo back.
    log += _run(imgaudio.main,
                [*grid, "--sr", "44100", "decode",
                 f"{WORK}/melody.wav", f"{WORK}/melody.png"])
    return log
