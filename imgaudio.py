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
from scipy.signal import stft


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
# Encoder
# ---------------------------------------------------------------------------
def encode(image_path, audio_path, *, sr, rows, cols, f_lo, f_hi, col_sec,
           gamma, threshold, auto_prep=False, no_normalize=False,
           lossless=False, lens="raw", lens_params=None, **_):
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

    freqs = np.geomspace(f_lo, f_hi, rows)
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
          f"{rows} bins × {cols} steps, {f_lo:.0f}–{f_hi:.0f} Hz log)")

# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def decode(audio_path, image_path, *, sr, rows, cols, f_lo, f_hi,
           lossless=False, lens="raw", lens_params=None, **_):
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
    target_freqs = np.geomspace(f_lo, f_hi, rows)
    img_arr = np.zeros((rows, mag.shape[1]), dtype=np.float32)
    for i, tf in enumerate(target_freqs):
        idx = int(np.argmin(np.abs(f - tf)))
        img_arr[i] = mag[idx]

    # Resize to the canonical (rows, cols) grid
    if img_arr.shape[1] != cols:
        img_arr = np.asarray(
            Image.fromarray(img_arr).resize((cols, rows), Image.LANCZOS)
        )

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

    Image.fromarray(img_arr, mode="L").save(image_path)
    print(f"decoded: {image_path}  ({cols}×{rows})")


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
    p.add_argument("--sr",       type=int,   default=DEFAULTS["sr"])
    p.add_argument("--rows",     type=int,   default=DEFAULTS["rows"])
    p.add_argument("--cols",     type=int,   default=DEFAULTS["cols"])
    p.add_argument("--f-lo",     type=float, default=DEFAULTS["f_lo"])
    p.add_argument("--f-hi",     type=float, default=DEFAULTS["f_hi"])
    p.add_argument("--col-sec",  type=float, default=DEFAULTS["col_sec"])
    p.add_argument("--gamma",    type=float, default=DEFAULTS["gamma"])
    p.add_argument("--threshold",type=float, default=DEFAULTS["threshold"])
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
    p.add_argument("--lens", default="raw", help=lens_help)
    p.add_argument("--lens-params", default="",
                   help="Comma-separated key=value pairs passed to the lens, "
                        "e.g. --lens-params threshold=0.3,blur=2")

    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="image → audio")
    pe.add_argument("image"); pe.add_argument("audio")

    pd = sub.add_parser("decode", help="audio → image")
    pd.add_argument("audio"); pd.add_argument("image")

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
              lens=args.lens, lens_params=_parse_lens_params(args.lens_params))

    if args.cmd == "encode":
        encode(args.image, args.audio, **kw)
    elif args.cmd == "decode":
        decode(args.audio, args.image, **kw)
    elif args.cmd == "roundtrip":
        roundtrip(args.image, out_dir=args.out_dir, **kw)


if __name__ == "__main__":
    main()
