#!/usr/bin/env python3
"""
render.py — render a recipe end to end.

A recipe is a small TOML file describing every stage of the pipeline for a
set of sources — images, recordings, or both. render.py runs the existing
tools (imgaudio.py and notate.py) with matching parameters, so encode and
decode can never disagree about geometry, the most common way a round-trip
silently goes wrong.

Stages, for each source NAME:

    encode        NAME.wav  (+ NAME.chroma.png with color)     image sources
    convert       NAME.wav  mono, resampled, split into tiles  audio sources
    decode        NAME.png                  photo round-trip / spectrogram
    notate        NAME.notes.wav  (+ NAME.notes.mid, NAME.verify.json)
    notes_decode  NAME.notes.png            picture of the performance
    report        report.csv                one fidelity row per source/tile
    sheets        NAME.sheet.png            tiles of a long recording, in order
    assemble      one joined WAV            once every source is done

Audio sources (anything ffmpeg reads: wav, mp3, m4a, flac, ogg...) skip
encode — there's no hidden photo to recover, so decode shows the sound's own
structure. Long recordings are split into equal tiles (30 s by default) named
NAME.t001, NAME.t002, ... and the frequency band is detected automatically
from where the recording's energy actually sits.

Sections you write are stages you get: encode/convert and decode always run;
notate and notes_decode run when [notate] is present; assemble runs when
[assemble] is present. Any stage can be switched off with `enabled = false`.

Resumable: finished outputs are skipped. Writes are atomic — each job renders
into a private temp folder and moves results into place only when complete,
so a killed job never leaves a half-written file that a resumed run would
mistake for finished.

A hash of each stage's settings lives in .recipe-hash. Changing a parameter
invalidates that stage and everything downstream of it: change [notate] and
the encodes are kept; change [encode] and everything reruns.

Usage:
    python3 render.py recipes/year.toml              render everything
    python3 render.py recipes/year.toml --jobs 4     parallel on this machine
    python3 render.py recipes/year.toml --dry-run    show the plan, run nothing

Fleet (shared output folder, e.g. NFS):
    python3 render.py recipes/year.toml --prepare        once, before launching
    python3 render.py recipes/year.toml --shard 3/40     on each worker
    python3 render.py recipes/year.toml                  once more to assemble

Dependencies: Python 3.11+ (tomllib), numpy, scipy, Pillow, ffmpeg + ffprobe.
"""

import argparse
import concurrent.futures as cf
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

import math

import numpy as np
from PIL import Image
from scipy.io import wavfile

HERE = Path(__file__).resolve().parent
PY = sys.executable                 # same interpreter, so the venv carries over
IMGAUDIO = str(HERE / "imgaudio.py")
NOTATE = str(HERE / "notate.py")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac",
              ".wma", ".aif", ".aiff"}

# Geometry for audio sources. There's no encode stage to match, so these are
# a free choice about resolution. f_lo/f_hi "auto" measures the recording.
AUDIO_DEFAULTS = {"sr": 22050, "rows": 300, "cols": 600,
                  "f_lo": "auto", "f_hi": "auto",
                  "tile": 30, "sheet_columns": 4}

# 10-stop approximations of matplotlib's perceptually uniform colour maps.
# Used for grayscale decodes; a colour sidecar, when present, takes priority.
PALETTES = {
    "viridis": ["#440154", "#482878", "#3e4989", "#31688e", "#26828e",
                "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
    "magma":   ["#000004", "#180f3d", "#440f76", "#721f81", "#9e2f7f",
                "#cd4071", "#f1605d", "#fd9668", "#feca8d", "#fcfdbf"],
    "inferno": ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60",
                "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"],
    "gray":    None,
}

# Keys each section accepts. Anything else is almost certainly a typo, and a
# typo silently falling back to a default is exactly what this tool prevents.
SECTIONS = {
    "source":       {"frames", "audio"},
    "audio":        {"sr", "rows", "cols", "f_lo", "f_hi", "tile",
                     "sheet_columns"},
    "output":       {"dir"},
    "encode":       {"sr", "rows", "cols", "f_lo", "f_hi", "col_sec", "gamma",
                     "threshold", "auto_prep", "no_normalize", "color",
                     "color_width", "lens", "lens_params"},
    "decode":       {"enabled", "lens", "lens_params", "palette"},
    "notate":       {"enabled", "base", "scale", "root", "bpm", "grid",
                     "voices", "f_lo", "f_hi", "peak_rel", "decay", "program",
                     "dry_mix", "transcribe_opts", "midi", "verify"},
    "notes_decode": {"enabled", "color", "trim"},
    "assemble":     {"enabled", "from", "output"},
}

# Encode settings that decode must share, or the recovered image smears.
GEOMETRY = ("sr", "rows", "cols", "f_lo", "f_hi")

# Files each stage produces, {n} = source stem. The LAST entry is the
# completion marker: files are moved into place in this order, so if the
# marker exists, everything before it does too.
STAGE_OUTPUTS = {
    "encode":       ["{n}.chroma.png", "{n}.wav"],
    "decode":       ["{n}.png"],
    "notate":       ["{n}.notes.mid", "{n}.verify.json", "{n}.notes.wav"],
    "notes_decode": ["{n}.notes.png"],
}
STAGES = list(STAGE_OUTPUTS)

_print_lock = threading.Lock()


def say(*a):
    with _print_lock:
        print(*a, flush=True)


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------
def load_recipe(path):
    with open(path, "rb") as fp:
        r = tomllib.load(fp)
    for sec, val in r.items():
        if sec not in SECTIONS:
            sys.exit(f"recipe: unknown section [{sec}]. "
                     f"known: {', '.join(SECTIONS)}")
        if sec == "encode" and "lossless" in val:
            sys.exit("recipe: lossless is an archive format, not a render — "
                     "use imgaudio.py --lossless directly")
        bad = set(val) - SECTIONS[sec]
        if bad:
            sys.exit(f"recipe: unknown key(s) in [{sec}]: "
                     f"{', '.join(sorted(bad))}")
    src = r.get("source", {})
    if "frames" not in src and "audio" not in src:
        sys.exit('recipe: [source] needs frames = "path/*.jpg" '
                 'and/or audio = "path/*.m4a"')

    a = {**AUDIO_DEFAULTS, **r.get("audio", {})}
    for k in ("f_lo", "f_hi"):
        if a[k] != "auto" and not isinstance(a[k], (int, float)):
            sys.exit(f'recipe: [audio] {k} must be a number or "auto"')
    if not isinstance(a["tile"], (int, float)) or a["tile"] < 0:
        sys.exit("recipe: [audio] tile must be seconds >= 0 (0 = no tiling)")

    pal = r.get("decode", {}).get("palette")
    if pal is not None:
        if isinstance(pal, str):
            if pal not in PALETTES:
                sys.exit(f"recipe: unknown palette '{pal}'. "
                         f"try: {', '.join(PALETTES)}, or a list of hex colours")
        elif not (isinstance(pal, list) and len(pal) >= 2 and all(
                isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c)
                for c in pal)):
            sys.exit('recipe: palette must be a name or a list of at least two '
                     '"#rrggbb" colours')
    return r


def audio_cfg(r):
    return {**AUDIO_DEFAULTS, **r.get("audio", {})}


def enabled(r):
    """Which stages run. Sections you write are stages you get."""
    notate_on = "notate" in r and r["notate"].get("enabled", True)
    return {
        "encode":       True,
        "decode":       r.get("decode", {}).get("enabled", True),
        "notate":       notate_on,
        "notes_decode": notate_on and r.get("notes_decode", {}).get("enabled", True),
        "assemble":     "assemble" in r and r["assemble"].get("enabled", True),
    }


class Source:
    """One input file. Images yield one output name; recordings yield one
    per tile, known only after planning (see plan_audio)."""
    def __init__(self, path, kind):
        self.path, self.kind, self.stem = path, kind, Path(path).stem

    def __repr__(self):
        return f"Source({self.path!r}, {self.kind})"


def find_sources(r):
    spec = r["source"]
    found = {}
    for key, kind, exts in (("frames", "image", IMAGE_EXTS),
                            ("audio", "audio", AUDIO_EXTS)):
        pats = spec.get(key, [])
        pats = [pats] if isinstance(pats, str) else list(pats)
        for pat in pats:
            for f in glob.glob(pat, recursive=True):
                if Path(f).suffix.lower() in exts:
                    found[os.path.normpath(f)] = kind
    if not found:
        sys.exit(f"recipe: no sources match {spec}")
    sources = [Source(p, k) for p, k in sorted(found.items())]
    seen = {}
    for s in sources:
        if s.stem in seen:
            sys.exit(f"two sources share the name '{s.stem}' and would "
                     f"overwrite each other: {seen[s.stem]} and {s.path}")
        seen[s.stem] = s.path
    return sources


def _h(obj):
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _conv_part(r):
    a = audio_cfg(r)
    return {"sr": a["sr"], "tile": a["tile"]}


def _geo_part(r):
    a = audio_cfg(r)
    return {k: a[k] for k in ("rows", "cols", "f_lo", "f_hi")}


def stage_keys(r):
    """Per-stage settings hash, chained so upstream changes flow downstream.
    'encode' covers both image encoding and audio conversion."""
    dec = r.get("decode", {})
    k = {"encode": _h([r.get("encode", {}), _conv_part(r)])}
    k["decode"] = _h([k["encode"], _geo_part(r), dec])
    k["notate"] = _h([k["encode"], r.get("notate", {})])
    k["notes_decode"] = _h([k["notate"], _geo_part(r),
                            r.get("notes_decode", {}), dec.get("palette")])
    return k


def flags(cfg, keys):
    """Recipe keys -> CLI flags. True -> bare flag, table -> key=val,key=val."""
    out = []
    for k in sorted(set(cfg) & set(keys)):
        v = cfg[k]
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                out.append(flag)
        elif isinstance(v, dict):
            out += [flag, ",".join(f"{a}={b}" for a, b in v.items())]
        else:
            out += [flag, str(v)]
    return out


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
class ToolError(RuntimeError):
    pass


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    text = p.stdout + p.stderr
    if p.returncode != 0:
        raise ToolError(f"$ {' '.join(cmd)}\n{text.strip()}")
    return text


def _atomic_json(path, obj):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Audio planning — duration, tiles, and frequency band, once per recording
# ---------------------------------------------------------------------------
def analyze_audio(path, sr, N=4096):
    """Stream the recording through ffmpeg once, returning (duration, f_lo,
    f_hi). Memory stays flat however long the file is.

    The band is where the long-term average spectrum's peak envelope sits
    within 40 dB of its maximum, padded a quarter-octave each side. A peak
    envelope rather than an average, because averaging decibels across a
    window drowns sparse harmonic lines among the silent bins between them.
    """
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL)
    win = np.hanning(N).astype(np.float32)
    acc = np.zeros(N // 2 + 1)
    frames, total, buf = 0, 0, np.zeros(0, np.float32)
    while True:
        chunk = p.stdout.read(N * 4 * 64)
        if not chunk:
            break
        x = np.frombuffer(chunk, np.float32)
        total += len(x)
        buf = np.concatenate([buf, x])
        k = len(buf) // N
        if k:
            blk = buf[:k * N].reshape(k, N) * win
            acc += (np.abs(np.fft.rfft(blk, axis=1)) ** 2).sum(axis=0)
            frames += k
            buf = buf[k * N:]
    err = p.stderr.read().decode(errors="replace")
    p.wait()
    if p.returncode != 0:
        raise ToolError(f"ffmpeg could not read {path}\n{err.strip()}")
    if frames == 0 and len(buf):
        acc += np.abs(np.fft.rfft(np.pad(buf, (0, N - len(buf))) * win)) ** 2
        frames = 1
    duration = total / sr
    nyq = 0.97 * sr / 2
    fallback = (80.0, min(8000.0, nyq))
    if frames == 0 or acc.max() <= 1e-20:
        return duration, *fallback          # silent: nothing to measure

    f = np.fft.rfftfreq(N, 1 / sr)
    db = 10 * np.log10(acc / frames + 1e-20)
    env = np.empty_like(db)
    for i, fi in enumerate(f):
        sel = (f >= fi / 2 ** (1 / 12)) & (f <= fi * 2 ** (1 / 12))
        env[i] = db[sel].max()
    ok = (f >= 20) & (f <= nyq)
    above = f[ok][env[ok] >= env[ok].max() - 40]
    if len(above) == 0:
        return duration, *fallback
    f_lo = max(20.0, above.min() / 2 ** 0.25)
    f_hi = min(nyq, above.max() * 2 ** 0.25)
    if f_hi / f_lo < 4:                     # keep at least two octaves
        c = math.sqrt(f_lo * f_hi)
        f_lo, f_hi = max(20.0, c / 2), min(nyq, c * 2)
    return duration, round(f_lo, 1), round(f_hi, 1)


def tile_names(stem, duration, tile):
    # Half a second of grace: lossy decoders pad a few milliseconds, and a
    # 40.02s file shouldn't become three 13s tiles instead of two 20s ones.
    n = 1 if tile <= 0 else max(1, math.ceil((duration - 0.5) / tile))
    names = [stem] if n == 1 else [f"{stem}.t{k:03d}" for k in range(1, n + 1)]
    return names, duration / n


def plan_path(out, src):
    return out / f"{src.stem}.plan.json"


def plan_audio(src, out, r):
    """Duration, tile names, and detected band for one recording. Cached in
    NAME.plan.json, keyed on the settings that affect it."""
    a = audio_cfg(r)
    key = _h(_conv_part(r))
    pf = plan_path(out, src)
    if pf.exists():
        plan = json.loads(pf.read_text())
        if plan.get("key") == key:
            return plan
    duration, f_lo, f_hi = analyze_audio(src.path, a["sr"])
    names, tile_sec = tile_names(src.stem, duration, float(a["tile"]))
    plan = {"key": key, "source": src.path, "duration": round(duration, 3),
            "tile_sec": round(tile_sec, 3), "tiles": names,
            "band": [f_lo, f_hi]}
    _atomic_json(pf, plan)
    return plan


def output_names(src, out):
    """Names this source's outputs use. Recordings need their plan."""
    if src.kind == "image":
        return [src.stem]
    pf = plan_path(out, src)
    return json.loads(pf.read_text())["tiles"] if pf.exists() else []


def probe_duration(path):
    try:
        return float(run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "default=nw=1:nk=1",
                          path]).strip())
    except (ToolError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------
def check_hashes(out, keys, sources, recipe_path, *, sharded, prepare):
    """Compare stored stage hashes, delete stale outputs, record new hashes.

    Sharded workers never delete: several machines deleting on shared storage
    while others render would race. They refuse instead and ask for
    --prepare, which is run once from one machine before launching.
    """
    hf = out / ".recipe-hash"
    old = json.loads(hf.read_text()) if hf.exists() else {}
    changed = [s for s in STAGES if s in old and old[s] != keys[s]]

    if changed and sharded:
        sys.exit(f"recipe changed since the last render ({', '.join(changed)}).\n"
                 f"run once without --shard, or with --prepare, to invalidate "
                 f"stale outputs — then launch the shards.")

    if changed:
        removed = 0
        for src in sources:
            names = output_names(src, out)      # read before plans go
            for stage in changed:
                for name in names:
                    for pat in STAGE_OUTPUTS[stage]:
                        f = out / pat.format(n=name)
                        if f.exists():
                            f.unlink()
                            removed += 1
            if src.kind == "audio":
                for f in (out / f"{src.stem}.sheet.png",
                          out / f"{src.stem}.notes.sheet.png"):
                    if f.exists():
                        f.unlink()
                if "encode" in changed and plan_path(out, src).exists():
                    plan_path(out, src).unlink()   # tiling may have changed
        say(f"settings changed for: {', '.join(changed)} — "
            f"removed {removed} stale file(s)")

    if prepare:
        shutil.rmtree(out / ".tmp", ignore_errors=True)

    if changed or prepare or not hf.exists():
        _atomic_json(hf, keys)
        shutil.copyfile(recipe_path, out / "recipe.toml")   # ships with the piece


# ---------------------------------------------------------------------------
# Rendering one job (an image, or one tile of a recording)
# ---------------------------------------------------------------------------
def place(tmp, out, name, stage):
    """Move a stage's outputs into place, completion marker last."""
    for pat in STAGE_OUTPUTS[stage]:
        fn = pat.format(n=name)
        if (tmp / fn).exists():
            os.replace(tmp / fn, out / fn)


def parse_verify(text):
    def grab(pat, cast=float):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None
    return {
        "fidelity":         grab(r"spectral fidelity\s+([\d.]+)"),
        "distinct_pitches": grab(r"distinct pitches\s+(\d+) of", int),
        "notes":            grab(r"distinct pitches\s+\d+ of (\d+)", int),
        "note_density":     grab(r"note density\s+([\d.]+)"),
        "bits_per_note":    grab(r"-> ([\d.]+) bits"),
        "mean_cents":       grab(r"pitch residual\s+mean\s+([\d.]+)"),
    }


def _lut(stops):
    cols = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in stops],
                    dtype=float)
    xs, t = np.linspace(0, 1, len(cols)), np.linspace(0, 1, 256)
    return np.stack([np.interp(t, xs, cols[:, c]) for c in range(3)],
                    axis=-1).astype(np.uint8)


def apply_palette(path, palette):
    """Map a grayscale decode through a colour palette. Leaves RGB images
    (already coloured by a chroma sidecar) alone."""
    stops = PALETTES.get(palette) if isinstance(palette, str) else palette
    if not stops:
        return
    im = Image.open(path)
    if im.mode != "L":
        return
    a = np.asarray(im, dtype=float)
    lo, hi = np.percentile(a, [1, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    Image.fromarray(_lut(stops)[(a * 255).astype(np.uint8)], "RGB").save(path)


def render_one(job, ctx):
    src, name = job["src"], job["name"]
    out, r, on = ctx["out"], ctx["recipe"], ctx["on"]
    enc = r.get("encode", {})
    dec = r.get("decode", {})
    palette = dec.get("palette")
    if src.kind == "image":
        geo = {k: enc[k] for k in GEOMETRY if k in enc}
    else:
        a = audio_cfg(r)
        lo, hi = job["band"]
        geo = {"sr": a["sr"], "rows": a["rows"], "cols": a["cols"],
               "f_lo": lo if a["f_lo"] == "auto" else a["f_lo"],
               "f_hi": hi if a["f_hi"] == "auto" else a["f_hi"]}
    tmp = out / ".tmp" / f"{ctx['tag']}-{name}"
    tmp.mkdir(parents=True, exist_ok=True)
    status = []
    try:
        # --- encode (image) or convert (audio) ---------------------------
        wav = out / f"{name}.wav"
        if wav.exists():
            status.append("encode·" if src.kind == "image" else "convert·")
        elif src.kind == "image":
            run([PY, IMGAUDIO, *flags(enc, SECTIONS["encode"]),
                 "encode", src.path, str(tmp / f"{name}.wav")])
            place(tmp, out, name, "encode")
            status.append("encode✓")
        else:
            cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
            if job["tile"]:
                start, dur = job["tile"]
                cmd += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}"]
            cmd += ["-i", src.path, "-ac", "1", "-ar", str(geo["sr"]),
                    "-c:a", "pcm_s16le", str(tmp / f"{name}.wav")]
            run(cmd)
            place(tmp, out, name, "encode")
            status.append("convert✓")

        # --- decode: photo round-trip, or the sound's own spectrogram ------
        if on["decode"]:
            png = out / f"{name}.png"
            if png.exists():
                status.append("decode·")
            else:
                cmd = [PY, IMGAUDIO, *flags(geo, GEOMETRY),
                       *flags(dec, {"lens", "lens_params"})]
                if src.kind == "image" and enc.get("color"):
                    cmd.append("--color")
                run(cmd + ["decode", str(wav), str(tmp / f"{name}.png")])
                apply_palette(tmp / f"{name}.png", palette)
                place(tmp, out, name, "decode")
                status.append("decode✓")

        # --- notate ------------------------------------------------------
        nt = r.get("notate", {})
        if on["notate"]:
            notes = out / f"{name}.notes.wav"
            if notes.exists():
                status.append("notate·")
            else:
                cmd = [PY, NOTATE, "notes", str(wav),
                       str(tmp / f"{name}.notes.wav"),
                       *flags(nt, SECTIONS["notate"] - {"enabled", "midi", "verify"})]
                if nt.get("midi", False):
                    cmd += ["--midi", str(tmp / f"{name}.notes.mid")]
                verify = nt.get("verify", True)
                if verify:
                    cmd.append("--verify")
                try:
                    text = run(cmd)
                    metrics = parse_verify(text) if verify else None
                    tag = "notate✓"
                except ToolError as e:
                    if "no notes detected" not in str(e):
                        raise
                    # A dark frame or a silent tile has nothing to transcribe.
                    # Write silence of the right length so the assembled
                    # timeline keeps its shape and a resume doesn't retry it.
                    sr_s, x = wavfile.read(wav)
                    n = int(round(len(x) / sr_s * 44100))
                    wavfile.write(tmp / f"{name}.notes.wav", 44100,
                                  np.zeros(n, dtype=np.int16))
                    metrics = {"fidelity": None, "notes": 0} if verify else None
                    tag = "notate∅"
                if metrics is not None:
                    metrics = {"name": name, "source": src.path, **metrics}
                    (tmp / f"{name}.verify.json").write_text(json.dumps(metrics))
                place(tmp, out, name, "notate")
                fid = metrics.get("fidelity") if metrics else None
                status.append(f"{tag}({fid:.3f})" if fid is not None else tag)

        # --- notes_decode: picture of the performance ---------------------
        if on["notes_decode"]:
            npng = out / f"{name}.notes.png"
            if npng.exists():
                status.append("notes·")
            else:
                nd = r.get("notes_decode", {})
                dry = float(nt.get("dry_mix", 0.0))
                use_color = (src.kind == "image" and
                             nd.get("color", bool(enc.get("color")) and dry > 0))
                sr_n, x = wavfile.read(out / f"{name}.notes.wav")
                if nd.get("trim", True):
                    # notate adds a ~3s ringing tail; trim to the source's
                    # length so a dry-mixed source lines up with its columns.
                    sr_s, s = wavfile.read(wav)
                    x = x[:int(round(len(s) / sr_s * sr_n))]
                stage = tmp / "nd"          # staged as NAME.wav so the
                stage.mkdir(exist_ok=True)  # chroma sidecar name lines up
                wavfile.write(stage / f"{name}.wav", sr_n, x)
                chroma = out / f"{name}.chroma.png"
                cmd = [PY, IMGAUDIO, *flags({**geo, "sr": sr_n}, GEOMETRY)]
                if use_color and chroma.exists():
                    shutil.copyfile(chroma, stage / f"{name}.chroma.png")
                    cmd.append("--color")
                run(cmd + ["decode", str(stage / f"{name}.wav"),
                           str(tmp / f"{name}.notes.png")])
                apply_palette(tmp / f"{name}.notes.png", palette)
                place(tmp, out, name, "notes_decode")
                status.append("notes✓")

        return name, status, None
    except ToolError as e:
        return name, status, str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Sheets, report, and assembly
# ---------------------------------------------------------------------------
def build_sheets(out, sources, columns):
    """Lay a long recording's tiles out in reading order, one image per
    recording. Rebuilt each run; removed if any tile is missing."""
    columns = max(1, int(columns))
    built = 0
    for src in sources:
        if src.kind != "audio":
            continue
        names = output_names(src, out)
        if len(names) < 2:
            continue
        for suffix, sheet_name in ((".png", ".sheet.png"),
                                   (".notes.png", ".notes.sheet.png")):
            files = [out / f"{n}{suffix}" for n in names]
            sheet = out / f"{src.stem}{sheet_name}"
            if not all(f.exists() for f in files):
                if sheet.exists():
                    sheet.unlink()
                continue
            tiles = [Image.open(f).convert("RGB") for f in files]
            w, h = tiles[0].size
            gap = max(2, w // 150)
            ncol = min(columns, len(tiles))   # no empty columns on short sheets
            rows = math.ceil(len(tiles) / ncol)
            W = ncol * w + (ncol - 1) * gap
            H = rows * h + (rows - 1) * gap
            canvas = Image.new("RGB", (W, H), (24, 24, 24))
            for i, t in enumerate(tiles):
                rr, cc = divmod(i, ncol)
                canvas.paste(t.resize((w, h)), (cc * (w + gap), rr * (h + gap)))
            tmp = sheet.with_name(sheet.name + f".{os.getpid()}.tmp.png")
            canvas.save(tmp)
            os.replace(tmp, sheet)
            built += 1
    if built:
        say(f"sheets: {built} built")


def write_report(out, names):
    rows = []
    for n in names:
        f = out / f"{n}.verify.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
    if not rows:
        return
    cols = ["name", "source", "fidelity", "distinct_pitches", "notes",
            "note_density", "bits_per_note", "mean_cents"]
    tmp = out / "report.csv.tmp"
    with open(tmp, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out / "report.csv")

    scored = [x for x in rows if x.get("fidelity") is not None]
    say(f"\nreport.csv: {len(rows)} rows, {len(scored)} scored")
    if scored:
        f = np.array([x["fidelity"] for x in scored])
        lo = min(scored, key=lambda x: x["fidelity"])
        hi = max(scored, key=lambda x: x["fidelity"])
        say(f"  fidelity  mean {f.mean():.3f}  sd {f.std():.3f}  "
            f"min {lo['fidelity']:.3f} ({lo['name']})  "
            f"max {hi['fidelity']:.3f} ({hi['name']})")
    if len(rows) > len(scored):
        say(f"  {len(rows) - len(scored)} had no notes to transcribe")


def assemble(out, names, cfg, tag):
    kind = cfg.get("from", "notes")
    if kind not in ("notes", "source"):
        sys.exit('recipe: [assemble] from must be "notes" or "source"')
    suffix = ".notes.wav" if kind == "notes" else ".wav"
    files = [out / f"{n}{suffix}" for n in names]
    missing = [f for f in files if not f.exists()]
    if missing:
        say(f"assemble: {len(missing)} of {len(files)} inputs missing — skipped")
        return
    rates = {wavfile.read(f, mmap=True)[0] for f in files}
    if len(rates) > 1:
        # Image encodes and converted recordings can differ in sample rate;
        # concat's stream copy needs them identical.
        say(f"assemble: inputs have mixed sample rates {sorted(rates)} — "
            f"set [audio] sr to match [encode] sr, or assemble from notes")
        return
    target = out / cfg.get("output", "assembled.wav")
    # ffmpeg resolves list entries relative to the list file, so the list
    # lives beside the inputs and holds bare names.
    lst = out / ".assemble-list.txt"
    lst.write_text("".join(
        "file '{}'\n".format(f.name.replace("'", "'\\''")) for f in files))
    tmp = out / ".tmp" / f"assemble-{tag}{target.suffix}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "concat",
           "-safe", "0", "-i", str(lst)]
    if target.suffix.lower() == ".wav":
        cmd += ["-c", "copy"]   # same format throughout, no re-encode
    run(cmd + [str(tmp)])
    os.replace(tmp, target)
    say(f"assembled: {target}  ({len(files)} × {kind})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def make_jobs(sources, out, r, plans):
    jobs = []
    for src in sources:
        if src.kind == "image":
            jobs.append({"src": src, "name": src.stem, "tile": None,
                         "band": None})
            continue
        plan = plans[src.stem]
        tiled = len(plan["tiles"]) > 1
        for i, name in enumerate(plan["tiles"]):
            span = (i * plan["tile_sec"], plan["tile_sec"]) if tiled else None
            jobs.append({"src": src, "name": name, "tile": span,
                         "band": tuple(plan["band"])})
    return jobs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe")
    ap.add_argument("--jobs", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="jobs rendered in parallel (default: cores - 1)")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="render only this machine's slice, e.g. 3/40")
    ap.add_argument("--prepare", action="store_true",
                    help="invalidate stale outputs, analyze recordings, and "
                         "record hashes — then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would happen, run nothing")
    args = ap.parse_args()

    r = load_recipe(args.recipe)
    on = enabled(r)
    enc = r.get("encode", {})
    if enc.get("color") and enc.get("lens") in ("spectral", "phyllotaxis"):
        say(f"note: lens '{enc['lens']}' rearranges the image, so the photo's "
            f"colour won't line up with the decode — it will tint rather than "
            f"recover. Colour matches raw, edges, and fractal.\n")
    sources = find_sources(r)
    out = Path(r.get("output", {}).get("dir",
               f"renders/{Path(args.recipe).stem}"))
    out.mkdir(parents=True, exist_ok=True)
    keys = stage_keys(r)
    tag = f"{socket.gethostname()}-{os.getpid()}"
    a = audio_cfg(r)

    mine = sources
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/"))
            assert 1 <= i <= n
        except (ValueError, AssertionError):
            sys.exit("--shard wants I/N with 1 <= I <= N, e.g. 3/40")
        # Interleaved by source: each worker gets a spread across the whole
        # timeline. A recording's tiles stay together on one machine, where
        # --jobs still renders them in parallel.
        mine = sources[i - 1::n]

    n_img = sum(s.kind == "image" for s in sources)
    n_aud = len(sources) - n_img
    stages = [s for s in STAGES + ["assemble"] if on[s]]
    if n_img == 0:
        stages = ["convert" if s == "encode" else s for s in stages]
    say(f"recipe:  {args.recipe}")
    say(f"output:  {out}/")
    say(f"sources: {n_img} image(s), {n_aud} recording(s)"
        + (f" — this shard {args.shard}: {len(mine)}" if args.shard else ""))
    say(f"stages:  {' → '.join(stages)}")

    if args.dry_run:
        hf = out / ".recipe-hash"
        old = json.loads(hf.read_text()) if hf.exists() else {}
        changed = [s for s in STAGES if s in old and old[s] != keys[s]]
        if changed:
            say(f"would invalidate: {', '.join(changed)}")
        names = []
        for src in mine:
            if src.kind == "image":
                names.append(src.stem)
                continue
            got = output_names(src, out)
            if not got:
                d = probe_duration(src.path)
                got = tile_names(src.stem, d, float(a["tile"]))[0] if d else [src.stem]
                say(f"  {src.path}: {d:.1f}s -> {len(got)} tile(s)" if d
                    else f"  {src.path}: duration unknown")
            names += got
        for stage in STAGES:
            if not on[stage]:
                continue
            marker = STAGE_OUTPUTS[stage][-1]
            done = 0 if stage in changed else sum(
                (out / marker.format(n=nm)).exists() for nm in names)
            label = "convert" if (stage == "encode" and n_img == 0) else stage
            say(f"  {label:13s} {done:5d} done, {len(names) - done:5d} to render")
        return

    check_hashes(out, keys, sources, args.recipe,
                 sharded=bool(args.shard), prepare=args.prepare)

    # Analyze recordings (duration, tiles, band). --prepare does all of them
    # so fleet workers don't each re-analyze; otherwise just this slice.
    to_plan = [s for s in (sources if args.prepare else mine)
               if s.kind == "audio"]
    plans = {}
    key = _h(_conv_part(r))
    for s in list(to_plan):              # reuse cached analysis silently
        pf = plan_path(out, s)
        if pf.exists():
            p = json.loads(pf.read_text())
            if p.get("key") == key:
                plans[s.stem] = p
                to_plan.remove(s)
    if to_plan:
        say(f"analyzing {len(to_plan)} recording(s)...")
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(plan_audio, s, out, r): s for s in to_plan}
            for fut in cf.as_completed(futs):
                s = futs[fut]
                try:
                    p = fut.result()
                    plans[s.stem] = p
                    say(f"  {s.stem}: {p['duration']:.1f}s, "
                        f"{len(p['tiles'])} tile(s), band "
                        f"{p['band'][0]:.0f}-{p['band'][1]:.0f} Hz")
                except ToolError as e:
                    say(f"  {s.stem}: FAILED\n  " + str(e).replace("\n", "\n  "))
    if args.prepare:
        say("prepared — launch the shards")
        return

    runnable = [s for s in mine if s.kind == "image" or s.stem in plans]
    jobs = make_jobs(runnable, out, r, plans)
    say(f"jobs:    {len(jobs)} on {args.jobs} worker(s)\n")
    ctx = {"out": out, "recipe": r, "on": on, "tag": tag}
    plan_failures = len([s for s in mine if s.kind == "audio" and s.stem not in plans])
    failures = 0
    try:
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(render_one, j, ctx) for j in jobs]
            for k, fut in enumerate(cf.as_completed(futures), 1):
                name, status, err = fut.result()
                say(f"[{k}/{len(jobs)}] {name}  {'  '.join(status)}")
                if err:
                    failures += 1
                    say("  FAILED\n  " + err.replace("\n", "\n  "))
    except KeyboardInterrupt:
        say("\ninterrupted — rerun the same command to resume")
        sys.exit(130)

    all_names = [n for s in sources for n in output_names(s, out)]
    if on["decode"]:
        build_sheets(out, sources, a["sheet_columns"])
    if on["notate"] and r.get("notate", {}).get("verify", True):
        write_report(out, all_names)

    if on["assemble"]:
        if args.shard:
            say("\nsharded run — assemble skipped; run once without --shard "
                "when all workers finish")
        else:
            assemble(out, all_names, r["assemble"], tag)

    say(f"\ndone — {len(jobs) - failures} ok, {failures + plan_failures} failed  "
        f"(✓ rendered · skipped ∅ no notes)")
    if failures or plan_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
