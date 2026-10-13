#!/usr/bin/env python3
"""
motion.py — hear only what moves in a video.

Each frame is subtracted from the one before it, so anything still (sky,
trunk, house) cancels to nothing and only the motion is left. That motion is
turned into a picture and the picture into sound with imgaudio's encoder, so
a still scene is silent and a gust is heard as it moves through the frame.

Two ways to listen:

  profile   real time. Every video frame becomes one moment of sound; height
            in the frame becomes pitch (top = high). The sound lasts as long
            as the clip, so it can go back onto the video as its soundtrack.
            Where across the frame something moved is not kept.

  frames    full frame to frame. Every difference between two frames becomes
            a complete small picture of its own (height = pitch, width = time
            within its slot), played one after another. Frames where nothing
            moved are left out. Slower than real time, but decoding the sound
            gives the motion frames back.

  read      read the motion with a motion lens (motion_lenses/) and write
            what it found as a short text summary beside the WAV. The first
            lens, euler, reads each moving thing as turns about a pivot, after
            Euler's rotation theorem: where it pivots, how far it swings, its
            rhythm. Each thing it reads becomes a voice: a gliding tone
            (--sound glide) or notes on an instrument (--sound notes).

  python3 motion.py profile wind.mp4 wind.wav --mux wind-heard.mp4
  python3 motion.py frames  wind.mp4 wind-frames.wav --fps 5
  python3 motion.py profile wind.mp4 tree.wav --box 0.2,0,0.5,0.6 --start 3 --duration 10
  python3 motion.py read wind.mp4 turns.wav            # also writes turns.txt
  python3 motion.py read wind.mp4 both.wav --motion-lens euler,rhythm   # and where they agree
  python3 motion.py read wind.mp4 turns.wav --sound notes --voice harp --scale nid_qablim

Then make music from only the motion:
  python3 notate.py notes wind.wav wind-music.wav --voice lyre --base 60 --scale nid_qablim

Needs ffmpeg for reading video (and for --mux).
"""

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import imgaudio

SR_TRACKS = 22050      # sample rate for read mode's voices


# ---------------------------------------------------------------------------
# Reading frames
# ---------------------------------------------------------------------------
def probe(path):
    """Width, height and duration of a video, via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    s = info["streams"][0]
    return int(s["width"]), int(s["height"]), float(info["format"]["duration"])


def parse_box(text, width, height):
    """x,y,w,h in pixels, or as fractions of the frame when all are <= 1."""
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 4:
        raise SystemExit("--box needs four numbers: x,y,w,h")
    if all(v <= 1 for v in vals):
        vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
    x, y, w, h = (int(round(v)) for v in vals)
    x, y = max(0, x), max(0, y)
    w, h = min(w, width - x), min(h, height - y)
    if w < 8 or h < 8:
        raise SystemExit(f"--box {text} leaves a {w}x{h} region; it needs at least 8x8")
    return x, y, w // 2 * 2, h // 2 * 2


def read_frames(path, fps, size, box=None, start=0.0, duration=None):
    """Greyscale frames as a (n, rows, cols) float32 array in [0, 1]."""
    cols, rows = size
    filters = [f"fps={fps}"]
    if box:
        x, y, w, h = box
        filters.append(f"crop={w}:{h}:{x}:{y}")
    filters += [f"scale={cols}:{rows}:flags=area", "format=gray"]
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", path, "-vf", ",".join(filters), "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8)
    n = len(frames) // (rows * cols)
    if n < 2:
        raise SystemExit("fewer than two frames in that range; nothing can move")
    return frames[: n * rows * cols].reshape(n, rows, cols).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------
def deltas(frames, floor):
    """|frame - previous frame| with camera and compression noise removed.
    floor is in brightness units (0-1): changes smaller than it count as still."""
    d = np.abs(np.diff(frames, axis=0))
    return np.maximum(d - floor, 0.0)


def scale_to_unit(x, top_pct=99.5):
    """Divide by a high percentile of the moving parts, so one flash can't
    push everything else into the dark; calm stays quiet relative to gusts."""
    moving = x[x > 0]
    if moving.size == 0:
        return x
    return np.clip(x / max(np.percentile(moving, top_pct), 1e-6), 0.0, 1.0)


def motion_profile(d):
    """(rows, n_deltas): how much moved at each height, per frame step.
    Row 0 is the top of the frame, matching imgaudio (top = high pitch)."""
    return scale_to_unit(d.mean(axis=2).T)


def motion_frames(d, still):
    """The deltas side by side, leaving out steps where nothing moved.
    A step counts as moving when more than `still` (a fraction of the frame's
    pixels) changed beyond the noise floor. This is absolute, not relative to
    the strongest motion, so a small bird is kept even during a big gust.
    Returns the strip and the indices of the frame steps kept."""
    changed = (d > 0).mean(axis=(1, 2))
    keep = np.flatnonzero(changed > still)
    if keep.size == 0:
        return None, keep
    strip = np.concatenate([d[i] for i in keep], axis=1)
    return scale_to_unit(strip), keep


# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------
def encode_picture(grid, out_wav, col_sec, args):
    """Write a (rows, cols) grid in [0,1] as a picture and encode it."""
    png = Path(args.png) if args.png else Path(tempfile.mkstemp(suffix=".png")[1])
    Image.fromarray((grid * 255).round().astype(np.uint8), "L").save(png)
    rows, cols = grid.shape
    kw = dict(imgaudio.DEFAULTS, rows=rows, cols=cols, col_sec=col_sec,
              f_lo=args.f_lo, f_hi=args.f_hi, sr=args.sr, gamma=args.gamma,
              freq_scale=args.freq_scale, auto_prep=False, color=False,
              lens=args.lens, lens_params=imgaudio._parse_lens_params(args.lens_params))
    imgaudio.encode(str(png), out_wav, **kw)
    if not args.png:
        png.unlink()
    else:
        print(f"motion picture: {png}  ({cols}x{rows})")


# ---------------------------------------------------------------------------
# Rulers on the motion picture
# ---------------------------------------------------------------------------
def frame_skips(video, start=0.0, duration=None):
    """Times (s) where the camera skipped frames, and for how long: gaps
    longer than half a second and four times the usual frame spacing."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "frame=pts_time", "-of", "csv=p=0", video],
                         capture_output=True, text=True).stdout.split()
    t = np.array([float(x.split(",")[0]) for x in out if x.strip() and x != "N/A"])
    if len(t) < 3:
        return []
    end = start + duration if duration else np.inf
    t = t[(t >= start) & (t <= end)]
    if len(t) < 3:
        return []
    g = np.diff(t)
    big = np.where(g > max(0.5, 4 * np.median(g)))[0]
    return [(float(t[i]), float(g[i])) for i in big]


def annotate_motion(grid, out_path, *, video, mode, start, fps, keep=None, width=None,
                    box=None, skips=()):
    """A copy of the motion picture with rulers: seconds along the bottom,
    height in the frame up the side, and the camera's skipped frames marked.
    For reading only; the plain picture is what becomes sound."""
    from PIL import ImageDraw
    rows, cols = grid.shape
    pic = Image.fromarray((grid * 255).round().astype(np.uint8), "L").convert("RGB")
    Wp = int(np.clip(cols, 900, 1800))
    Hp = 300
    pic = pic.resize((Wp, Hp), Image.NEAREST if Wp >= cols else Image.BOX)
    L, R, T, B = 78, 30, 38, 92
    img = Image.new("RGB", (L + Wp + R, T + Hp + B), "white")
    img.paste(pic, (L, T))
    d = ImageDraw.Draw(img)
    small, label = imgaudio._font(13), imgaudio._font(14)
    ink, faint, red = (20, 20, 20), (110, 110, 110), (200, 30, 30)
    d.rectangle((L - 1, T - 1, L + Wp, T + Hp), outline=ink)

    # height in the frame (left)
    for frac, name in ((0, "top"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1, "bottom")):
        y = T + frac * (Hp - 1)
        d.line((L - 6, y, L - 1, y), fill=ink)
        d.text((L - 9, y), name, fill=ink, font=small, anchor="rm")
    d.text((L - 9, T - 14), "height", fill=faint, font=small, anchor="rm")

    # time (bottom)
    def x_of_col(c):
        return L + (c + 0.5) * Wp / cols
    if mode == "profile":
        # column i is the change between frames i and i+1 of the sampled run
        t0, t1 = start + 1 / fps, start + cols / fps
        step = imgaudio._nice_step(t1 - t0, 10)
        for t in np.arange(np.ceil(t0 / step) * step, t1 + 1e-9, step):
            x = x_of_col((t - start) * fps - 1)
            d.line((x, T + Hp, x, T + Hp + 6), fill=ink)
            d.text((x, T + Hp + 9), f"{round(t, 3):g}", fill=ink, font=small, anchor="mt")
        for ts, gap in skips:
            x0 = x_of_col((ts - start) * fps - 1)
            x1 = max(x0 + 2, x_of_col((ts + gap - start) * fps - 1))
            d.rectangle((x0, T + Hp + 1, x1, T + Hp + 5), fill=red)
    else:
        # one block of `width` columns per kept step; time jumps between blocks
        n = len(keep)
        every = max(1, int(np.ceil(n / 10)))
        for k in range(0, n, every):
            x = L + k * width * Wp / cols
            d.line((x, T + Hp, x, T + Hp + 6), fill=ink)
            d.text((x, T + Hp + 9), f"{start + (keep[k] + 1) / fps:.1f}", fill=ink, font=small,
                   anchor="mt")
        for ts, gap in skips:
            k = int(np.searchsorted((start + (np.asarray(keep) + 1) / fps), ts))
            x = L + min(k, n) * width * Wp / cols
            d.rectangle((x - 1, T + Hp + 1, x + 1, T + Hp + 5), fill=red)
    d.text((L + Wp + 6, T + Hp + 9), "s", fill=faint, font=small)

    # title and reading notes
    what = (f"{Path(video).name}: motion, height in the {'box' if box else 'frame'} up, time across"
            if mode == "profile" else
            f"{Path(video).name}: each block is one moment where something moved, left to right in time")
    d.text((L, 8), what, fill=ink, font=label)
    notes = ["brighter = more moved at that height and moment.  "
             "A bright stripe from top to bottom = everything moved at once (the camera, or the light)."]
    if mode == "frames":
        notes.append("Time jumps between blocks: still moments are left out. Labels are the seconds of each block.")
    if skips:
        notes.append("Red marks under the time ruler: the camera skipped frames there "
                     + ", ".join(f"{a:.1f} s ({g:.1f} s)" for a, g in skips[:6]) + ".")
    for i, line in enumerate(notes):
        d.text((L, T + Hp + 32 + 17 * i), line, fill=faint if i else ink, font=small)
    img.save(out_path)
    return out_path


def mux(video, sound, out, start, duration):
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", video, "-i", sound, "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out]
    subprocess.run(cmd, check=True)
    print(f"video with motion soundtrack: {out}")


# ---------------------------------------------------------------------------
# Read mode: motion lenses
# ---------------------------------------------------------------------------
def load_motion_lenses():
    """Every .py in motion_lenses/ (except those starting with _) that
    defines NAME and read(frames, times, params)."""
    here = Path(__file__).resolve().parent
    found = {}
    if not (here / "motion_lenses").is_dir():
        return found
    sys.path.insert(0, str(here))
    for path in sorted((here / "motion_lenses").glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"motion_lenses.{path.stem}")
            found[getattr(mod, "NAME", path.stem)] = mod
        except Exception as e:
            print(f"warning: failed to load motion lens {path.name}: {e}", file=sys.stderr)
    return found


def read_frames_timed(path, width, box=None, start=0.0, duration=None, fps=None):
    """Greyscale frames as (n, rows, cols) uint8, with each frame's real time
    in seconds. Security cameras do not record evenly, and counting frames
    instead of reading their times invents rhythms that are not there."""
    filters = [f"fps={fps}"] if fps else []
    if box:
        x, y, w, h = box
        filters.append(f"crop={w}:{h}:{x}:{y}")
    filters += [f"scale={width}:-2:flags=area", "format=gray", "showinfo"]
    cmd = ["ffmpeg", "-hide_banner", "-v", "info", "-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", path, "-vf", ",".join(filters), "-fps_mode", "passthrough",
            "-f", "rawvideo", "-"]
    r = subprocess.run(cmd, capture_output=True, check=True)
    times = [float(l.split("pts_time:")[1].split()[0])
             for l in r.stderr.decode(errors="replace").splitlines() if "pts_time:" in l]
    n = len(times)
    if n < 2:
        raise SystemExit("fewer than two frames in that range; nothing can move")
    rows = len(r.stdout) // (n * width)
    frames = np.frombuffer(r.stdout, np.uint8)[: n * rows * width].reshape(n, rows, width)
    return frames, np.asarray(times) + start


GLIDE_CENTRES = [220.0, 329.63, 146.83, 440.0, 110.0, 293.66]   # A3 E4 D3 A4 A2 D4
_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _name(hz):
    m = int(round(69 + 12 * np.log2(hz / 440.0)))
    return f"{_NOTE[m % 12]}{m // 12 - 1}"


def glide_voice(t, value, size, centre, n, t0, semis=7):
    """A tone whose pitch follows the track (plus or minus size -> plus or
    minus semis semitones) and whose loudness follows how fast it changes,
    so a still track is silent."""
    ta = t0 + np.arange(n) / SR_TRACKS
    a = np.interp(ta, t, value)
    f = centre * 2 ** (np.clip(a / size, -1.5, 1.5) * semis / 12)
    ph = 2 * np.pi * np.cumsum(f) / SR_TRACKS
    tone = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
    k = max(1, int(0.15 * SR_TRACKS))
    speed = np.convolve(np.abs(np.gradient(a)) * SR_TRACKS, np.ones(k) / k, "same")
    top = np.percentile(speed, 98)
    return tone * np.clip(speed / top if top > 0 else speed * 0, 0, 1.2) ** 0.8


def track_notes(t, value, size, root, ratios, t0, span_oct=1.5):
    """A note each time the track turns around. Its pitch comes from how far
    it had swung (snapped to the scale's ratios over the root) and its
    loudness from how big that swing was. Notes are (start s, 0.5, Hz, vel)."""
    import notate
    hop = 0.05
    tt = np.arange(t[0], t[-1], hop)
    a = np.interp(tt, t, value)
    notes = []
    for i in range(1, len(a) - 1):
        if (a[i] - a[i - 1]) * (a[i + 1] - a[i]) >= 0:
            continue                                     # still going the same way
        lo, hi = max(0, i - 40), min(len(a), i + 40)
        depth = abs(a[i] - np.median(a[lo:hi]))
        if depth < 0.15 * size:
            continue
        hz = root * 2 ** (np.clip(a[i] / size, -1.2, 1.2) * span_oct / 1.2 * 0.5 + 0.5)
        notes.append((tt[i] - t0, 0.5, notate.snap_to_ratio(hz, root, ratios),
                      int(np.clip(60 + 60 * depth / size, 50, 120))))
    return notes


def _place(box_):
    """Thirds of the picture, in words, for a box (x0, y0, x1, y1) in fractions."""
    cx, cy = (box_[0] + box_[2]) / 2, (box_[1] + box_[3]) / 2
    v = ["upper", "middle", "lower"][min(2, int(cy * 3))]
    h = ["left", "centre", "right"][min(2, int(cx * 3))]
    return (f"{v} {h} (x {box_[0]:.0%}-{box_[2]:.0%}, y {box_[1]:.0%}-{box_[3]:.0%})")


def compare_findings(named):
    """Lines saying which things more than one lens saw, and which only one.
    named: [(lens name, [finding, ...]), ...]; a finding is a dict with
    box (x0, y0, x1, y1 as fractions of the picture), rate (repeats a second
    or None) and name. Two findings are one thing when their boxes overlap or
    lie close, and their rates agree (or one is twice the other: a sway can
    brighten a patch twice per swing)."""
    def same(f, g):
        (a0, b0, a1, b1), (c0, d0, c1, d1) = f["box"], g["box"]
        overlap = min(a1, c1) > max(a0, c0) and min(b1, d1) > max(b0, d0)
        close = np.hypot((a0 + a1 - c0 - c1) / 2, (b0 + b1 - d0 - d1) / 2) < 0.15
        if not (overlap or close):
            return False
        if not f.get("rate") or not g.get("rate"):
            return True
        r = f["rate"] / g["rate"]
        return any(abs(r / k - 1) < 0.2 for k in (1, 2, 0.5))
    groups = []                                   # each: [(lens, finding), ...]
    for lens, fs in named:
        for f in fs:
            # one lens may split one thing into pieces; those join it too
            home = next((g for g in groups if any(same(f, h) for _, h in g)), None)
            (home.append((lens, f)) if home else groups.append([(lens, f)]))
    if not groups:
        return ["neither lens found a moving thing"]
    out = ["agreement between the lenses (a thing seen by both is one thing; count it once):"]
    both = [g for g in groups if len({l_ for l_, _ in g}) > 1]
    single = [g for g in groups if len({l_ for l_, _ in g}) == 1]
    for g in both + single:
        rates = [f["rate"] for _, f in g if f.get("rate")]
        rate = f"; about {min(rates):.2f} swings a second" if rates else ""
        who = ", ".join(f"{l_} {f['name']}" for l_, f in g)
        lenses_ = {l_ for l_, _ in g}
        kind = ("seen by both" if len(lenses_) > 1 else
                f"only {g[0][0]} saw" + (" (in pieces)" if len(g) > 1 else ""))
        out.append(f"  {kind}: {_place(g[0][1]['box'])}{rate} ({who})")
    if single:
        out.append("  a thing only one lens saw is likely real but less certain: euler finds "
                   "things that turn about a point, rhythm anything that repeats")
    return out


def read_mode(args, width, height, box):
    import notate
    lenses = load_motion_lenses()
    names = [x.strip() for x in args.motion_lens.split(",") if x.strip()]
    for nm in names:
        if nm not in lenses:
            raise SystemExit(f"unknown motion lens '{nm}'. available: {sorted(lenses)}")
    mods = [lenses[nm] for nm in names]
    multi = len(mods) > 1
    frames, times = read_frames_timed(args.video, args.read_width, box, args.start,
                                      args.duration, args.fps)
    gaps = np.diff(times)
    print(f"  {len(frames)} frames at {frames.shape[2]}x{frames.shape[1]}, "
          f"{times[-1] - times[0]:.1f} s; reading with {', '.join(m.NAME for m in mods)}")
    params = imgaudio._parse_lens_params(args.motion_params)
    results = [(m, m.read(frames, times, params)) for m in mods]
    tracks = []
    for m, found in results:
        for tr in found.get("tracks", []):
            tr = dict(tr, unit=getattr(m, "UNIT", ""))
            if multi:
                tr["name"] = f"{m.NAME} {tr['name']}"
            tracks.append(tr)

    # sound: every track a voice, as long as the clip
    t0, t1 = times[0], times[-1]
    n = int((t1 - t0) * SR_TRACKS)
    mix = np.zeros(n)
    voices = []
    if args.sound == "glide":
        for i, tr in enumerate(tracks):
            c = GLIDE_CENTRES[i % len(GLIDE_CENTRES)]
            mix += glide_voice(np.asarray(tr["t"]), np.asarray(tr["value"]), tr["size"], c, n, t0)
            voices.append(f"{tr['name']}: gliding tone around {_name(c)} ({c:.0f} Hz)")
    else:
        if args.scale in notate.RATIO_SCALES:
            ratios = notate.RATIO_SCALES[args.scale]
        elif args.scale in notate.SCALES:
            ratios = [2 ** (s / 12) for s in notate.SCALES[args.scale]]
        else:
            raise SystemExit(f"unknown scale '{args.scale}'; see: python3 notate.py scales")
        for i, tr in enumerate(tracks):
            root = args.root * 2 ** [0, 1, -1, 2][i % 4]
            notes = track_notes(np.asarray(tr["t"]), np.asarray(tr["value"]), tr["size"],
                                root, ratios, t0)
            if notes:
                y = notate.synthesize(notes, 60, sr=SR_TRACKS, voice=args.voice)[:n]
                mix[:len(y)] += y
            voices.append(f"{tr['name']}: {len(notes)} {args.voice} notes in {args.scale} "
                          f"from {_name(root)} ({root:.0f} Hz)")
    peak = np.abs(mix).max()
    if peak > 0:
        mix = mix / peak * 0.85
    f = min(int(0.05 * SR_TRACKS), n // 4)
    if f:
        mix[:f] *= np.linspace(0, 1, f)
        mix[-f:] *= np.linspace(1, 0, f)
    imgaudio.wavfile.write(args.output, SR_TRACKS, (mix * 32767).astype(np.int16))

    # text summary
    skips = [(times[i], gaps[i]) for i in np.where(gaps > max(0.5, 4 * np.median(gaps)))[0]]
    head = [f"{Path(args.video).name}: motion read by motion.py read --motion-lens "
            f"{','.join(m.NAME for m in mods)}",
            f"segment: {t0:.2f} to {t1:.2f} s, {len(frames)} frames read at "
            f"{frames.shape[2]}x{frames.shape[1]}"
            + (f", box {box[2]}x{box[3]} at ({box[0]},{box[1]}) of {width}x{height}" if box else
               f" (from {width}x{height})")
            + f"; frames {np.min(gaps):.2f} to {np.max(gaps):.2f} s apart"]
    head += [f"lens: {m.NAME}: {getattr(m, 'DESCRIPTION', '')}" for m in mods]
    if skips:
        head.append("the camera skipped frames at "
                    + ", ".join(f"{int(a // 60)}:{a % 60:04.1f} ({g:.1f} s)" for a, g in skips[:8]))
    body = []
    for m, found in results:
        if multi:
            body += ([""] if body else []) + [f"[{m.NAME}]"]
        body += list(found.get("lines", []))
    if multi:
        body += ["", "[both]"] + compare_findings(
            [(m.NAME, found.get("findings", [])) for m, found in results])
    snd = ([f"sound: {Path(args.output).name}, {(t1 - t0):.1f} s, one voice per track"
            + (" (pitch follows the value, plus or minus a typical swing = plus or minus "
               "7 semitones; loudness follows how fast it changes)" if args.sound == "glide" else
               " (a note each time a track turns around; pitch from how far it swung)")]
           + [f"  {v}; typical swing ±{tr['size']:.2g} {tr['unit']}".rstrip()
              for v, tr in zip(voices, tracks)])
    if not tracks:
        snd = ["sound: silent, nothing to voice"]
    txt = Path(args.output).with_suffix(".txt")
    txt.write_text("\n".join(head + [""] + body + [""] + snd) + "\n")
    print(f"wrote {args.output} and {txt}")
    if args.mux:
        mux(args.video, args.output, args.mux, args.start, args.duration)


def main(argv=None):
    class _Fmt(argparse.RawDescriptionHelpFormatter,
               argparse.ArgumentDefaultsHelpFormatter):
        pass

    p = argparse.ArgumentParser(description=__doc__, formatter_class=_Fmt)
    p.add_argument("mode", choices=["profile", "frames", "read"],
                   help="profile: real time, height = pitch; frames: every "
                        "changed frame as its own small picture; read: a motion "
                        "lens reads the motion and writes a text summary (see above)")
    p.add_argument("video", help="any video ffmpeg can read")
    p.add_argument("output", help="WAV to write")
    p.add_argument("--fps", type=float, default=None,
                   help="frames per second to sample (profile 15, frames 5)")
    p.add_argument("--box", default=None,
                   help="only this region: x,y,w,h in pixels, or fractions of "
                        "the frame (e.g. 0.2,0,0.5,0.6 for the upper middle)")
    p.add_argument("--start", type=float, default=0.0, help="seconds into the clip")
    p.add_argument("--duration", type=float, default=None, help="seconds to use")
    p.add_argument("--rows", type=int, default=200,
                   help="pitch rows (height resolution)")
    p.add_argument("--width", type=int, default=48,
                   help="frames mode: columns per frame (width resolution)")
    p.add_argument("--slot", type=float, default=1.0,
                   help="frames mode: seconds of sound per frame step; shorter is "
                        "quicker but blurs the frames on the way back")
    p.add_argument("--floor", type=float, default=0.03,
                   help="changes smaller than this (0-1 brightness) count as "
                        "still; raise it for noisy or compressed video")
    p.add_argument("--still", type=float, default=0.0,
                   help="frames mode: leave out steps where no more than this "
                        "fraction of the pixels changed (0 keeps any motion; "
                        "--floor already removes camera noise)")
    p.add_argument("--png", default=None, help="also save the motion picture here")
    p.add_argument("--annotate", action="store_true",
                   help="profile and frames modes: also write NAME.annotated.png (beside "
                        "--png, or beside the WAV) with seconds along the bottom, height in "
                        "the frame up the side and the camera's skipped frames marked. "
                        "For reading only; the sound is unchanged")
    p.add_argument("--mux", default=None,
                   help="profile and read modes: write the clip with this sound as its soundtrack")
    p.add_argument("--lens", default="raw",
                   help="imgaudio lens applied to the motion picture before "
                        "encoding (raw, edges, reveal, ...; see imgaudio.py -h)")
    p.add_argument("--lens-params", default="",
                   help="comma-separated key=value settings for the lens")
    p.add_argument("--f-lo", type=float, default=imgaudio.DEFAULTS["f_lo"],
                   help="pitch of the bottom of the frame, Hz")
    p.add_argument("--f-hi", type=float, default=imgaudio.DEFAULTS["f_hi"],
                   help="pitch of the top of the frame, Hz")
    p.add_argument("--freq-scale", choices=["log", "linear"], default="log",
                   help="how height maps to pitch; decode with the same setting")
    p.add_argument("--sr", type=int, default=imgaudio.DEFAULTS["sr"],
                   help="sample rate of the WAV, Hz")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="contrast of the motion picture; 1.0 keeps loudness "
                        "proportional to how much moved")
    lenses = load_motion_lenses()
    p.add_argument("--motion-lens", default="euler",
                   help="read mode: how to read the motion; several, comma-separated "
                        "(e.g. euler,rhythm), read the same frames and say where they agree. "
                        + "; ".join(f"{k}: {getattr(m, 'DESCRIPTION', '')}"
                                    for k, m in sorted(lenses.items())))
    p.add_argument("--motion-params", default="",
                   help="read mode: comma-separated key=value settings for the motion lens")
    p.add_argument("--sound", choices=["glide", "notes"], default="glide",
                   help="read mode: each track as a gliding tone (pitch follows it), "
                        "or as notes on --voice each time it turns around")
    p.add_argument("--voice", default="harp",
                   help="read mode, --sound notes: any notate.py instrument "
                        "(harp, lyre, piano, reed, vibraphone, vibes, pad)")
    p.add_argument("--scale", default="nid_qablim",
                   help="read mode, --sound notes: any scale from "
                        "'python3 notate.py scales' (base 60 tunings included)")
    p.add_argument("--root", type=float, default=220.0,
                   help="read mode, --sound notes: the scale's first note, Hz")
    p.add_argument("--read-width", type=int, default=320,
                   help="read mode: frame width the lens reads at, in pixels")
    args = p.parse_args(argv)

    width, height, length = probe(args.video)
    box = parse_box(args.box, width, height) if args.box else None
    bw, bh = (box[2], box[3]) if box else (width, height)
    if args.mode == "read":
        print(f"{args.video}: {width}x{height}, {length:.1f}s"
              + (f"; box {box[2]}x{box[3]} at ({box[0]},{box[1]})" if box else ""))
        read_mode(args, width, height, box)
        return
    fps = args.fps or (15.0 if args.mode == "profile" else 5.0)
    rows = args.rows
    cols = max(8, round(rows * bw / bh)) if args.mode == "profile" else args.width
    print(f"{args.video}: {width}x{height}, {length:.1f}s"
          + (f"; box {box[2]}x{box[3]} at ({box[0]},{box[1]})" if box else "")
          + f"; sampling {fps:g} frames/s")

    frames = read_frames(args.video, fps, (cols, rows), box, args.start, args.duration)
    d = deltas(frames, args.floor)
    moving = (d.mean(axis=(1, 2)) > 0).mean()
    print(f"  {len(frames)} frames, {len(d)} steps; something moved in {moving:.0%} of them")

    def ruled(grid, **kw):
        base = Path(args.png) if args.png else Path(args.output)
        out = base.with_name(base.stem + ".annotated.png")
        annotate_motion(grid, out, video=args.video, mode=args.mode, start=args.start, fps=fps,
                        box=box, skips=frame_skips(args.video, args.start, args.duration), **kw)
        print(f"annotated: {out}  (rulers for reading)")

    if args.mode == "profile":
        grid = motion_profile(d)
        encode_picture(grid, args.output, 1.0 / fps, args)
        if args.annotate:
            ruled(grid)
        if args.mux:
            mux(args.video, args.output, args.mux, args.start, args.duration)
    else:
        if args.mux:
            print("  --mux is for profile mode; frames mode runs slower than the video",
                  file=sys.stderr)
        strip, keep = motion_frames(d, args.still)
        if strip is None:
            raise SystemExit("nothing moved; try lowering --floor or --still")
        print(f"  kept {len(keep)} of {len(d)} steps (left out {len(d) - len(keep)} still ones)")
        encode_picture(strip, args.output, args.slot / args.width, args)
        if args.annotate:
            ruled(strip, keep=keep, width=args.width)


if __name__ == "__main__":
    main()
