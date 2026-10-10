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

  python3 motion.py profile wind.mp4 wind.wav --mux wind-heard.mp4
  python3 motion.py frames  wind.mp4 wind-frames.wav --fps 5
  python3 motion.py profile wind.mp4 tree.wav --box 0.2,0,0.5,0.6 --start 3 --duration 10

Then make music from only the motion:
  python3 notate.py notes wind.wav wind-music.wav --voice lyre --base 60 --scale nid_qablim

Needs ffmpeg for reading video (and for --mux).
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import imgaudio


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


def mux(video, sound, out, start, duration):
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", video, "-i", sound, "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out]
    subprocess.run(cmd, check=True)
    print(f"video with motion soundtrack: {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["profile", "frames"])
    p.add_argument("video")
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
    p.add_argument("--mux", default=None,
                   help="profile mode: write the clip with this sound as its soundtrack")
    p.add_argument("--lens", default="raw")
    p.add_argument("--lens-params", default="")
    p.add_argument("--f-lo", type=float, default=imgaudio.DEFAULTS["f_lo"])
    p.add_argument("--f-hi", type=float, default=imgaudio.DEFAULTS["f_hi"])
    p.add_argument("--freq-scale", choices=["log", "linear"], default="log")
    p.add_argument("--sr", type=int, default=imgaudio.DEFAULTS["sr"])
    p.add_argument("--gamma", type=float, default=1.0,
                   help="contrast of the motion picture; 1.0 keeps loudness "
                        "proportional to how much moved")
    args = p.parse_args(argv)

    width, height, length = probe(args.video)
    box = parse_box(args.box, width, height) if args.box else None
    bw, bh = (box[2], box[3]) if box else (width, height)
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

    if args.mode == "profile":
        grid = motion_profile(d)
        encode_picture(grid, args.output, 1.0 / fps, args)
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


if __name__ == "__main__":
    main()
