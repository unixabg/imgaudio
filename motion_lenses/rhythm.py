"""
rhythm.py — a rhythm map: which parts of the scene move in a repeating way,
and how often.

The motion version of the spectral lens. That lens finds patterns that repeat
across space; this one finds motion that repeats over time, region by region:
leaves that flutter fast, branches that sway slowly, a wing beating, a fan
turning.

How it reads a clip:
  1. Cut the picture into 16-pixel tiles, each split in four, and follow the
     brightness of each quarter from frame to frame. A small sway shows up as
     a small rise and fall in brightness.
  2. Put every series on an even clock using the frames' real times (cameras
     do not record evenly), and remove changes slower than 20 seconds, such
     as passing light.
  3. Find each tile's clearest repeat: the peak that stands highest above
     its neighbours from half to twice its rate. (Judged against the whole
     spectrum instead, the slow wandering in every real camera picture would
     pass for a rhythm.)
  4. Join neighbouring tiles that repeat at the same rate into regions.
  5. A rate found clearly across most of the picture, still ground included,
     is not things moving: it is the camera (exposure, frame timing,
     compression, flicker) or the light changing. It is named as such and
     kept out of the regions. A small swaying thing also brightens a few of
     its tiles at twice or three times its rate; those echoes are folded
     back into its region.

Each of the four strongest regions becomes a track, so it can be heard.

Params (--motion-params):
    tile     tile size in pixels (default 16)
    clear    how far a tile's peak must stand above the median of its
             neighbours (half to twice its rate) to count as a rhythm (default 8)
    longest  slowest rhythm looked for, in seconds per repeat (default 20)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

NAME = "rhythm"
DESCRIPTION = "rhythm map: which parts of the scene repeat their motion, and how often"
UNIT = "brightness"


def _where(y, x, rows, cols):
    v = ["upper", "middle", "lower"][min(2, int(y / rows * 3))]
    h = ["left", "centre", "right"][min(2, int(x / cols * 3))]
    return "centre" if (v, h) == ("middle", "centre") else f"{v} {h}"


def read(frames, times, params):
    tile = int(params.get("tile", "16"))
    clear = float(params.get("clear", "8"))
    longest = float(params.get("longest", "20"))
    n, H, W = frames.shape
    ty, tx = H // tile, W // tile
    lines, tracks = [], []
    span = times[-1] - times[0]
    if n < 16 or span < 4:
        return dict(lines=["clip too short to judge rhythms"], tracks=tracks)

    # brightness of each quarter tile, per frame: (n, ty, tx, 4)
    h2 = tile // 2
    q = frames[:, :ty * tile, :tx * tile].astype(np.float32)
    q = q.reshape(n, ty, 2, h2, tx, 2, h2).mean(axis=(3, 6))       # (n, ty, 2, tx, 2)
    q = q.transpose(0, 1, 3, 2, 4).reshape(n, ty, tx, 4)

    # an even clock at the camera's usual rate
    dt = float(np.median(np.diff(times)))
    fs = 1.0 / dt
    clock = np.arange(times[0], times[-1], dt)
    flat = q.reshape(n, -1)
    even = np.empty((len(clock), flat.shape[1]), np.float32)
    for j in range(flat.shape[1]):
        even[:, j] = np.interp(clock, times, flat[:, j])
    # take away changes slower than `longest` (passing light). A smooth
    # (Gaussian) average is used: a plain running average leaves ripples just
    # above its cut-off, which look like a slow rhythm across the whole picture
    slow = gaussian_filter1d(even, sigma=max(1.0, longest * fs / 2), axis=0, mode="nearest")
    even = even - slow

    nper = min(len(clock), max(64, int(4 * longest * fs)))
    f, P = welch(even, fs=fs, nperseg=nper, axis=0, detrend="linear")
    P = P.reshape(len(f), ty, tx, 4).sum(axis=3)                  # power per tile
    band = (f >= 1.0 / longest) & (f <= 0.45 * fs)
    if band.sum() < 4:
        return dict(lines=["clip too short to judge rhythms"], tracks=tracks)
    fb, Pb = f[band], P[band]
    # a rhythm is a true peak: higher than its neighbours, not the band's edge
    # (slow drift always makes the lowest frequencies the strongest)
    # Each peak is judged against its neighbours in pitch, from half to twice
    # its rate (leaving out the peak itself). Judged against the whole
    # spectrum, the slow wandering every real camera picture has would make
    # the slowest bins look like clear rhythms.
    nb = len(fb)
    lo_i = np.clip(np.searchsorted(fb, fb / 2), 0, nb - 1)
    hi_i = np.clip(np.searchsorted(fb, fb * 2), 0, nb - 1)
    near_med = np.empty_like(Pb)
    for i in range(nb):
        idx = np.r_[lo_i[i]:max(lo_i[i], i - 2), min(hi_i[i], i + 3):hi_i[i] + 1]
        if len(idx) < 3:                       # too close to the band's edge to judge
            near_med[i] = np.inf
        else:
            near_med[i] = np.median(Pb[idx], axis=0)
    local = Pb / np.maximum(near_med, 1e-12)
    inner = np.zeros_like(Pb)
    inner[1:-1] = np.where((Pb[1:-1] > Pb[:-2]) & (Pb[1:-1] > Pb[2:]), local[1:-1], 0)
    peak_i = np.argmax(inner, axis=0)                               # (ty, tx)
    clarity = np.take_along_axis(local, peak_i[None], axis=0)[0]
    rate = fb[peak_i]
    motion = np.sqrt(Pb.sum(axis=0))                                # how much it moves
    # brightness levels (0-255) a tile's quarters swing by, on average
    amp = even.reshape(len(clock), ty, tx, 4).std(axis=0).mean(axis=2)
    alive = amp > 0.2                                               # beyond rounding and noise
    moving = alive & (motion > max(3 * np.median(motion), 1e-6))

    # a rate clear across most of the picture, still ground included, is the camera
    df = fb[1] - fb[0]
    camera_rates = []
    clear_any = (clarity > clear) & alive
    shared = (clarity > 2 * clear) & alive                          # the camera's own pulse is sharp
    for r0 in np.unique(rate[shared]):
        share = (shared & (np.abs(rate - r0) <= 1.5 * df)).mean()
        if share > 0.4:
            camera_rates.append((float(r0), float(share)))
    if camera_rates:
        lo, hi = min(r for r, _ in camera_rates), max(r for r, _ in camera_rates)
        share = max(sh for _, sh in camera_rates)
        when = (f"about {lo:.2f} times a second (every {1 / lo:.1f} s)" if hi - lo < 2 * df else
                f"between {lo:.2f} and {hi:.2f} times a second (every {1 / hi:.1f} to {1 / lo:.1f} s)")
        lines.append(f"camera or light: up to {share:.0%} of the picture, still parts included, "
                     f"rises and falls together {when}. That comes from the camera (exposure, "
                     "frame timing, compression, flicker) or from the light changing, not from "
                     "things moving; it is left out below")
    if not camera_rates:
        lines.append("camera or light: no rhythm shared by the whole picture")
    # moments when most of the picture changed at once (a bump, a flash)
    step_change = np.abs(np.diff(frames[:, :ty * tile, :tx * tile].astype(np.float32), axis=0))
    tile_change = step_change.reshape(n - 1, ty, tile, tx, tile).mean(axis=(2, 4))
    base = np.median(tile_change, axis=0)
    # a typical tile changing 20 times its usual amount: a bump measured 39;
    # the camera's own refresh (a full keyframe every few seconds) reaches 8
    usual = np.median(tile_change / np.maximum(base, 0.5), axis=(1, 2))
    wide = usual > 20
    tmid = (times[1:] + times[:-1]) / 2
    if wide.any() and wide.mean() < 0.2:
        moments = ", ".join(f"{int(x // 60)}:{x % 60:04.1f}" for x in tmid[wide][:8])
        lines.append(f"most of the picture changed at once at {moments}"
                     + (" and more" if wide.sum() > 8 else "")
                     + " (a bump to the camera, or a flash of light)")
    lively = (amp > 1.0).mean()                     # a whole brightness level or more
    if lively > 0.6:
        lines.append(f"note: {lively:.0%} of the picture changes, so the camera itself "
                     "may have moved or the light changed; a shared rhythm above may be that")

    def is_camera(r):
        return any(abs(r - c) <= 1.5 * df for c, _ in camera_rates)

    good = clear_any & moving & ~np.vectorize(is_camera)(rate)
    # join neighbouring tiles that repeat at the same rate
    lab = -np.ones((ty, tx), int)
    regions = []
    for y in range(ty):
        for x in range(tx):
            if not good[y, x] or lab[y, x] >= 0:
                continue
            gi = len(regions)
            stack, lab[y, x], cells = [(y, x)], gi, []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny in range(cy - 1, cy + 2):
                    for nx in range(cx - 1, cx + 2):
                        if (0 <= ny < ty and 0 <= nx < tx and good[ny, nx] and lab[ny, nx] < 0
                                and abs(rate[ny, nx] - rate[y, x]) <= max(1.5 * df, 0.1 * rate[y, x])):
                            lab[ny, nx] = gi
                            stack.append((ny, nx))
            regions.append(cells)

    found = []
    for gi, cells in enumerate(regions):
        ys, xs = np.array(cells).T
        w = clarity[ys, xs] * motion[ys, xs]
        r = float(np.average(rate[ys, xs], weights=w))
        found.append(dict(cells=cells, rate=r, clarity=float(np.median(clarity[ys, xs])),
                          size=len(cells), weight=float(w.sum()), lab=gi))
    # a swaying thing brightens its tiles unevenly, so a few of them repeat at
    # twice or three times its rate; fold those small echoes into the region
    found.sort(key=lambda g: -g["weight"])
    kept = []
    for g in found:
        home = None
        for p in kept:
            ratio = g["rate"] / p["rate"]
            near = min(abs(cy - py) + abs(cx - px) for cy, cx in g["cells"]
                       for py, px in p["cells"]) <= 3
            if near and g["size"] < p["size"] and round(ratio) in (2, 3) \
                    and abs(ratio - round(ratio)) < 0.06 * round(ratio):
                home = p
                break
        if home:
            home["cells"] = home["cells"] + g["cells"]
            home["echoes"] = home.get("echoes", 0) + g["size"]
        else:
            kept.append(g)
    found = [g for g in kept if g["size"] >= 2 or g["clarity"] > 3 * clear]
    found.sort(key=lambda g: -g["weight"])
    found = found[:9]

    letters = "ABCDEFGHI"
    grid = [["." for _ in range(tx)] for _ in range(ty)]
    for L, g in zip(letters, found):
        for cy, cx in g["cells"]:
            grid[cy][cx] = L
    if not found:
        lines.append("no part of the scene repeats its motion clearly")
        return dict(lines=lines, tracks=tracks)

    lines.append(f"rhythm map ({tx} x {ty} tiles, top of the picture first; "
                 ". = no clear repeat, a letter = a region below):")
    lines += ["  " + " ".join(row) for row in grid]
    for L, g in zip(letters, found):
        ys, xs = np.array(g["cells"]).T
        place = _where((ys.mean() + 0.5) * tile, (xs.mean() + 0.5) * tile, H, W)
        x0, x1 = xs.min() * tile / W, (xs.max() + 1) * tile / W
        y0, y1 = ys.min() * tile / H, (ys.max() + 1) * tile / H
        lines.append(f"{L}: {place} (x {x0:.0%}-{x1:.0%}, y {y0:.0%}-{y1:.0%}), {g['size']} tiles; "
                     f"repeats about {g['rate']:.2f} times a second (every {1 / g['rate']:.1f} s); "
                     f"clarity {g['clarity']:.0f} (8 is just clear)"
                     + (f"; {g['echoes']} of its tiles repeat at a multiple of that rate "
                        "(the same motion, seen unevenly)" if g.get("echoes") else ""))
        # the region's motion as a track: its tiles' brightness, on the even clock
        cols = [(cy * tx + cx) * 4 + qq for cy, cx in g["cells"] for qq in range(4)]
        sig = even[:, cols]
        sig = sig - sig.mean(axis=0)
        # line the quarters up so they add rather than cancel
        ref = sig[:, np.argmax(sig.std(axis=0))]
        sig = sig * np.sign((sig * ref[:, None]).sum(axis=0) + 1e-12)
        val = sig.mean(axis=1)
        if len(tracks) < 4:                       # more voices than four is a crowd
            tracks.append(dict(name=f"region {L}", t=clock, value=val,
                               size=float(np.percentile(np.abs(val), 95)) or 1.0))
    return dict(lines=lines, tracks=tracks)
