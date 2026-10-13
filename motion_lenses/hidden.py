"""
hidden.py — look for faint, regular changes too small to see.

The other lenses read what moves visibly. This one looks for what does not
show: a patch of the picture whose brightness, colour, position or texture
rises and falls more regularly than chance, however faintly. In a blind test
it found a flicker of half a brightness level out of 255 (about 1%) in a
windy garden scene, where the rhythm lens saw nothing.

For every 16x16 patch (and the same grid shifted by half a patch), it follows
six channels of what a generic camera records:

  brightness      how light the patch is
  blue-yellow     colour apart from brightness: blue minus brightness
  red-green       colour apart from brightness: red minus brightness
  shift-across    the patch's contents moving left or right, far below a pixel
  shift-up-down   the same, up or down
  texture         how crisp the patch is (its contrast); shimmer, haze, focus

Each is put on an even clock from the frames' real times, slow changes
(light passing, over `slowest` seconds) are taken away, and its spectrum is
taken. A rhythm counts when its peak stands far above its close neighbours
in rate: a regular thing is a narrow spike, while gusts and other irregular
change make a broad hump that raises its neighbours too. It must repeat at
least 8 times within the clip. The chance of a peak that strong turning up
by luck anywhere in the whole search (every patch, channel and rate) is
reported. A rhythm is listed as found only if that chance is under `luck`
(default 1%) and it makes up at least `share` (default 10%) of the patch's
own change. The second test matters: a camera that records unevenly copies
a sliver of any large motion onto a few fixed rates, and those slivers can
look unlikely to be luck while being a tiny part of what the patch does.

A place that shows the same rhythm in several channels, or in neighbouring
patches, is one finding. A rhythm shared by much of the picture is the
camera's or the light's. Findings in the top or bottom strip of the picture
are flagged: cameras print the time there, and a clock is regular. When the
camera is in night (infrared) mode the colour channels carry nothing and are
left out; infrared is itself light beyond human sight.

Params (--motion-params):
    luck       chance-by-luck below which a rhythm is reported (default 0.01)
    share      least share of a patch's own change the rhythm must be (default 0.1)
    slowest    slowest rhythm looked for, seconds per repeat (default 20)
    channels   which to search, e.g. channels=brightness+texture (default all)
    tile       patch size in pixels (default 16)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

NAME = "hidden"
DESCRIPTION = ("faint regular changes too small to see, in brightness, colour, tiny motion "
               "and texture, with the chance each is luck")
UNIT = "levels"
COLOUR = True                                    # wants RGB frames

CHANNELS = ["brightness", "blue-yellow", "red-green", "shift-across", "shift-up-down", "texture"]
UNITS = dict(brightness="levels", **{"blue-yellow": "levels", "red-green": "levels",
             "shift-across": "px", "shift-up-down": "px", "texture": "levels"})


# ---------------------------------------------------------------------------
# Channels: one value per patch per frame
# ---------------------------------------------------------------------------
def patch_grid(H, W, tile, shifted=True):
    """Top-left corners of the patches: the grid, and the grid moved by half
    a patch (so a small thing on a patch border is not split in two)."""
    out = []
    for off in ((0, tile // 2) if shifted else (0,)):
        for y in range(off, H - tile + 1, tile):
            for x in range(off, W - tile + 1, tile):
                out.append((y, x))
    return np.array(out)


def channels(frames, tile=16, which=None, shifted=True, block=128):
    """frames: (n, H, W, 3) uint8 RGB or (n, H, W) grey. Returns
    (dict name -> (n, P) float array, corners (P, 2), night: bool).
    Works through the frames a block at a time, so a long clip fits in
    memory; only one value per patch per frame is kept."""
    which = which or CHANNELS
    n, H, W = frames.shape[:3]
    corners = patch_grid(H, W, tile, shifted)
    P = len(corners)
    y0, x0 = corners[:, 0], corners[:, 1]

    def luma_chroma(fr):
        f = fr.astype(np.float32)
        if f.ndim == 3:
            return f, None, None
        Y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        return Y, f[..., 2] - Y, f[..., 0] - Y

    def patch_mean(img):
        S = np.zeros((img.shape[0], H + 1, W + 1), np.float64)
        S[:, 1:, 1:] = img.cumsum(1).cumsum(2)
        tot = (S[:, y0 + tile, x0 + tile] - S[:, y0, x0 + tile]
               - S[:, y0 + tile, x0] + S[:, y0, x0])
        return (tot / (tile * tile)).astype(np.float32)

    # night (infrared) mode: no colour anywhere; judged on a sample of frames
    sample = frames[:: max(1, n // 20)]
    _, U_, V_ = luma_chroma(sample)
    night = U_ is None or bool(np.abs(U_).mean() < 1.0 and np.abs(V_).mean() < 1.0)

    want_shift = "shift-across" in which or "shift-up-down" in which
    out = {k: np.zeros((n, P), np.float32) for k in which
           if not (night and k in ("blue-yellow", "red-green"))}
    u_all = np.zeros((n, P), np.float32); v_all = np.zeros((n, P), np.float32)
    prev = None                                  # last smoothed frame of the previous block
    for s0 in range(0, n, block):
        Y, U, V = luma_chroma(frames[s0:s0 + block])
        if U is None:
            U = V = np.zeros_like(Y)
        sl = slice(s0, s0 + len(Y))
        if "brightness" in out:
            out["brightness"][sl] = patch_mean(Y)
        if "blue-yellow" in out:
            out["blue-yellow"][sl] = patch_mean(U)
        if "red-green" in out:
            out["red-green"][sl] = patch_mean(V)
        if "texture" in out:
            m = patch_mean(Y)
            out["texture"][sl] = np.sqrt(np.maximum(patch_mean(Y * Y) - m * m, 0))
        if want_shift:
            # one Lucas-Kanade step per patch between neighbouring frames:
            # tiny shifts, a fraction of a pixel
            Ys = gaussian_filter(Y, (0, 1.0, 1.0))
            seq = Ys if prev is None else np.concatenate([prev[None], Ys])
            Iy, Ix = np.gradient(seq, axis=(1, 2))
            It = np.diff(seq, axis=0)
            Ix, Iy = (Ix[:-1] + Ix[1:]) / 2, (Iy[:-1] + Iy[1:]) / 2
            sxx, syy, sxy = patch_mean(Ix * Ix), patch_mean(Iy * Iy), patch_mean(Ix * Iy)
            sxt, syt = patch_mean(Ix * It), patch_mean(Iy * It)
            det = sxx * syy - sxy ** 2
            ok = det > 1e-3 * np.maximum(sxx + syy, 1e-6) ** 2
            u = np.where(ok, -(syy * sxt - sxy * syt) / np.where(ok, det, 1), 0)
            v = np.where(ok, -(sxx * syt - sxy * sxt) / np.where(ok, det, 1), 0)
            first = s0 if prev is not None else s0 + 1     # step i -> i+1 stored at i+1
            u_all[first:first + len(u)] = u
            v_all[first:first + len(v)] = v
            prev = Ys[-1]
    if "shift-across" in out:
        out["shift-across"] = np.cumsum(u_all, axis=0)
    if "shift-up-down" in out:
        out["shift-up-down"] = np.cumsum(v_all, axis=0)
    return out, corners, night


def even(series, times, slowest=20.0):
    """On an even clock at the camera's usual rate, slow changes taken away.
    Returns (clock, values (m, P))."""
    dt = float(np.median(np.diff(times)))
    clock = np.arange(times[0], times[-1] + 1e-9, dt)
    ev = np.empty((len(clock), series.shape[1]), np.float32)
    for j in range(series.shape[1]):
        ev[:, j] = np.interp(clock, times, series[:, j])
    ev -= gaussian_filter1d(ev, sigma=max(1.0, slowest / dt / 2), axis=0, mode="nearest")
    return clock, ev


def spectrum(ev, dt):
    """Periodogram of each column (Hann window): (freqs, power)."""
    w = np.hanning(len(ev))[:, None]
    F = np.fft.rfft((ev - ev.mean(axis=0)) * w, axis=0)
    return np.fft.rfftfreq(len(ev), dt), (np.abs(F) ** 2).astype(np.float64)


def local_ratio(P, f, lo, hi, near=3, far=15, floor=0.0):
    """Each bin's power over the median of its close neighbours, from `near`
    to `far` bins away on either side. A regular rhythm is a narrow spike and
    stands far above them; gusts and other irregular change make a broad
    hump, which raises its neighbours too. Under noise this ratio falls off
    as exp(-ratio * ln 2), whatever the slope of the spectrum, so its tail
    gives an honest chance of luck."""
    band = np.where((f >= lo) & (f <= hi))[0]
    R = np.zeros((len(band), P.shape[1]))
    first = int(np.searchsorted(f, floor))      # below this, slow light was taken away
    for k, i in enumerate(band):
        # the same number of neighbours on each side, or a sloping spectrum
        # makes the edge of the search look lifted
        side = min(far, i - first, len(f) - 1 - i)
        if side < near + 4:
            continue
        idx = np.r_[i - side:i - near + 1, i + near:i + side + 1]
        R[k] = P[i] / np.maximum(np.median(P[idx], axis=0), 1e-30)
    return band, R


def camera_beat(times):
    """The rate at which the camera's frame-timing pattern repeats, if it
    records unevenly in a repeating pattern (None if even). Large motion is
    copied faintly to rates mirrored about this beat."""
    g = np.diff(times)
    if len(g) < 30:
        return None
    med = np.median(g)
    g = np.where(g > 2 * med, med, g)           # skipped frames are not the pattern
    if g.std() < 0.05 * g.mean():
        return None
    t2 = np.concatenate([[0.0], np.cumsum(g)])
    jit = t2 - np.linspace(0, t2[-1], len(t2))
    f = np.fft.rfftfreq(len(jit), t2[-1] / (len(t2) - 1))
    P = np.abs(np.fft.rfft(jit - jit.mean())) ** 2
    ok = f > 0.3
    if not ok.any():
        return None
    strong = np.where(ok & (P > 20 * np.median(P[ok])))[0]
    if not len(strong):
        return None
    # the pattern's own repeat is the lowest of its strong lines (the others
    # are its multiples)
    top = strong[np.argmax(P[strong])]
    for i in strong:
        k = f[top] / f[i]
        if abs(k - round(k)) < 0.05 and P[i] > 0.1 * P[top]:
            return float(f[i])
    return float(f[top])


def _where(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    v = ["upper", "middle", "lower"][min(2, int(cy * 3))]
    h = ["left", "centre", "right"][min(2, int(cx * 3))]
    return f"{v} {h} (x {box[0]:.0%}-{box[2]:.0%}, y {box[1]:.0%}-{box[3]:.0%})"


# ---------------------------------------------------------------------------
# The lens
# ---------------------------------------------------------------------------
def read(frames, times, params):
    luck = float(params.get("luck", "0.01"))
    min_share = float(params.get("share", "0.1"))
    slowest = float(params.get("slowest", "20"))
    tile = int(params.get("tile", "16"))
    which = params.get("channels", "all")
    which = CHANNELS if which == "all" else which.split("+")
    H, W = frames.shape[1:3]
    lines, tracks, findings = [], [], []
    if times[-1] - times[0] < 10:
        return dict(lines=["clip too short to look for rhythms"], tracks=tracks, findings=findings)
    chans, corners, night = channels(frames, tile, which)
    if night:
        lines.append("camera: in night (infrared) mode, no colour; the colour channels are "
                     "left out. Infrared is itself light beyond human sight")
    dt = float(np.median(np.diff(times)))
    lo = max(1.0 / slowest, 8.0 / (times[-1] - times[0]))     # at least 8 repeats in the clip
    hi = 0.45 / dt
    cand = []                                   # (chance, channel, patch, rate, share)
    wide = []                                   # (channel, rate, lift) lifted across the picture
    per_chan = {}
    tests = 0
    for name, series in chans.items():
        clock, ev = even(series, times, slowest)
        f, Pw = spectrum(ev, dt)
        band, R = local_ratio(Pw, f, lo, hi, floor=1.0 / slowest)
        if not len(band):
            continue
        # only true peaks
        Rp = np.zeros_like(R)
        Rp[1:-1] = np.where((R[1:-1] > R[:-2]) & (R[1:-1] >= R[2:]), R[1:-1], 0)
        k = np.argmax(Rp, axis=0)
        best = Rp[k, np.arange(R.shape[1])]
        rate = f[band][k]
        tests += len(band) * R.shape[1] // 2      # Hann: neighbouring bins are not independent
        # how much of the patch's own change sits at that rate: a real rhythm
        # is a good part of it; a rate the camera's uneven timing copies over
        # from larger motion is a sliver
        total = Pw[band].sum(axis=0) + 1e-30
        i_pk = band[k]
        line = np.array([Pw[max(0, i - 2):i + 3, p].sum() for p, i in enumerate(i_pk)])
        share = line / total
        per_chan[name] = (clock, ev, f, rate, best)
        # a rate that lifts the typical patch everywhere is the camera's or the
        # light's: under chance the typical patch's ratio is about 1
        lift = np.median(R, axis=0 if False else 1)
        for i_b in np.where(lift > 2.0)[0]:
            wide.append((name, float(f[band][i_b]), float(lift[i_b])))
        for p in range(R.shape[1]):
            cand.append([best[p], name, p, float(rate[p]), float(share[p])])
    if not cand:
        lines.append("clip too short to look for rhythms")
        return dict(lines=lines, tracks=tracks, findings=findings)
    for c in cand:
        c[0] = min(1.0, tests * np.exp(-c[0] * np.log(2)))     # chance of luck, whole search
    cand.sort(key=lambda c: c[0])

    # a rhythm in much of the picture is the camera's or the light's
    shared = set()
    for name, r_, lift in wide:
        shared.add((name, round(r_, 2)))
    if wide:
        rates_ = np.array(sorted({round(r_, 3) for _, r_, _ in wide}))
        tol = max(2.5 / (times[-1] - times[0]), 0.01)
        # a comb: many rates at multiples of one base (a camera that writes a
        # full fresh picture every so many seconds pulses at all its multiples)
        comb, rest, best_n = None, rates_, 0
        for b in rates_[(rates_ >= 0.1) & (rates_ <= 0.6)]:
            on = (np.abs(rates_ / b - np.round(rates_ / b)) * b <= tol) & (rates_ >= b - tol)
            if on.sum() >= max(4, best_n + 1):
                comb, rest, best_n = b, rates_[~on], int(on.sum())
        say = []
        if comb:
            say.append(f"{comb:.2f} a second and its multiples (something every "
                       f"{1 / comb:.1f} s: likely the camera writing a full fresh picture)")
        if len(rest):
            say.append(", ".join(f"{r_:.2f}" for r_ in rest[:8])
                       + (f" and {len(rest) - 8} more" if len(rest) > 8 else "") + " a second")
        lines.append("camera or light: the whole picture is faintly lifted at "
                     + "; ".join(say) + ". That is the camera's or the light's, not one "
                     "place's; those exact rates are left out below")
    for name, (clock, ev, f, rate, best) in per_chan.items():
        strong = best > np.log(tests / 0.5) / np.log(2)          # beyond luck at 50%
        if strong.sum() >= 3:
            vals, counts = np.unique(np.round(rate[strong], 2), return_counts=True)
            df = f[1] - f[0]
            for v_ in vals[counts > 0]:
                share = (strong & (np.abs(rate - v_) <= 2 * df)).mean()
                if share > 0.3:
                    shared.add((name, round(float(v_), 2)))
                    lines.append(f"camera or light: {share:.0%} of the picture repeats its {name} "
                                 f"about {v_:.2f} times a second; that is the camera's or the "
                                 "light's, and is left out below")

    dfw = 1.0 / (times[-1] - times[0])
    found = [c for c in cand if c[0] < luck and c[4] >= min_share and not any(
        n_ == c[1] and abs(r_ - c[3]) <= 2.5 * dfw for n_, r_ in shared)]
    # join: same rate (within 5%) and touching patches, across channels
    groups = []
    for c in found:
        y, x = corners[c[2]]
        home = None
        for g in groups:
            if abs(g["rate"] / c[3] - 1) > 0.05:
                continue
            if any(abs(y - yy) <= tile and abs(x - xx) <= tile for yy, xx in g["at"]):
                home = g
                break
        if home is None:
            home = dict(rate=c[3], at=[], chans={}, chance=c[0], best=c, share=0.0)
            groups.append(home)
        home["at"].append((y, x))
        home["chans"][c[1]] = min(home["chans"].get(c[1], 1.0), c[0])
        home["chance"] = min(home["chance"], c[0])
        home["share"] = max(home["share"], c[4])

    lines.append(f"searched {len(corners)} patches x {len(chans)} channels "
                 f"({', '.join(chans)}), rhythms from {lo:.2f} to {hi:.2f} a second; "
                 f"listed below only if the chance of luck anywhere in the search is under {luck:.0%} "
                 f"and the rhythm is at least {min_share:.0%} of that patch's own change")
    if not groups:
        c = min(cand, key=lambda c: (c[0] >= luck, -c[4] if c[0] < luck else c[0]))
        y, x = corners[c[2]]
        box = (x / W, y / H, (x + tile) / W, (y + tile) / H)
        lines.append(f"nothing more regular than chance. The most regular: {c[1]} at "
                     f"{_where(box)}, {c[3]:.2f} a second, {c[4]:.0%} of that patch's own change, "
                     f"chance of luck {c[0]:.0e}")
        return dict(lines=lines, tracks=tracks, findings=findings)

    # --- sort what was found -------------------------------------------------
    span_t = times[-1] - times[0]
    df_ = 1.0 / span_t
    beat = camera_beat(times)
    for g in groups:
        ys = [a for a, _ in g["at"]]; xs = [b for _, b in g["at"]]
        g["box"] = (min(xs) / W, min(ys) / H, (max(xs) + tile) / W, (max(ys) + tile) / H)
        g["c"] = ((g["box"][0] + g["box"][2]) / 2, (g["box"][1] + g["box"][3]) / 2)
        g["echo"], g["copy"], g["piece"] = [], [], 0
    groups.sort(key=lambda g: (g["chance"], -g["share"]))
    kept = []
    for g in groups:
        home, why = None, None
        for h in kept:
            near = np.hypot(g["c"][0] - h["c"][0], g["c"][1] - h["c"][1]) < 0.25
            tol = max(0.03 * g["rate"], 2 * df_)
            if near and abs(g["rate"] - h["rate"]) <= tol:
                home, why = h, "piece"
            elif near and any(abs(g["rate"] - k * h["rate"]) <= max(0.03 * k * h["rate"], 2 * df_)
                              for k in (2, 3, 4, 5)):
                home, why = h, "echo"
            elif near and beat and any(abs(g["rate"] - abs(k * beat - h["rate"])) <= 2 * df_
                                       for k in (1, 2, 3)):
                # the camera's uneven timing copies a rhythm to its mirror about
                # the frame pattern, in the same place and much weaker. If the
                # two are about as strong, neither is clearly the original:
                # both come from the camera.
                home, why = h, "copy"
                if g["share"] > h["share"] / 3:
                    h["camera_tied"] = True
            if home:
                break
        if home is None:
            kept.append(g)
        elif why == "piece":
            home["at"] += g["at"]; home["piece"] += 1
            home["box"] = (min(home["box"][0], g["box"][0]), min(home["box"][1], g["box"][1]),
                           max(home["box"][2], g["box"][2]), max(home["box"][3], g["box"][3]))
            for k_, v_ in g["chans"].items():
                home["chans"][k_] = min(home["chans"].get(k_, 1.0), v_)
        else:
            home[why].append(g["rate"])
    # how big the rhythm itself is, in each channel it was seen in: plainly
    # visible (a brightness swing of 1.5 levels or more, or a shift of 0.2 px
    # or more) or hidden; and too small to mean anything (below what 8-bit
    # video can carry, averaged over a patch)
    def wave(name, idx, rate):
        clock, ev = per_chan[name][0], per_chan[name][1]
        w_ = 2 * np.pi * rate
        X = np.stack([np.sin(w_ * clock), np.cos(w_ * clock)], 1)
        cs = np.linalg.lstsq(X, ev[:, idx] - ev[:, idx].mean(axis=0), rcond=None)[0]
        return float(np.hypot(cs[0], cs[1]).max())
    VISIBLE = dict(brightness=1.5, **{"blue-yellow": 1.5, "red-green": 1.5, "texture": 1.5,
                   "shift-across": 0.2, "shift-up-down": 0.2})
    FLOOR = dict(brightness=0.03, **{"blue-yellow": 0.03, "red-green": 0.03, "texture": 0.03,
                 "shift-across": 0.003, "shift-up-down": 0.003})
    for g in kept:
        idx = [int(np.where((corners[:, 0] == y) & (corners[:, 1] == x))[0][0]) for y, x in g["at"]]
        g["size"] = {n_: wave(n_, idx, g["rate"]) for n_ in g["chans"]}
        g["visible"] = any(v_ >= VISIBLE[n_] for n_, v_ in g["size"].items())
        g["tiny"] = all(v_ < FLOOR[n_] for n_, v_ in g["size"].items())
    kept = [g for g in kept if not g["tiny"]]
    tied_ = [g for g in kept if g.get("camera_tied")]
    hidden_ = [g for g in kept if g not in tied_ and not g["visible"]]
    seen_ = [g for g in kept if g not in hidden_ and g not in tied_]
    if beat:
        lines.append(f"camera: records frames unevenly, in a pattern that repeats {beat:.2f} times a "
                     "second; two rhythms mirrored about it come from that pattern working on "
                     "the picture, and are listed last as probably the camera's")
    lines.append(f"{len(hidden_)} hidden (regular but too small to see: under 1.5 brightness "
                 f"levels or 0.2 pixel), {len(seen_)} large enough to see, {len(tied_)} probably "
                 "the camera's (mirrored about its frame pattern)")

    for i, g in enumerate(hidden_ + seen_ + tied_, 1):
        box = g["box"]
        name, p = g["best"][1], g["best"][2]
        clock, ev, f, rate, best = per_chan[name]
        # its size: the fitted wave at that rate, in the channel's own unit
        w_ = 2 * np.pi * g["rate"]
        X = np.stack([np.sin(w_ * clock), np.cos(w_ * clock)], 1)
        cs = np.linalg.lstsq(X, ev[:, p] - ev[:, p].mean(), rcond=None)[0]
        amp = float(np.hypot(*cs))
        chans_txt = ", ".join(f"{k} ({v:.0e})" for k, v in sorted(g["chans"].items(), key=lambda kv: kv[1]))
        edge = (" — at the edge of the picture, where cameras often print the time; "
                "a clock is regular" if box[1] < 0.08 or box[3] > 0.92 else "")
        kind = ("probably the camera: it and another rhythm mirror each other about the "
                f"camera's {beat:.2f}-a-second frame pattern" if g in tied_ else
                "hidden: too small to see" if g in hidden_ else
                "large enough to see: a regular sway, flutter or flicker")
        extra = ""
        if g["echo"]:
            extra += f"; also at {', '.join(f'{r:.2f}' for r in sorted(set(g['echo'])))} (multiples of it)"
        if g["copy"]:
            extra += (f"; its mirror about the camera's pattern at "
                      f"{', '.join(f'{r:.2f}' for r in sorted(set(g['copy'])))} folded in")
        lines.append(f"found {i} ({kind}): {_where(box)}, {len(set(g['at']))} patch"
                     f"{'es' if len(set(g['at'])) > 1 else ''}; repeats {g['rate']:.2f} times a "
                     f"second (every {1 / g['rate']:.2f} s); by about ±{amp:.2g} "
                     f"{UNITS.get(name, '')} in {name}, {g['share']:.0%} of that patch's own change; "
                     f"chance of luck {g['chance']:.0e}; "
                     f"seen in {chans_txt}{extra}{edge}")
        findings.append(dict(name=f"found {i}", box=box, rate=g["rate"]))
        if len(tracks) < 4:
            bp = ev[:, p] - ev[:, p].mean()
            Fq = np.fft.rfft(bp)
            fq = np.fft.rfftfreq(len(bp), dt)
            bp = np.fft.irfft(Fq * np.exp(-0.5 * ((fq - g["rate"]) / (0.15 * g["rate"])) ** 2), n=len(bp))
            tracks.append(dict(name=f"found {i}", t=clock, value=bp,
                               size=float(np.percentile(np.abs(bp), 95)) or 1.0))
    return dict(lines=lines, tracks=tracks, findings=findings)
