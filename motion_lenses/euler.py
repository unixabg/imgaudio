"""
euler.py — read motion as turns about a pivot (Euler's rotation theorem).

Leonhard Euler showed in 1775 ("Formulae generales pro translatione
quacunque corporum rigidorum") that however a rigid body is moved about a
fixed point, the result is a single turn about one axis by one angle. One
camera sees the flat version of that: a pivot in the picture and an angle.
A stalk sways about its root, a branch about where it joins the trunk, and a
shaken camera turns the whole picture at once.

How it reads a clip:
  1. Between each pair of frames, measure how every 16-pixel patch shifted
     (Lucas-Kanade, coarse to fine).
  2. Group the patches that move together and touch into movers. Pieces of
     one mover split by a dull stretch are joined again.
  3. For each mover, find the one pivot that best explains its motion as
     turning, and say how much a turn explains compared with a plain slide.
  4. Measure the turn itself on the frames about that pivot, using only the
     pixels that changed, and add the steps up into an angle over time.
  5. Report where each mover is, its pivot, how far it swings, its strongest
     rhythm, and when it was busiest. Each mover's angle becomes a track.

If most of the picture moves together for much of the clip, the camera moved;
that motion is taken away first, and the summary says the movers may be
incomplete. If the whole picture jumps at once only now and then (a bump, a
gust shaking the mount), those moments are named and left out.

Params (--motion-params):
    tile       patch size in pixels (default 16)
    min_tiles  smallest mover, in patches (default 3)
"""

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates
from scipy.signal import lombscargle

NAME = "euler"
DESCRIPTION = "turns about a pivot (Euler's rotation theorem): pivot, swing and rhythm of each mover"
UNIT = "degrees"


# ---------------------------------------------------------------------------
# Motion between two frames
# ---------------------------------------------------------------------------
def tile_flow(a, b, tile=16, levels=3, iters=5):
    """How far each tile moved from frame a to frame b (Lucas-Kanade,
    coarse to fine). Returns U, V in pixels and each tile's confidence
    (the smaller eigenvalue of its structure tensor)."""
    ty, tx = a.shape[0] // tile, a.shape[1] // tile
    a, b = a[:ty * tile, :tx * tile], b[:ty * tile, :tx * tile]
    U = np.zeros((ty, tx)); V = np.zeros((ty, tx)); conf = None
    for lev in reversed(range(levels)):
        s = 2 ** lev
        A = gaussian_filter(a, s * 0.7)[::s, ::s]
        B = gaussian_filter(b, s * 0.7)[::s, ::s]
        t = tile // s
        h, w = A.shape
        Iy, Ix = np.gradient(A)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        def tsum(z):
            return z.reshape(ty, t, tx, t).sum(axis=(1, 3))
        Sxx, Sxy, Syy = tsum(Ix * Ix), tsum(Ix * Iy), tsum(Iy * Iy)
        det = Sxx * Syy - Sxy ** 2
        tr = Sxx + Syy
        conf = (tr / 2 - np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))) / (t * t)
        ok = det > 1e-6
        for _ in range(iters):
            fu = np.kron(U / s, np.ones((t, t))); fv = np.kron(V / s, np.ones((t, t)))
            It = map_coordinates(B, [yy + fv, xx + fu], order=1, mode="nearest") - A
            bx, by = -tsum(Ix * It), -tsum(Iy * It)
            safe = np.where(ok, det, 1)
            U += np.clip(np.where(ok, (Syy * bx - Sxy * by) / safe, 0), -2, 2) * s
            V += np.clip(np.where(ok, (Sxx * by - Sxy * bx) / safe, 0), -2, 2) * s
    return U, V, conf


def turn_step(a, b, pivot, area, levels=3, iters=6):
    """The angle (radians) that turns frame a into frame b about pivot (x, y),
    from the pixels inside `area` that changed. Pixels that stayed still carry
    no weight, so the still background does not pull the answer to zero."""
    th = 0.0
    for lev in reversed(range(levels)):
        s = 2 ** lev
        A = gaussian_filter(a, 0.7 * s + 0.5)[::s, ::s]
        B = gaussian_filter(b, 0.7 * s + 0.5)[::s, ::s]
        ar = area[::s, ::s][:A.shape[0], :A.shape[1]]
        h, w = A.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = xx - pivot[0] / s, yy - pivot[1] / s
        wgt = gaussian_filter(np.abs(B - A), 1.0) * ar
        Iy, Ix = np.gradient(A)
        g = Ix * (-dy) + Iy * dx                    # change of brightness per radian
        den = (wgt * g * g).sum()
        if den <= 0:
            continue
        for _ in range(iters):
            c, sn = np.cos(th), np.sin(th)
            Bw = map_coordinates(B, [pivot[1] / s + sn * dx + c * dy,
                                     pivot[0] / s + c * dx - sn * dy],
                                 order=1, mode="nearest")
            th -= (wgt * g * (Bw - A)).sum() / den
    return th


# ---------------------------------------------------------------------------
# Pivots
# ---------------------------------------------------------------------------
def fit_pivot(r, v, extent):
    """The fixed pivot p (complex, pixels) that best explains velocities
    v[t, k] = i * theta_t * (r_k - p) for tiles at r_k. Searched on a grid,
    then refined. Returns p, theta per step, and how much of the motion a
    turn explains and how much a plain slide would."""
    E = (np.abs(v) ** 2).sum()
    A = (np.conj(r)[None, :] * v).imag.sum(axis=1)
    Bt = v.sum(axis=1)
    (x0, x1), (y0, y1) = extent
    best = None
    for fine in (False, True):
        if not fine:
            gx, gy = np.linspace(x0, x1, 81), np.linspace(y0, y1, 61)
        else:
            dx, dy = (x1 - x0) / 40, (y1 - y0) / 30
            gx = np.linspace(best.real - dx, best.real + dx, 41)
            gy = np.linspace(best.imag - dy, best.imag + dy, 41)
        P = gx[None, :] + 1j * gy[:, None]
        S = A[:, None, None] - (np.conj(P)[None] * Bt[:, None, None]).imag
        D = (np.abs(r[:, None, None] - P[None]) ** 2).sum(axis=0)
        best = P[np.unravel_index(np.argmax((S ** 2).sum(axis=0) / D), D.shape)]
    z = r - best
    theta = (np.conj(1j * z)[None, :] * v).real.sum(axis=1) / (np.abs(z) ** 2).sum()
    if E <= 0:
        return best, theta, 0.0, 0.0
    turn = 1 - (np.abs(v - theta[:, None] * 1j * z[None, :]) ** 2).sum() / E
    slide = 1 - (np.abs(v - v.mean(axis=1, keepdims=True)) ** 2).sum() / E
    return best, theta, float(turn), float(slide)


def _where(px, py, W, H):
    """A place in the picture in words."""
    off = []
    if py > 1.05 * H: off.append("below")
    if py < -0.05 * H: off.append("above")
    if px < -0.05 * W: off.append("left of")
    if px > 1.05 * W: off.append("right of")
    if off:
        return " and ".join(off) + " the picture"
    if py > 0.95 * H:
        return "at the bottom edge"
    if py < 0.05 * H:
        return "at the top edge"
    v = ["upper", "middle", "lower"][min(2, int(py / H * 3))]
    h = ["left", "centre", "right"][min(2, int(px / W * 3))]
    return "in the centre" if (v, h) == ("middle", "centre") else f"{v} {h}"


def _clock(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


# ---------------------------------------------------------------------------
# The lens
# ---------------------------------------------------------------------------
def read(frames, times, params):
    """frames: (n, rows, cols) uint8 greyscale; times: (n,) seconds.
    Returns dict(lines=[...], tracks=[...])."""
    tile = int(params.get("tile", "16"))
    min_tiles = int(params.get("min_tiles", "3"))
    fr = frames
    n = len(fr)
    T = n - 1

    def f32(i):
        return fr[i].astype(np.float32)

    flows = [tile_flow(f32(i), f32(i + 1), tile) for i in range(T)]
    U = np.stack([f[0] for f in flows]); V = np.stack([f[1] for f in flows])
    conf = np.median(np.stack([f[2] for f in flows]), axis=0)
    ty, tx = U.shape[1:]
    H, W = ty * tile, tx * tile
    tm = (times[1:] + times[:-1]) / 2
    yc, xc = (np.mgrid[0:ty, 0:tx] + 0.5) * tile
    r = (xc + 1j * yc).ravel()
    v = (U + 1j * V).reshape(T, -1)
    good = conf.ravel() > np.percentile(conf, 20) * 0.5
    lines, tracks = [], []

    # did the camera move? (most of the picture moving together)
    together = (np.abs(v[:, good]) > 0.15).mean(axis=1)
    if np.median(together) > 0.6:
        cam_p, cam_theta, _, _ = fit_pivot(r[good], v[:, good],
                                           ((-3 * W, 4 * W), (-3 * H, 4 * H)))
        common = np.median(v[:, good].real, axis=1) + 1j * np.median(v[:, good].imag, axis=1)
        v = v - common[:, None] - cam_theta[:, None] * 1j * (r - cam_p)[None, :]
        lines.append(f"camera: the whole picture moved together in {np.mean(together > 0.6):.0%} "
                     "of the steps, so the camera moved (shaken or turned). That motion is "
                     "taken away first; movers below may be incomplete.")
    else:
        # a jolt: the whole picture jumps at once, now and then. Judged step by
        # step, since a moment's jolt is outvoted by minutes of stillness.
        jolt = np.where(together > 0.6)[0]
        if len(jolt):
            shift = np.median(np.abs(v[jolt][:, good]), axis=1)
            when = ", ".join(f"{_clock(tm[i])}.{int(tm[i] * 10) % 10} ({s_:.1f} px)"
                             for i, s_ in zip(jolt[:8], shift[:8]))
            lines.append(f"camera: jolted {len(jolt)} time{'s' if len(jolt) > 1 else ''}, the whole "
                         f"picture jumping at once at {when}"
                         + (" and more" if len(jolt) > 8 else "")
                         + "; those steps are left out of everything below")
            v[jolt] = 0
        else:
            lines.append("camera: still")
    skip = set(np.where(together > 0.6)[0].tolist()) if np.median(together) <= 0.6 else set()

    act = np.sqrt((np.abs(v) ** 2).mean(axis=0))
    active = np.where(good & (act > max(3 * np.median(act[good]), 0.05)))[0]
    if len(active) == 0:
        lines.append("nothing turned or moved beyond the noise")
        return dict(lines=lines, tracks=tracks)

    # group patches that move together and touch
    X = v[:, active]
    Xn = X / np.maximum(np.linalg.norm(X, axis=0), 1e-9)
    C = np.abs(Xn.conj().T @ Xn)
    pos = np.stack([r[active].real, r[active].imag], 1) / tile
    link = (C > 0.6) & (np.linalg.norm(pos[:, None] - pos[None], axis=2) <= 2.1)
    lab = -np.ones(len(active), int)
    g = 0
    for s in range(len(active)):
        if lab[s] >= 0:
            continue
        stack, lab[s] = [s], g
        while stack:
            k = stack.pop()
            for m in np.where(link[k] & (lab < 0))[0]:
                lab[m] = g
                stack.append(m)
        g += 1

    # join pieces of one mover that a dull stretch split apart
    def series(gi):
        z = X[:, lab == gi]
        return z[:, np.argmax(np.abs(z).sum(axis=0))]
    changed = True
    while changed:
        changed = False
        ids = list(np.unique(lab))
        for i, a_ in enumerate(ids):
            for b_ in ids[i + 1:]:
                if not (lab == a_).any() or not (lab == b_).any():
                    continue
                gap = np.linalg.norm(pos[lab == a_][:, None] - pos[lab == b_][None], axis=2).min()
                if gap > 4.1:
                    continue
                sa, sb = series(a_), series(b_)
                if abs(np.vdot(sa, sb)) / max(np.linalg.norm(sa) * np.linalg.norm(sb), 1e-9) > 0.7:
                    lab[lab == b_] = a_
                    changed = True

    movers = []
    step = float(np.median(np.diff(tm))) if T > 1 else 1.0
    for gi in np.unique(lab):
        idx = active[lab == gi]
        if len(idx) < min_tiles:
            continue
        p, _, turn_f, slide_f = fit_pivot(r[idx], v[:, idx], ((-W, 2 * W), (-H, 2 * H)))
        tmask = np.zeros((ty, tx), bool)
        tmask.flat[idx] = True
        area = np.zeros(fr.shape[1:], np.float32)
        area[:H, :W] = np.kron(binary_dilation(tmask, iterations=1), np.ones((tile, tile)))
        steps = np.array([0.0 if i in skip else
                          turn_step(f32(i), f32(i + 1), (p.real, p.imag), area)
                          for i in range(T)])
        ang = np.degrees(np.cumsum(steps))
        k = max(3, int(10 / step))                  # drift slower than 10 s is not a swing
        ang = ang - np.convolve(np.pad(ang, k // 2, mode="edge"), np.ones(k) / k, "valid")[:len(ang)]
        swing = float((np.percentile(ang, 95) - np.percentile(ang, 5)) / 2)
        rate = strength = None
        if len(tm) >= 20 and ang.std() > 0:
            fq = np.linspace(0.03, 0.4 / step, 1500)
            pw = lombscargle(tm, ang - ang.mean(), 2 * np.pi * fq, normalize=True)
            rate, strength = float(fq[int(np.argmax(pw))]), float(pw.max())
        kk = max(1, k // 2)
        env = np.convolve(np.abs(ang), np.ones(kk) / kk, "same")
        busy = []
        for tt in tm[env > np.percentile(env, 90)]:
            if not busy or tt - busy[-1][-1] > 3:
                busy.append([tt])
            else:
                busy[-1].append(tt)
        movers.append(dict(idx=idx, p=p, turn=turn_f, slide=slide_f, swing=swing,
                           rate=rate, strength=strength, angle=ang,
                           busy=[float(np.mean(b)) for b in busy][:6],
                           size=float(np.sqrt((np.abs(v[:, idx]) ** 2).mean())) * len(idx)))

    movers.sort(key=lambda m: -m["size"])
    for i, m in enumerate(movers, 1):
        x0, x1 = r[m["idx"]].real.min() - tile / 2, r[m["idx"]].real.max() + tile / 2
        y0, y1 = r[m["idx"]].imag.min() - tile / 2, r[m["idx"]].imag.max() + tile / 2
        place = _where((x0 + x1) / 2, (y0 + y1) / 2, W, H)
        pv = _where(m["p"].real, m["p"].imag, W, H)
        how = ("turns" if m["turn"] > m["slide"] + 0.05 else
               "slides more than it turns" if m["slide"] > m["turn"] + 0.05 else
               "turns or slides (cannot tell apart)")
        if m["rate"] is None:
            rh = "too short to judge a rhythm"
        elif m["strength"] >= 0.3:
            rh = f"steady rhythm about {m['rate']:.2f} swings a second (one every {1 / m['rate']:.1f} s)"
        else:
            rh = (f"no steady rhythm (strongest {m['rate']:.2f} a second, one every "
                  f"{1 / m['rate']:.1f} s, but weak)")
        busy = ", ".join(_clock(b) for b in m["busy"]) or "-"
        lines.append(
            f"mover {i}: {place} (x {x0 / W:.0%}-{x1 / W:.0%}, y {y0 / H:.0%}-{y1 / H:.0%} of the "
            f"picture); {how} about a pivot {pv} (x {m['p'].real / W:.0%}, y {m['p'].imag / H:.0%}); "
            f"a turn explains {max(m['turn'], 0):.0%} of its motion, a slide {max(m['slide'], 0):.0%}; "
            f"swings about ±{m['swing']:.1f}°; {rh}; busiest around {busy}")
        tracks.append(dict(name=f"mover {i}", t=tm, value=m["angle"], size=max(m["swing"], 1e-3)))
    if not movers:
        lines.append("something moved, but nothing large enough to count as a mover")
    return dict(lines=lines, tracks=tracks)
