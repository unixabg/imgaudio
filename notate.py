#!/usr/bin/env python3
"""
notate.py — turn any audio into quantized musical notes.

Tracks which pitches are present over time, snaps them to a musical scale and
a rhythmic grid, then re-synthesizes as discrete struck/plucked notes. Also
exports MIDI so the same notes can be played by any instrument in a DAW.

Two tuning systems, selected with --base:

  --base 10  (default)  Twelve-tone equal temperament. Pitch is
                        440 * 2^((n-69)/12) — the 12th root of 2, irrational,
                        terminating in no number base at all.

  --base 60             Just intonation. Every degree is a small whole-number
                        ratio to the root: a fifth is 3/2, a fourth 4/3, a
                        whole tone 9/8. These terminate exactly in
                        sexagesimal (3/2 = 1;30, 4/3 = 1;20, 9/8 = 1;07,30),
                        which is why base 60 was chosen for arithmetic, and
                        why Mesopotamian tuning was built from cycles of
                        fifths and fourths.

Base-60 tunings derived from fifths (babylonian, babylonian_pent) land within
~6 cents of equal temperament. The audible difference comes from scales using
higher partials — `harmonic` reaches ~49 cents, close to a quarter-tone.

Transcription is lossy by design: scale snapping, grid quantization, the
--voices cap, and note merging all discard information in exchange for
musicality. --verify measures how much survived.

Pipeline:
    STFT -> peak picking -> snap to scale -> snap to beat grid
         -> merge repeats into sustained notes -> synthesize + write MIDI

Subcommands:
    notes IN.wav OUT.wav        transcribe and re-synthesize
    scales                      list available scales for both bases

Examples:
    python3 notate.py notes drone.wav melody.wav
    python3 notate.py notes drone.wav melody.wav --verify
    python3 notate.py notes drone.wav melody.wav --base 60 --scale babylonian
    python3 notate.py notes drone.wav melody.wav --base 60 --scale harmonic
    python3 notate.py notes drone.wav melody.wav --bpm 60 --grid 12 --voices 1

Dependencies: numpy, scipy  (MIDI writer is built in)
"""

import argparse
import struct
import sys
from fractions import Fraction

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, find_peaks, resample


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------
# base 10: semitone offsets from the root, equal-tempered
SCALES = {
    "minor_pent":  [0, 3, 5, 7, 10],
    "major_pent":  [0, 2, 4, 7, 9],
    "major":       [0, 2, 4, 5, 7, 9, 11],
    "minor":       [0, 2, 3, 5, 7, 8, 10],
    "dorian":      [0, 2, 3, 5, 7, 9, 10],
    "phrygian":    [0, 1, 3, 5, 7, 8, 10],
    "lydian":      [0, 2, 4, 6, 7, 9, 11],
    "whole_tone":  [0, 2, 4, 6, 8, 10],
    "chromatic":   list(range(12)),
    "octatonic":   [0, 2, 3, 5, 6, 8, 9, 11],
    "hirajoshi":   [0, 2, 3, 7, 8],       # Japanese, sparse and moody
    "in_sen":      [0, 1, 5, 7, 10],      # Japanese, unsettled
}

# base 60: frequency ratios. Every degree is a small whole-number ratio, which
# terminates exactly in sexagesimal — the property base 60 was chosen for.
RATIO_SCALES = {
    # Stacked fifths and fourths, the tuning method described on the
    # Mesopotamian tuning tablets. (Later called "Pythagorean".)
    "babylonian":      [1/1, 9/8, 81/64, 4/3, 3/2, 27/16, 243/128],
    "babylonian_pent": [1/1, 9/8, 4/3, 3/2, 27/16],
    "just_major":      [1/1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8],
    "just_minor":      [1/1, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5],
    "just_pent":       [1/1, 9/8, 5/4, 3/2, 5/3],
    "just_minor_pent": [1/1, 6/5, 4/3, 3/2, 9/5],
    # Harmonic series partials 8-16 — the tuning nature actually uses.
    # Furthest from equal temperament; the 7th and 11th partials have no
    # equal-tempered equivalent at all.
    "harmonic":        [1/1, 9/8, 5/4, 11/8, 3/2, 13/8, 7/4, 15/8],
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name_to_midi(name):
    """'A2' -> 45, 'C4' -> 60, 'F#3' -> 54."""
    name = name.strip()
    for i, n in enumerate(NOTE_NAMES):
        if len(n) == 2 and name.upper().startswith(n.upper()):
            pc, rest = i, name[2:]
            break
    else:
        for i, n in enumerate(NOTE_NAMES):
            if len(n) == 1 and name.upper().startswith(n.upper()):
                pc, rest = i, name[1:]
                break
        else:
            raise ValueError(f"bad note name: {name}")
    octave = int(rest) if rest else 4
    return 12 * (octave + 1) + pc


def midi_to_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(np.maximum(f, 1e-6) / 440.0)


def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def snap_to_scale(midi_float, root_midi, scale):
    """base 10: snap a continuous MIDI pitch to the nearest scale degree."""
    rel = midi_float - root_midi
    octave = np.floor(rel / 12.0)
    within = rel - octave * 12.0
    candidates = np.array(scale + [12], dtype=float)
    idx = np.argmin(np.abs(candidates - within))
    return int(root_midi + octave * 12 + candidates[idx])


def snap_to_ratio(freq, root_hz, ratios):
    """base 60: snap a frequency to the nearest just-intonation degree.

    Works in frequency-ratio space rather than semitones, so the result is an
    exact whole-number ratio to the root (in some octave) — not an
    equal-tempered approximation of one.
    """
    if freq <= 0 or root_hz <= 0:
        return root_hz
    r = freq / root_hz
    octave = np.floor(np.log2(r))
    within = r / (2.0 ** octave)              # fold into [1, 2)
    cands = np.array(list(ratios) + [2.0])
    best = cands[np.argmin(np.abs(np.log2(cands) - np.log2(within)))]
    return float(root_hz * (2.0 ** octave) * best)


def cents_from_12tet(freq):
    """How far this frequency sits from the nearest equal-tempered note."""
    m = hz_to_midi(freq)
    return 100.0 * (m - round(m))


def parse_opts(s):
    """Parse 'key=val,key=val' into a dict. Values stay strings; callers cast.

    Extension point for future transcription options — add a new key here
    rather than a new CLI flag.
    """
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
# Verification — how much of the source actually survives transcription?
# ---------------------------------------------------------------------------
def _band_spectrogram(x, sr, n_bands=48, f_lo=80.0, f_hi=8000.0):
    """Log-band spectrogram, for comparing two signals' spectral content."""
    f, _, Z = stft(x, fs=sr, nperseg=2048, noverlap=1536,
                   window="hann", boundary=None, padded=False)
    mag = np.abs(Z)
    edges = np.geomspace(f_lo, min(f_hi, sr / 2 - 1), n_bands + 1)
    out = np.zeros((n_bands, mag.shape[1]), dtype=np.float32)
    for b in range(n_bands):
        sel = (f >= edges[b]) & (f < edges[b + 1])
        if np.any(sel):
            out[b] = mag[sel].mean(axis=0)
    return np.log1p(out * 100.0)


def spectral_fidelity(orig, resynth, sr):
    """Correlation between two signals' log-band spectrograms.

    1.0 = identical, ~0.04 = white noise, 0.0 = silence. Measures how much of
    the source's spectral structure the transcription preserved.
    """
    A = _band_spectrogram(orig, sr)
    B = _band_spectrogram(resynth, sr)
    n = min(A.shape[1], B.shape[1])
    if n < 2:
        return 0.0
    A, B = A[:, :n], B[:, :n]
    A = (A - A.mean()) / (A.std() + 1e-9)
    B = (B - B.mean()) / (B.std() + 1e-9)
    return float(np.mean(A * B))


def report_verification(orig, sr_orig, notes, synth, sr_synth, base):
    """Print what the transcription kept and what it threw away."""
    if sr_synth != sr_orig:
        synth = resample(synth, int(len(synth) * sr_orig / sr_synth))
    fid = spectral_fidelity(orig, synth, sr_orig)

    freqs = np.array([n[2] for n in notes])
    residuals = np.array([cents_from_12tet(f) for f in freqs])
    uniq = len(set(round(f, 1) for f in freqs))
    dur = len(orig) / sr_orig

    print("\n  --- verification ---")
    print(f"  spectral fidelity   {fid:.3f}   "
          f"(1.0 identical, 0.04 noise, 0.0 silence)")
    print(f"  distinct pitches    {uniq} of {len(notes)} notes")
    print(f"  note density        {len(notes)/dur:.1f} notes/sec")
    print(f"  pitch residual      mean {np.mean(np.abs(residuals)):5.1f} "
          f"cents, max {np.max(np.abs(residuals)):5.1f}")
    if base == 10:
        print("  (residual is ~0 by construction in base 10 — snapping is to "
              "the same equal-tempered grid the metric measures against)")
    print(f"  alphabet size       {uniq} symbols -> "
          f"{np.log2(max(uniq, 2)):.1f} bits per note")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def load_audio(path):
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32767.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483647.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128.0) / 128.0
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return sr, x


def transcribe(sr, audio, root_midi, scale, bpm, grid, voices,
               f_lo, f_hi, peak_rel=0.25, base=10):
    """Return a list of (start_beat, dur_beats, freq_hz, velocity).

    base=10 -> snap to 12-tone equal temperament (scale = semitone offsets)
    base=60 -> snap to just intonation          (scale = frequency ratios)
    """
    root_hz = midi_to_hz(root_midi)
    beat_sec = 60.0 / bpm
    step_sec = beat_sec * 4.0 / grid          # grid=8 -> eighth notes
    n_steps = max(1, int(len(audio) / sr / step_sec))

    nperseg = 4096                             # good freq resolution for pitch
    hop = max(256, int(step_sec * sr / 2))
    f, t, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=nperseg - hop,
                   window="hann", boundary=None, padded=False)
    mag = np.abs(Z)

    band = (f >= f_lo) & (f <= f_hi)
    f_band = f[band]
    mag = mag[band]

    print(f"  {mag.shape[1]} analysis frames -> {n_steps} grid steps "
          f"({grid}th notes at {bpm} BPM)")

    events = []                                # (step, freq, weight)
    for s in range(n_steps):
        t0, t1 = s * step_sec, (s + 1) * step_sec
        sel = (t >= t0) & (t < t1)
        if not np.any(sel):
            continue
        spec = mag[:, sel].mean(axis=1)
        if spec.max() <= 1e-8:
            continue

        peaks, props = find_peaks(spec, height=spec.max() * peak_rel)
        if len(peaks) == 0:
            continue
        order = np.argsort(props["peak_heights"])[::-1][:voices]
        for pi in order:
            k = peaks[pi]
            # parabolic interpolation for sub-bin pitch accuracy
            if 0 < k < len(spec) - 1:
                a, b, c = spec[k - 1], spec[k], spec[k + 1]
                denom = (a - 2 * b + c)
                shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
            else:
                shift = 0.0
            df = f_band[1] - f_band[0]
            freq = f_band[k] + shift * df

            if base == 60:
                fq = snap_to_ratio(freq, root_hz, scale)
            else:
                m = int(np.clip(snap_to_scale(hz_to_midi(freq),
                                              root_midi, scale), 21, 108))
                fq = midi_to_hz(m)
            if not (16.0 <= fq <= sr / 2):
                continue
            events.append((s, round(fq, 3), float(spec[k])))

    if not events:
        return []

    # Merge consecutive identical pitches into sustained notes
    weights = {}
    for s, fq, w in events:
        weights[(s, fq)] = max(weights.get((s, fq), 0.0), w)
    wmax = max(weights.values())

    active = {}                                # freq -> (start_step, weight)
    notes = []
    for s in range(n_steps + 1):
        now = {fq for (ss, fq) in weights if ss == s}
        for fq in list(active):
            if fq not in now:
                start, w = active.pop(fq)
                vel = int(np.clip(30 + 97 * (w / wmax), 1, 127))
                notes.append((start * 4.0 / grid,
                              (s - start) * 4.0 / grid, fq, vel))
        for fq in now:
            if fq not in active:
                active[fq] = (s, weights[(s, fq)])
            else:
                st, w = active[fq]
                active[fq] = (st, max(w, weights[(s, fq)]))

    notes.sort(key=lambda n: (n[0], n[2]))
    return notes


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def synthesize(notes, bpm, sr=44100, decay=2.5,
               harmonics=(1.0, 0.45, 0.22, 0.1)):
    """Render notes as struck/plucked tones with exponential decay."""
    beat_sec = 60.0 / bpm
    if not notes:
        return np.zeros(sr, dtype=np.float32)
    end_beat = max(n[0] + n[1] for n in notes)
    total = int((end_beat * beat_sec + 3.0) * sr)
    out = np.zeros(total, dtype=np.float32)

    for start_b, dur_b, freq, vel in notes:
        start = int(start_b * beat_sec * sr)
        # let notes ring past their nominal length for a natural tail
        ring = max(dur_b * beat_sec, 0.25) + 1.2
        n = int(ring * sr)
        if start + n > total:
            n = total - start
        if n <= 8:
            continue
        t = np.arange(n) / sr
        amp = (vel / 127.0) ** 1.4 * 0.25

        wave = np.zeros(n, dtype=np.float64)
        for k, h in enumerate(harmonics, start=1):
            if freq * k > sr / 2:
                break
            wave += h * np.sin(2 * np.pi * freq * k * t)

        env = np.exp(-t * decay)
        atk = min(int(0.006 * sr), n)
        env[:atk] *= np.linspace(0, 1, atk)
        out[start:start + n] += (wave * env * amp).astype(np.float32)

    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.85
    fade = min(int(0.05 * sr), len(out) // 4)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out


# ---------------------------------------------------------------------------
# Minimal MIDI writer (no external library)
# ---------------------------------------------------------------------------
def _vlq(n):
    """MIDI variable-length quantity."""
    n = int(n)
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def write_midi(path, notes, bpm, tpb=480, program=11):
    """program 11 = vibraphone; 0 = piano, 46 = harp, 89 = warm pad.

    NOTE: standard MIDI notes are equal-tempered integers, so base-60
    just-intonation pitches are rounded to the nearest semitone here. The
    .wav output carries the true tuning; the .mid is an approximation.
    """
    ev = []
    for start_b, dur_b, freq, vel in notes:
        midi = int(np.clip(round(hz_to_midi(freq)), 0, 127))
        ev.append((int(start_b * tpb), 1, 0x90, midi, vel))
        ev.append((int((start_b + max(dur_b, 0.25)) * tpb), 0, 0x80, midi, 0))
    ev.sort(key=lambda e: (e[0], e[1]))

    trk = bytearray()
    usec = int(60_000_000 / bpm)
    trk += b"\x00\xff\x51\x03" + struct.pack(">I", usec)[1:]
    trk += b"\x00\xc0" + bytes([program & 0x7F])

    last = 0
    for tick, _, status, note, vel in ev:
        trk += _vlq(tick - last) + bytes([status, note & 0x7F, vel & 0x7F])
        last = tick
    trk += b"\x00\xff\x2f\x00"

    with open(path, "wb") as fp:
        fp.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpb))
        fp.write(b"MTrk" + struct.pack(">I", len(trk)) + bytes(trk))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_notes(args):
    opts = parse_opts(args.transcribe_opts)   # reserved for future options
    table = RATIO_SCALES if args.base == 60 else SCALES
    if args.base == 60 and args.scale == "minor_pent":
        args.scale = "babylonian_pent"   # base-10 default has no base-60 twin
    if args.scale not in table:
        sys.exit(f"unknown scale '{args.scale}' for base {args.base}. "
                 f"try: {', '.join(table)}")
    scale = table[args.scale]
    root = note_name_to_midi(args.root)
    root_hz = midi_to_hz(root)

    sr, audio = load_audio(args.input)
    print(f"loaded {args.input}: {len(audio)/sr:.1f}s at {sr} Hz")
    if args.base == 60:
        print(f"  base 60 — just intonation, {args.scale} on {args.root} "
              f"({root_hz:.2f} Hz)")
        print("  degrees: " + ", ".join(
            f"{Fraction(r).limit_denominator(999)}" for r in scale))
    else:
        print(f"  base 10 — equal temperament, {args.scale} on {args.root}")
        print("  degrees: " + ", ".join(midi_to_name(root + s) for s in scale))

    notes = transcribe(sr, audio, root, scale, args.bpm, args.grid,
                       args.voices, args.f_lo, args.f_hi, args.peak_rel,
                       base=args.base)
    if not notes:
        sys.exit("no notes detected — try lowering --peak-rel or widening "
                 "--f-lo/--f-hi")

    freqs = [n[2] for n in notes]
    print(f"  {len(notes)} notes, range {min(freqs):.1f}–{max(freqs):.1f} Hz "
          f"({midi_to_name(int(round(hz_to_midi(min(freqs)))))}"
          f"–{midi_to_name(int(round(hz_to_midi(max(freqs)))))})")
    if args.base == 60:
        dev = [abs(cents_from_12tet(f)) for f in freqs]
        print(f"  mean deviation from 12-TET: {np.mean(dev):.1f} cents "
              f"(max {np.max(dev):.1f})")

    out = synthesize(notes, args.bpm, decay=args.decay)
    wavfile.write(args.output, 44100, (out * 32767).astype(np.int16))
    print(f"wrote {args.output}  ({len(out)/44100:.1f}s)")

    if args.midi:
        write_midi(args.midi, notes, args.bpm, program=args.program)
        note = " (pitches rounded to semitones)" if args.base == 60 else ""
        print(f"wrote {args.midi}  (GM program {args.program}){note}")

    if args.verify:
        report_verification(audio, sr, notes, out, 44100, args.base)

    if args.print_notes:
        print("\n  beat   dur       Hz  near   cents  vel")
        for st, d, fq, v in notes[:args.print_notes]:
            nm = midi_to_name(int(round(hz_to_midi(fq))))
            print(f"  {st:6.2f} {d:5.2f} {fq:8.2f}  {nm:>4}  "
                  f"{cents_from_12tet(fq):+6.1f}  {v:3d}")


def cmd_scales(args):
    print("base 10 (equal temperament) — semitone offsets:")
    for name, offs in SCALES.items():
        print(f"  {name:17s} {offs}")
    print("\nbase 60 (just intonation) — frequency ratios:")
    for name, rs in RATIO_SCALES.items():
        pretty = ", ".join(str(Fraction(r).limit_denominator(999)) for r in rs)
        print(f"  {name:17s} {pretty}")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pn = sub.add_parser("notes", help="transcribe audio into musical notes")
    pn.add_argument("input"); pn.add_argument("output")
    pn.add_argument("--base",   type=int, choices=[10, 60], default=10,
                    help="10 = equal temperament (default); "
                         "60 = just intonation, whole-number ratios")
    pn.add_argument("--scale",  default="minor_pent")
    pn.add_argument("--root",   default="A2")
    pn.add_argument("--bpm",    type=float, default=100.0)
    pn.add_argument("--grid",   type=int,   default=8,
                    help="rhythmic grid: 4=quarter, 8=eighth, 16=sixteenth; "
                         "3/6/12 give triplet subdivisions")
    pn.add_argument("--voices", type=int,   default=3,
                    help="max simultaneous notes per grid step")
    pn.add_argument("--f-lo",   type=float, default=80.0)
    pn.add_argument("--f-hi",   type=float, default=4000.0)
    pn.add_argument("--peak-rel", type=float, default=0.25,
                    help="peak height as fraction of the frame max (0-1); "
                         "lower = more notes")
    pn.add_argument("--decay",  type=float, default=2.5,
                    help="note decay rate; lower = longer sustain")
    pn.add_argument("--midi",   default=None, help="also write a .mid file")
    pn.add_argument("--program", type=int, default=11,
                    help="General MIDI program (11=vibraphone, 0=piano, "
                         "46=harp, 89=pad)")
    pn.add_argument("--verify", action="store_true",
                    help="re-synthesize the transcription and report how much "
                         "of the source's spectral structure survived")
    pn.add_argument("--transcribe-opts", default="",
                    help="comma-separated key=value options; extension point "
                         "for settings that don't warrant their own flag")
    pn.add_argument("--print-notes", type=int, default=0,
                    help="print the first N notes")

    sub.add_parser("scales", help="list available scales for both bases")
    return p


def main():
    args = build_parser().parse_args()
    {"notes": cmd_notes, "scales": cmd_scales}[args.cmd](args)


if __name__ == "__main__":
    main()
