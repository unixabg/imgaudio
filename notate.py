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

Instruments (--voice): vibes (the original struck tone), piano, harp,
vibraphone, pad, and two ancient Mesopotamian instruments, lyre (sammûm)
and reed (embūbum, a reed pipe). Piano, harp, vibraphone, lyre and reed are
modelled on how the real instrument makes its sound (plucked strings, felt
hammers, a blown reed); they are synthesized, not recorded.

Tunings (--base 60 --scale ...): besides the just scales, the seven
Babylonian tunings from the cuneiform texts: isartum, kitmum, embubum,
pitum, nid_qablim (the tuning named on the Hurrian hymn, same notes as
"babylonian"), nis_gabari, qablitum. `scales` lists them all.

Examples:
    python3 notate.py notes drone.wav melody.wav
    python3 notate.py notes drone.wav melody.wav --verify
    python3 notate.py notes drone.wav melody.wav --base 60 --scale babylonian
    python3 notate.py notes drone.wav melody.wav --base 60 --scale harmonic
    python3 notate.py notes drone.wav melody.wav --bpm 60 --grid 12 --voices 1
    python3 notate.py notes wind.wav wind-music.wav --voice lyre --base 60 --scale nid_qablim
    python3 notate.py notes drone.wav melody.wav --voice reed --midi melody.mid

Timing: --bpm and --grid set the size of the time step the audio is cut into
(step = 60/bpm * 4/grid seconds). Notes are placed at the real time they
occurred, so the music stays as long as the source and in step with any
video it came from; a slower tempo gives fewer, longer notes, not a slower
piece. --bpm 60 --grid 32 and --bpm 120 --grid 16 cut the same steps.

Dependencies: numpy, scipy  (MIDI writer is built in)
"""

import argparse
import struct
import sys
from fractions import Fraction

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, find_peaks, resample, iirpeak, lfilter


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

# The seven Old Babylonian lyre tunings. The tuning tablets name seven
# tunings and give a cycle for moving between them by retuning one string
# at a time; each tuning is a chain of fifths and fourths. Notes on strings
# 1-7 follow Kilmer's reading (strings rising), as tabled by Crickmore,
# "New light on the Babylonian tonal system" (2008). Each step of the cycle
# raises one note a semitone (F->F#, C->C#, G->G#, ...), matching the texts'
# one-string retuning. Today's mode names are given for orientation.
#
# West (1994) reads the strings as falling. That gives the same seven
# patterns under different names: on his reading isartum is the major
# pattern rather than the Phrygian one. The names here follow Kilmer.
BABYLONIAN_TUNINGS = {
    "isartum":    "E F G A B C D",           # Phrygian
    "kitmum":     "E F# G A B C D",          # Aeolian (natural minor)
    "embubum":    "E F# G A B C# D",         # Dorian
    "pitum":      "E F# G# A B C# D",        # Mixolydian
    "nid_qablim": "E F# G# A B C# D#",       # Ionian (major); same as "babylonian"
    "nis_gabari": "E F# G# A# B C# D#",      # Lydian
    "qablitum":   "E# F# G# A# B C# D#",     # Locrian
}

# Position of each note on the chain of fifths, C = 0.
_FIFTHS = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5,
           "F#": 6, "C#": 7, "G#": 8, "D#": 9, "A#": 10, "E#": 11}


def _chain_of_fifths(notes):
    """Ratios from the first note, built from pure 3:2 fifths and folded
    into one octave: the way the tablets' tunings are built."""
    first = _FIFTHS[notes[0]]
    out = []
    for n in notes:
        r = 1.5 ** (_FIFTHS[n] - first)
        while r >= 2.0:
            r /= 2.0
        while r < 1.0:
            r *= 2.0
        out.append(r)
    return sorted(out)


for _name, _notes in BABYLONIAN_TUNINGS.items():
    RATIO_SCALES[_name] = _chain_of_fifths(_notes.split())

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
# Instrument voices. Most are modelled on how the real instrument makes its
# sound, rather than recorded:
#   string     a plucked string (harp, lyre): a loop one period long that
#              feeds back through a gentle low-pass, started from the shape a
#              finger or plectrum leaves, heard through a wooden body. Higher
#              partials and higher notes die sooner, as on a real string.
#   piano      stiff strings (overtones run sharp), two or three strings per
#              note slightly out of tune with each other, a felt hammer that
#              gives fewer high partials when played softly, a fast then slow
#              decay, and a knock at the onset
#   reed       a blown double reed: pitch settles from slightly flat, vibrato
#              grows in, a slow waver, breath shaped by the pipe's resonances
#   vibraphone a tuned aluminium bar (modes at 1 : 4 : 10), a resonator, and
#              the motor-driven tremolo
# "vibes" is notate.py's original tone and keeps its original code path;
# "pad" is a synthesizer sound and stays an additive recipe:
#   harmonics  relative amplitude of partials 1, 2, 3, ...
#   held       sustain for the note's length, then release
#   attack, release, tail  seconds
#   vibrato    (rate Hz, depth as a fraction of pitch)
#   chorus     detuned copies in cents, for width
#   program    General MIDI instrument written to --midi files


# --- plucked string -------------------------------------------------------
def _lagrange3(d):
    """3rd-order Lagrange fractional-delay taps for a delay d (1 <= d < 2)."""
    h = np.ones(4)
    for k in range(4):
        for m in range(4):
            if m != k:
                h[k] *= (d - m) / (k - m)
    return h


def _body(y, sr, peaks):
    """A wooden body: the sound plus a few broad resonances (Hz, Q, gain)."""
    out = y.copy()
    for f, q, g in peaks:
        if f < sr / 2 * 0.9:
            b, a = iirpeak(f, q, fs=sr)
            out += g * lfilter(b, a, y)
    return out


def _string_note(freq, dur_s, sr, rng, decay, *, t60_ref, t60_exp, t60_min,
                 t60_max, loss, pluck_at, finger, plectrum_noise, tail, body,
                 release=0.06, **_):
    period = sr / freq                         # samples per vibration
    # loop delay: N whole samples + 1 (loss filter) + d (fractional) = period,
    # so the tuning is exact, which matters for the base-60 scales
    m = period - 1.0
    N = int(np.floor(m)) - 1
    if N < 2:                                  # above what this rate can model
        return None
    d = m - N
    t60 = float(np.clip(t60_ref * (110.0 / freq) ** t60_exp, t60_min, t60_max))
    t60 *= 2.5 / decay                         # --decay: lower = longer
    g = 10 ** (-3.0 / (t60 * freq))            # loss per trip along the string
    # ring until the hand stops it, or until it has faded to nothing
    n = int(min(max(dur_s, 0.1) + tail, max(dur_s, 0.1) + 1.3 * t60) * sr)
    h = g * np.convolve([loss, 1 - 2 * loss, loss], _lagrange3(d))

    # start: a triangle peaking where the string was plucked, its corner
    # rounded by a fleshy finger
    L = int(round(period))
    x = np.arange(L) / L
    shape = np.where(x < pluck_at, x / pluck_at, (1 - x) / (1 - pluck_at))
    if finger > 0:
        w = max(1, int(finger * L))
        shape = np.convolve(shape, np.ones(w) / w, mode="same")

    exc = np.zeros(n)
    exc[:min(L, n)] = shape[:n]
    if N < 64:
        # short loop (high note): a plain recursive filter is quickest
        den = np.zeros(N + 6)
        den[0] = 1.0
        den[N:N + 6] = -h
        y = lfilter([1.0], den, exc)
    else:
        # long loop: a block of N samples depends only on samples at least
        # N earlier, so each block is one vectorised step
        off = N + 6
        y = np.zeros(n + off)
        y[off:] = exc
        for s in range(off, n + off, N):
            e = min(s + N, n + off)
            acc = np.zeros(e - s)
            for j in range(6):
                acc += h[j] * y[s - N - j:e - N - j]
            y[s:e] += acc
        y = y[off:]

    # the string drives the body through the bridge (its slope, brighter)
    y = _body(np.diff(y, prepend=0.0) * (period / 6.0), sr, body)
    if plectrum_noise:                         # the plectrum's tick
        k = min(int(0.012 * sr), n)
        level = np.sqrt(np.mean(y[:int(0.05 * sr)] ** 2))
        tick = rng.standard_normal(k) * np.exp(-np.arange(k) / (0.0015 * sr))
        y[:k] += plectrum_noise * 3 * level * lfilter([1, -1], [1, -0.6], tick)
    r = min(int(release * sr), n)              # a hand stops the string
    y[n - r:] *= np.linspace(1, 0, r) ** 2
    return y


# --- piano ------------------------------------------------------------------
def _piano_note(freq, dur_s, sr, rng, decay, vel=96, tail=1.4, **_):
    n = int((max(dur_s, 0.1) + tail) * sr)
    t = np.arange(n) / sr
    v = vel / 127.0
    B = float(np.clip(1.0e-4 * (freq / 220.0) ** 1.2, 4e-5, 3e-3))   # stiffness
    fc = 900 + 3800 * v ** 2                   # softer felt when played gently
    tau0 = float(np.clip(2.2 * (261.6 / freq) ** 0.7, 0.35, 7.0)) * 2.5 / decay
    strings = (-1.1, 0.0, 1.3) if freq > 110 else ((-0.9, 0.9) if freq > 60 else (0.0,))
    out = np.zeros(n)
    for k in range(1, 41):
        fk = freq * k * np.sqrt(1 + B * k * k) / np.sqrt(1 + B)
        if fk > sr * 0.45:
            break
        amp = abs(np.sin(k * np.pi / 8.3)) / k ** 0.7 * np.exp(-fk / fc)  # struck near 1/8
        if amp < 2e-4:
            continue
        tau1 = tau0 / (1 + 0.6 * (fk / 1000.0) ** 1.4)
        m = min(n, int(4.5 * tau1 * 9.2 * sr))   # until it is 80 dB down
        tm = t[:m]
        env = 0.72 * np.exp(-tm / tau1) + 0.28 * np.exp(-tm / (4.5 * tau1))
        use = strings if k <= 8 else (0.0,)
        part = np.zeros(m)
        for c in use:
            part += np.sin(2 * np.pi * fk * 2 ** (c / 1200) * tm + rng.uniform(0, 2 * np.pi))
        out[:m] += amp * env * part / len(use)
    k = min(int(0.03 * sr), n)                 # the hammer's knock
    knock = rng.standard_normal(k) * np.exp(-np.arange(k) / (0.004 * sr))
    out[:k] += 0.06 * v * lfilter([0.15], [1, -0.85], knock)
    atk = max(1, int(0.0015 * sr))
    out[:atk] *= np.linspace(0, 1, atk)
    r = min(int(0.12 * sr), n)                 # dampers fall
    out[n - r:] *= np.linspace(1, 0, r) ** 2
    return out


# --- reed pipe ----------------------------------------------------------------
_REED_FORMANTS = [(1150, 600, 1.0), (2700, 900, 0.45), (550, 400, 0.4)]


def _reed_note(freq, dur_s, sr, rng, decay, release=0.12, **_):
    ring = max(dur_s, 0.1) + release
    n = int(ring * sr)
    t = np.arange(n) / sr
    scoop = -28 * np.exp(-t / 0.045)           # starts flat, settles
    vib = 9 * np.clip((t - 0.18) / 0.35, 0, 1) * np.sin(2 * np.pi * 5.0 * t)
    pts = rng.standard_normal(max(2, int(ring * 4)) + 1) * 3.0
    drift = np.interp(t, np.linspace(0, ring, len(pts)), pts)   # breath waver
    phase = 2 * np.pi * np.cumsum(freq * 2 ** ((scoop + vib + drift) / 1200)) / sr
    out = np.zeros(n)
    for k in range(1, 31):
        fk = freq * k
        if fk >= sr * 0.45:
            break
        gain = 0.25 + sum(a / (1 + ((fk - fc) / (bw / 2)) ** 2)
                          for fc, bw, a in _REED_FORMANTS)
        out += gain / k ** 0.55 * np.sin(k * phase)
    noise = rng.standard_normal(n)             # breath, through the pipe
    for fc, bw, a in _REED_FORMANTS[:2]:
        b, aa = iirpeak(fc, fc / bw, fs=sr)
        noise = noise + a * lfilter(b, aa, noise)
    noise *= 0.03 + 0.10 * np.exp(-t / 0.05)
    out = out / max(np.abs(out).max(), 1e-9) + noise * 0.25
    env = np.ones(n)
    atk = max(1, int(0.04 * sr))
    env[:atk] = np.linspace(0, 1, atk) ** 1.5
    stop = min(int(dur_s * sr), n)
    env[stop:] *= np.exp(-(t[stop:] - t[stop]) / (release / 4))
    return out * env


# --- vibraphone ---------------------------------------------------------------
def _vibraphone_note(freq, dur_s, sr, rng, decay, vel=96, tail=2.5, **_):
    n = int((max(dur_s, 0.1) + tail) * sr)
    t = np.arange(n) / sr
    v = vel / 127.0
    out = np.zeros(n)
    for ratio, amp, tau in [(1, 1.0, 3.5), (4, 0.25 * (0.4 + v), 0.6), (10, 0.06 * v, 0.15)]:
        f = freq * ratio
        if f < sr * 0.45:
            tau *= (440 / freq) ** 0.3 * 2.5 / decay
            out += amp * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    out *= 1 - 0.35 * (0.5 - 0.5 * np.cos(2 * np.pi * 5.5 * t))   # motor tremolo
    atk = max(1, int(0.001 * sr))
    out[:atk] *= np.linspace(0, 1, atk)
    r = min(int(0.1 * sr), n)
    out[n - r:] *= np.linspace(1, 0, r) ** 2
    return out


_MODELS = {"string": _string_note, "piano": _piano_note,
           "reed": _reed_note, "vibraphone": _vibraphone_note}

VOICES = {
    "vibes": dict(program=11),
    "piano": dict(model="piano", program=0),
    # concert harp: plucked a third of the way along the string by a finger,
    # long-ringing in the bass, a broad soundboard
    "harp":  dict(model="string", t60_ref=7.0, t60_exp=0.6, t60_min=1.2, t60_max=9.0,
                  loss=0.13, pluck_at=0.33, finger=0.05, plectrum_noise=0.0, tail=2.6,
                  body=[(260, 2.0, 0.6), (520, 2.5, 0.35), (1100, 3.0, 0.2)],
                  program=46),
    "pad":   dict(harmonics=(1, .35, .18, .1, .05), held=True, attack=.35,
                  release=.9, chorus=(-7, 0, 7), vibrato=(.3, .002), program=89),
    "vibraphone": dict(model="vibraphone", program=11),
    # Ancient Mesopotamian instruments, informed by what is known of them:
    # the lyre (sammûm) the Hurrian hymns were written for, as in the Lyres
    # of Ur: gut strings struck with a plectrum near the bridge, so bright
    # and shorter-lived, over a small wooden box.
    "lyre":  dict(model="string", t60_ref=4.5, t60_exp=0.5, t60_min=0.6, t60_max=4.0,
                  loss=0.10, pluck_at=0.13, finger=0.0, plectrum_noise=0.25, tail=1.3,
                  body=[(210, 3.0, 0.8), (430, 3.5, 0.6), (950, 4.0, 0.35), (2400, 4.0, 0.2)],
                  program=107),     # GM has no lyre; koto is the nearest pluck
    # The reed pipe (embūbum, also the name of a tuning), like the silver
    # pipes from Ur: a double reed, blown, so it settles, wavers and breathes.
    "reed":  dict(model="reed", program=111),   # GM shanai, a double reed
}


def _voice_note(freq, dur_s, sr, v, decay, rng, vel=96):
    """One note in a VOICES recipe (any voice except "vibes")."""
    if "model" in v:
        note = _MODELS[v["model"]](freq, dur_s, sr, rng, decay, vel=vel, **v)
        if note is not None:
            return note
        v = dict(harmonics=(1, .4, .15), tail=.5)   # beyond the model's range
    held = v.get("held", False)
    ring = max(dur_s, .1) + (v.get("release", .2) if held else v.get("tail", 1.2))
    n = int(ring * sr)
    t = np.arange(n) / sr

    clock = t
    if "vibrato" in v:
        rate, depth = v["vibrato"]
        clock = t - depth / (2 * np.pi * rate) * (np.cos(2 * np.pi * rate * t) - 1)

    wave = np.zeros(n)
    detunes = v.get("chorus", (0,))
    for cents in detunes:
        f0 = freq * 2 ** (cents / 1200)
        for k, h in enumerate(v["harmonics"], start=1):
            fk = f0 * k * np.sqrt(1 + v.get("stretch", 0) * k * k)
            if fk > sr / 2 * .95:
                break
            partial = h * np.sin(2 * np.pi * fk * clock + rng.uniform(0, 2 * np.pi))
            if not held:
                partial *= np.exp(-t * decay * v.get("decay_mul", 1)
                                  * (1 + v.get("hdecay", 0) * (k - 1)))
            wave += partial
    wave /= len(detunes)

    env = np.ones(n)
    atk = max(1, min(int(v.get("attack", .005) * sr), n))
    env[:atk] = np.linspace(0, 1, atk)
    if held:
        stop = min(int(dur_s * sr), n)
        env[stop:] *= np.exp(-(t[stop:] - t[stop]) / max(v.get("release", .2) / 4, 1e-3))

    if v.get("pluck"):
        burst = np.diff(rng.standard_normal(n + 1)) * np.exp(-t / .004)
        wave += v["pluck"] * 8 * burst
    if v.get("breath"):
        noise = rng.standard_normal(n)
        noise = np.convolve(noise, np.ones(8) / 8, mode="same")    # soften the hiss
        wave += v["breath"] * 8 * noise
    return wave * env


def synthesize(notes, bpm, sr=44100, decay=2.5,
               harmonics=(1.0, 0.45, 0.22, 0.1), voice="vibes"):
    """Render notes in one of VOICES. "vibes" is the original struck tone."""
    beat_sec = 60.0 / bpm
    if not notes:
        return np.zeros(sr, dtype=np.float32)
    if voice != "vibes":
        return _synthesize_voice(notes, beat_sec, sr, decay, VOICES[voice])
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


def _synthesize_voice(notes, beat_sec, sr, decay, v):
    rng = np.random.default_rng(0)           # same notes -> same file
    end_beat = max(n[0] + n[1] for n in notes)
    total = int((end_beat * beat_sec + 3.0) * sr)
    out = np.zeros(total, dtype=np.float64)
    for start_b, dur_b, freq, vel in notes:
        start = int(start_b * beat_sec * sr)
        note = _voice_note(freq, dur_b * beat_sec, sr, v, decay, rng, vel)
        n = min(len(note), total - start)
        if n <= 8:
            continue
        out[start:start + n] += note[:n] * (vel / 127.0) ** 1.4 * 0.25
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.85
    fade = min(int(0.05 * sr), len(out) // 4)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out.astype(np.float32)


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

    out = synthesize(notes, args.bpm, decay=args.decay, voice=args.voice)
    dry_only = out                  # keep the pure transcription for --verify

    if args.dry_mix > 0.0:
        # Blend the source back in underneath the transcription. This does not
        # recover anything — it layers the original beneath the notes. Because
        # the source still carries whatever image encoded it, decoding the
        # result will partly show that picture again.
        dry = resample(audio, int(len(audio) * 44100 / sr))
        n = max(len(out), len(dry))
        out = np.pad(out, (0, n - len(out)))
        dry = np.pad(dry, (0, n - len(dry)))
        out = (1.0 - args.dry_mix) * out + args.dry_mix * dry
        peak = np.max(np.abs(out))
        if peak > 0:
            out = out / peak * 0.85
        print(f"  dry mix: {args.dry_mix:.0%} source under the notes")

    wavfile.write(args.output, 44100, (out * 32767).astype(np.int16))

    print(f"wrote {args.output}  ({len(out)/44100:.1f}s)")

    if args.midi:
        if args.program is None:
            args.program = VOICES[args.voice]["program"]
        write_midi(args.midi, notes, args.bpm, program=args.program)
        note = " (pitches rounded to semitones)" if args.base == 60 else ""
        print(f"wrote {args.midi}  (GM program {args.program}){note}")

    if args.verify:
        # Measure the transcription itself, not the dry-mixed blend — mixing
        # the source back in would inflate the score without the notes
        # carrying any more information.
        report_verification(audio, sr, notes, dry_only, 44100, args.base)

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
    class _Fmt(argparse.RawDescriptionHelpFormatter,
               argparse.ArgumentDefaultsHelpFormatter):
        pass

    p = argparse.ArgumentParser(description=__doc__, formatter_class=_Fmt)
    sub = p.add_subparsers(dest="cmd", required=True)

    pn = sub.add_parser("notes", help="transcribe audio into musical notes",
                        formatter_class=_Fmt)
    pn.add_argument("input", help="source audio (WAV)")
    pn.add_argument("output", help="WAV to write, the notes played by --voice")
    pn.add_argument("--base",   type=int, choices=[10, 60], default=10,
                    help="10 = equal temperament; "
                         "60 = just intonation, whole-number ratios")
    pn.add_argument("--scale",  default="minor_pent",
                    help="scale to snap pitches to; `notate.py scales` lists "
                         "them for each --base (the seven Babylonian tunings "
                         "are --base 60)")
    pn.add_argument("--root",   default="A2",
                    help="root note of the scale, e.g. A2, C4, F#3")
    pn.add_argument("--bpm",    type=float, default=100.0,
                    help="tempo. With --grid it sets the time step; notes keep "
                         "their real timing, so slower means fewer, longer notes")
    pn.add_argument("--grid",   type=int,   default=8,
                    help="rhythmic grid: 4=quarter, 8=eighth, 16=sixteenth; "
                         "3/6/12 give triplet subdivisions")
    pn.add_argument("--voices", type=int,   default=3,
                    help="max simultaneous notes per grid step")
    pn.add_argument("--f-lo",   type=float, default=80.0,
                    help="lowest pitch to look for, Hz")
    pn.add_argument("--f-hi",   type=float, default=4000.0,
                    help="highest pitch to look for, Hz")
    pn.add_argument("--peak-rel", type=float, default=0.25,
                    help="peak height as fraction of the frame max (0-1); "
                         "lower = more notes")
    pn.add_argument("--decay",  type=float, default=2.5,
                    help="note decay rate; lower = longer sustain")
    pn.add_argument("--midi",   default=None, metavar="FILE",
                    help="also write the notes as a MIDI file")
    pn.add_argument("--voice", choices=list(VOICES), default="vibes",
                    help="instrument for the WAV: vibes (the original tone), "
                         "piano, harp, vibraphone, pad, and two ancient "
                         "Mesopotamian instruments, lyre (sammûm) and reed "
                         "(embūbum, reed pipe). Modelled on the real "
                         "instruments; synthesized, not recorded")
    pn.add_argument("--program", type=int, default=None,
                    help="General MIDI program for --midi; defaults to the "
                         "voice's nearest (vibes and vibraphone 11, piano 0, "
                         "harp 46, pad 89, lyre 107 koto, reed 111 shanai)")
    pn.add_argument("--verify", action="store_true",
                    help="re-synthesize the transcription and report how much "
                         "of the source's spectral structure survived")
    pn.add_argument("--transcribe-opts", default="",
                    help="comma-separated key=value options; extension point "
                         "for settings that don't warrant their own flag")
    pn.add_argument("--dry-mix", type=float, default=0.0,
                    help="blend this fraction of the source audio back under "
                         "the transcription (0.0-1.0). Layers the original "
                         "beneath the notes; does not recover anything, but "
                         "the source underneath still carries whatever image "
                         "encoded it")
    pn.add_argument("--print-notes", type=int, default=0, metavar="N",
                    help="print the first N notes as text")

    sub.add_parser("scales", help="list available scales for both bases, "
                                  "including the seven Babylonian tunings")
    return p


def main():
    args = build_parser().parse_args()
    {"notes": cmd_notes, "scales": cmd_scales}[args.cmd](args)


if __name__ == "__main__":
    main()
