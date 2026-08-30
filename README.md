# Year of Frames — image → sound → music

Turns timelapse photographs into audio, then into playable music. Each tool is
standalone; chain them or stop anywhere.

```
photos ──► spectrogram audio ──► notes / MIDI
              │
              ├──► recurrence plots (structure + anomalies)
              └──► audio diffs (what changed)
```

## Setup

```bash
sudo apt install python3-venv imagemagick ffmpeg sox

mkdir -p ~/imgaudio && cd ~/imgaudio
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy Pillow matplotlib
```

Activate with `source .venv/bin/activate` each session, or call the venv python
directly: `~/imgaudio/.venv/bin/python3 imgaudio.py ...`

## Layout

```
imgaudio.py              image ↔ audio, with lens support
notate.py                audio → musical notes + MIDI
anomaly.py               recurrence plots + anomaly ranking
flow_prep.py             consecutive frames → motion images
lenses/                  edges.py, fractal.py, phyllotaxis.py, spectral.py, _example.py
imgaudio-batch.sh        encode every frame
imgaudio-diff-batch.sh   per-frame diffs + change_log.txt
lens_recursion.sh        feed an image through lenses N times
knobs.sh                 sweep anomaly.py settings on one file
frames/  out/  flow/     data (gitignored)
```

## Shortest path — one image to MIDI

```bash
python3 imgaudio.py --auto-prep encode photo.jpg tmp.wav
python3 notate.py notes tmp.wav melody.wav --midi melody.mid
```

~5 seconds. `melody.wav` plays immediately; `melody.mid` opens in a DAW.

## Full workflow

**1. Collect frames.** Name them so alphabetical = chronological:
`frames/image-YYYYMMDD-HHMMSS.jpg`. Every batch script relies on this.

Start hourly (8,760 frames/year, ~3 days encode) rather than per-minute
(525,600 frames, months). Re-run finer later if it's worth it.

**2. Check settings on one frame.**

```bash
python3 imgaudio.py --auto-prep roundtrip frames/first.jpg --out-dir check/
```

Open `check/first_roundtrip.png`. Middle panel should resemble the top panel.

`--auto-prep` is not optional for outdoor photos — it percentile-clips each
frame and inverts if still mostly bright. Without it, blown-out midday frames
encode as a constant chord.

**3. Batch encode.**

```bash
./imgaudio-batch.sh
```

Resumable — skips frames already done. Parallelize:

```bash
ls frames/*.jpg | parallel -j $(nproc) --bar \
  'test -f out/{/.}.wav || python3 imgaudio.py --auto-prep --rows 300 --cols 600 encode {} out/{/.}.wav'
```

**4. Optional — lenses.** Sonify hidden structure instead of raw brightness.

```bash
python3 imgaudio.py --help                       # lists installed lenses
python3 imgaudio.py --auto-prep --lens fractal encode img.jpg out.wav
```

| lens | surfaces |
|---|---|
| `edges` | contours (retinal lateral inhibition) |
| `fractal` | texture roughness — natural vs man-made |
| `phyllotaxis` | golden-angle spiral read order, center-out |
| `spectral` | 2D FFT — spatial frequency and orientation |

`spectral` has four modes via `--lens-params mode=...`:

| mode | keeps | discards |
|---|---|---|
| `magnitude` | full centered spectrum | nothing |
| `quadrant` | one quarter (the FFT of a real image is conjugate-symmetric, so the rest is redundant) | the mirror half |
| `radial` | what scales are present | orientation |
| `angular` | what directions are present | scale |

Man-made scenes are full of periodic structure — brick courses, siding,
mullions — and each appears as a discrete spike. Vegetation and clouds spread
diffusely. Use square `--rows`/`--cols` with this lens; the spectrum is
naturally square and a 200×400 grid stretches it.

Lenses also work on `decode`, transforming an audio's spectrogram into abstract
art. Look for `lens: <name>` in the output — if it's missing, it didn't fire.

**5. Optional — motion.**

```bash
python3 flow_prep.py batch frames/ flow/
python3 imgaudio.py --auto-prep encode flow/flow_x.png motion.wav
```

Static regions go silent; movement goes loud.

**6. Optional — find what changed.**

```bash
./imgaudio-diff-batch.sh          # writes change_log.txt, ranked by RMS
python3 anomaly.py recurrence out/img.wav rp.png --discords 15
```

Encode with `--no-normalize` for diffs to cancel cleanly.

Recurrence plots work best **one per frame**, not one per year — a year at
`--size 1000` averages each pixel over hours of drone and comes out gray.

```bash
for f in out/*.wav; do
  python3 anomaly.py --size 600 --bands 128 recurrence "$f" "rp/$(basename "$f" .wav).png"
done
montage rp/*.png -tile 24x -geometry 80x rp/grid.png
```

**7. Assemble and notate.**

```bash
sox out/*.wav year.wav norm -3

python3 notate.py notes year.wav year_music.wav \
    --scale minor_pent --root A2 --bpm 70 --grid 4 --voices 2 \
    --midi year_music.mid

ffmpeg -i year_music.wav -c:a libopus -b:a 96k year_music.opus
```

For very large sets, `sox out/*.wav` overflows the argument list — use
`ls out/*.wav | sort | sed 's|^|file |' > list.txt` then
`ffmpeg -f concat -safe 0 -i list.txt -c copy year.wav`.

## Tuning systems

`notate.py --base` selects how detected pitches are snapped.

**`--base 10`** (default) is twelve-tone equal temperament: pitch is
`440 · 2^((n−69)/12)`. The twelfth root of two is irrational and terminates in
no number base at all.

**`--base 60`** is just intonation: every degree is a small whole-number ratio
to the root. A fifth is 3/2, a fourth 4/3, a whole tone 9/8. These terminate
*exactly* in sexagesimal (3/2 = 1;30, 4/3 = 1;20, 9/8 = 1;07,30) — which is
why base 60 was adopted for arithmetic, and why Mesopotamian tuning was built
from cycles of fifths and fourths.

```bash
python3 notate.py scales                                    # both tables
python3 notate.py notes in.wav out.wav --base 60 --scale babylonian
python3 notate.py notes in.wav out.wav --base 60 --scale harmonic
```

Scales derived from stacked fifths (`babylonian`, `babylonian_pent`) land
within ~6 cents of equal temperament — cleaner intervals, but familiar. The
audible difference comes from scales using higher partials: `harmonic` reaches
~49 cents, close to a quarter-tone, because the 7th and 11th partials have no
equal-tempered equivalent at all.

Standard MIDI notes are equal-tempered integers, so `--midi` rounds base-60
pitches to the nearest semitone. The WAV carries the true tuning; the `.mid` is
an approximation.

## Measuring what survives

Transcription is lossy by design. Scale snapping, grid quantization, the
`--voices` cap, and note merging each throw away information in exchange for
musicality. `--verify` measures how much is left.

```bash
python3 notate.py notes in.wav out.wav --verify
```

It re-synthesizes the transcription and correlates its log-band spectrogram
against the source's:

```
  --- verification ---
  spectral fidelity   0.747   (1.0 identical, 0.04 noise, 0.0 silence)
  distinct pitches    30 of 392 notes
  note density        24.5 notes/sec
  pitch residual      mean  18.6 cents, max  48.7
  alphabet size       30 symbols -> 4.9 bits per note
```

The metric reads 1.0 for identical signals, ~0.04 for white noise, 0.0 for
silence, and degrades monotonically as noise is added. It turns parameter
choice from taste into measurement — at least for the question of how much
gets through. Taste still governs which one sounds good.

`--transcribe-opts key=val,...` is an extension point for future options, same
shape as `--lens-params`.

## Findings

Measured with `--verify`. **Read the caveats — most of this is n=1.**

**Holds up.** Base 60 beat base 10 in all four spectral-lens modes
(quadrant, angular, magnitude, radial), on both fidelity *and* alphabet size.
Four independent comparisons, consistent direction. Ratio snapping offers more
distinct landing places than twelve equal-tempered pitch classes, so fewer
peaks collide onto the same symbol.

**Dominates everything else: source type.** Spectral-lens drone scores ~0.74;
a field recording of frogs scores ~0.38. Which scale you pick is worth ~0.02.
What you feed it is worth ~0.35.

**Does not replicate.** On the original source, `harmonic` (0.737) beat
`just_major` (0.721) and `babylonian` (0.717). Retested on four other files,
`harmonic` won 2 of 4 with margins under 0.022 — within noise. The mechanism
is appealing (additive synthesis puts energy at harmonic partials; the
harmonic scale's degrees *are* harmonic partials) but a plausible mechanism is
a story, not evidence.

**Untested beyond n=1.** On one 20-second source, fidelity peaked at
`--voices 7` and `--grid 16`. Voices and grid proved separable — adding voices
cost a flat ~0.010 at every grid value. Grid 12 tied grid 8 at 43% higher note
density, which is the only place the sexagesimal argument extended from pitch
into rhythm. All of this could be an artifact of that file's length and
content.

Two operating points from that single sweep, offered as starting guesses
rather than recommendations:

```bash
# best reproduction on the file tested — 0.747, 30 pitches, 24.5 notes/sec
python3 imgaudio.py --auto-prep --lens spectral --lens-params mode=quadrant \
    --rows 400 --cols 400 encode photo.jpg spec.wav
python3 notate.py notes spec.wav out.wav \
    --base 60 --scale harmonic --voices 7 --grid 16 --verify

# more throughput for 0.010 fidelity — 36 pitches, 32.8 notes/sec
python3 notate.py notes spec.wav out.wav \
    --base 60 --scale harmonic --voices 17 --grid 16 --verify
```

Baseline for comparison: raw drone, base 10, `minor_pent`, defaults → 0.470.

## Key parameters

**imgaudio.py**

| flag | default | effect |
|---|---|---|
| `--auto-prep` | off | adaptive exposure handling — use it |
| `--rows` / `--cols` | 200 / 400 | vertical / horizontal resolution |
| `--col-sec` | 0.05 | seconds per column |
| `--f-lo` / `--f-hi` | 80 / 8000 | frequency band, Hz |
| `--gamma` | 1.7 | contrast; higher = starker |
| `--lossless` | off | byte-exact archive; audio is harsh noise |
| `--no-normalize` | off | needed for clean diffs |
| `--lens` / `--lens-params` | raw | apply a lens |

**notate.py**

| flag | default | effect |
|---|---|---|
| `--base` | 10 | 10 = equal temperament, 60 = just intonation |
| `--scale` / `--root` | minor_pent / A2 | run `notate.py scales` for both tables |
| `--bpm` / `--grid` | 100 / 8 | tempo; 4=quarters, 8=eighths, 16=sixteenths; 3/6/12 give triplets |
| `--voices` | 3 | max simultaneous notes |
| `--peak-rel` | 0.25 | density dial; lower = many more notes |
| `--decay` | 2.5 | lower = longer sustain |
| `--program` | 11 | GM instrument: 0 piano, 11 vibes, 46 harp, 89 pad |
| `--verify` | off | report how much of the source survived |
| `--transcribe-opts` | — | key=value extension point |

**anomaly.py**

| flag | default | effect |
|---|---|---|
| `--size` | 1000 | frames per axis; matrix is N², so 4000 needs ~250 MB |
| `--bands` | 64 | spectral bands per frame vector; higher sharpens striping |
| `--mode` | continuous | `binary` with `--percentile` gives the stark print look |
| `--discords` | 0 | also print the N most anomalous timestamps |

## Writing a lens

Copy `lenses/_example.py`, rename, implement three things:

```python
NAME = "my_lens"
DESCRIPTION = "one line, shown in --help"

def analyze(grid, params):
    # grid: (rows, cols) float32 in [0,1]; params: dict[str,str]
    return new_grid
```

Auto-discovered, no registration. Two rules learned the hard way:

- **Filename must match `NAME`** — `imgaudio.py` registers by `NAME`,
  `lens_recursion.sh` discovers by filename.
- **Keep output dense** — a lens producing >90% zeros collapses to silence
  after two or three recursions. This is why `phyllotaxis` smears each spiral
  sample across rows with a Gaussian instead of placing a single bright pixel.

## Gotchas

- `--lossless` ignores lenses and all spectrogram parameters. It's a byte copy.
- Encode and decode must use the same `--rows`, `--cols`, `--f-lo`, `--f-hi`.
- A lens during encode means the round-trip reconstructs what the lens *saw*,
  not the original. That's the point, but it surprises you the first time.
- `--base 60` with a base-10 scale name exits with "unknown scale". The default
  `minor_pent` is auto-swapped for `babylonian_pent`, but other base-10 names
  will fail — run `notate.py scales` to see both tables.
- The `spectral` lens output is conjugate-symmetric, so its audio has partials
  in mirrored pairs and sounds unusually consonant for reasons that have
  nothing to do with the picture. Use `mode=quadrant` to break the symmetry.
- `matplotlib: divide by zero in log10` on silent bins — harmless.
- Badly clipped sources need help beyond `--auto-prep`:
  `convert in.jpg -colorspace Gray -negate -level 0%,30%,1.2 prepped.jpg`
- Run `anomaly.py discords` against motion audio, not brightness audio — a year
  of drone is so self-similar that "most anomalous" just means "loudest."
- Transcribing `notate.py` output scores deceptively high (~0.89). The source
  is already sparse, pitched, and quantized, so `--verify` measures
  self-consistency rather than channel capacity. Test on unfamiliar material.

## The tradeoff

Raw bytes: perfect reconstruction, unlistenable noise. Pentatonic mapping:
music, no reconstruction. Spectrogram synthesis sits between, and every lens
trades a little more fidelity for a little more meaning.

Optimizing for `--verify` fidelity and optimizing for something worth hearing
are different objectives. `just_pent` scores worst of the base-60 scales
(0.657) despite being the prettiest — five degrees isn't enough resolution to
carry a signal. The metric tells you what got through, not whether it was
worth transmitting.
