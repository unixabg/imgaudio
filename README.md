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
lenses/                  edges.py, fractal.py, phyllotaxis.py, _example.py
imgaudio-batch.sh        encode every frame
imgaudio-diff-batch.sh   per-frame diffs + change_log.txt
lens_recursion.sh        feed an image through lenses N times
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
ls frames/*.jpg | parallel -j 8 \
  'python3 imgaudio.py --auto-prep --rows 300 --cols 600 encode {} out/{/.}.wav'
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
| `--scale` / `--root` | minor_pent / A2 | run `notate.py scales` for the list |
| `--bpm` / `--grid` | 100 / 8 | tempo; 4=quarters, 8=eighths, 16=sixteenths |
| `--voices` | 3 | max simultaneous notes |
| `--peak-rel` | 0.25 | density dial; lower = many more notes |
| `--decay` | 2.5 | lower = longer sustain |
| `--program` | 11 | GM instrument: 0 piano, 11 vibes, 46 harp, 89 pad |

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
  after two or three recursions.

## Gotchas

- `--lossless` ignores lenses and all spectrogram parameters. It's a byte copy.
- Encode and decode must use the same `--rows`, `--cols`, `--f-lo`, `--f-hi`.
- A lens during encode means the round-trip reconstructs what the lens *saw*,
  not the original. That's the point, but it surprises you the first time.
- `matplotlib: divide by zero in log10` on silent bins — harmless.
- Badly clipped sources need help beyond `--auto-prep`:
  `convert in.jpg -colorspace Gray -negate -level 0%,30%,1.2 prepped.jpg`
- Run `anomaly.py discords` against motion audio, not brightness audio — a year
  of drone is so self-similar that "most anomalous" just means "loudest."

## The tradeoff

Raw bytes: perfect reconstruction, unlistenable noise. Pentatonic mapping:
music, no reconstruction. Spectrogram synthesis sits between, and every lens
trades a little more fidelity for a little more meaning.
