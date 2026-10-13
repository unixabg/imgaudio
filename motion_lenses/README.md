# Motion lenses

A **motion lens** is a way of reading movement in a video. The lenses in
`lenses/` look at one still picture; a motion lens looks at a run of frames,
because what it measures only exists between them.

`motion.py read` finds motion lenses automatically. Drop a new `.py` file in
this directory and it becomes an option for `--motion-lens`. No edits to
`motion.py` are needed.

```bash
python3 motion.py read clip.mp4 clip.wav                      # euler, gliding tones
python3 motion.py read clip.mp4 clip.wav --sound notes --voice harp --scale nid_qablim
```

Each run writes the WAV and a text summary beside it (`clip.txt`): what the
lens found, in a few lines, and the tracks it turned into sound.

Several lenses can read the same frames: `--motion-lens euler,rhythm`. The
summary then has a section per lens and ends with `[both]`: each thing found,
whether more than one lens saw it, and pieces of one thing joined.

## Lenses

| lens | what it reads |
|---|---|
| `euler` | turns about a pivot, after Euler's rotation theorem (1775): where each mover is, where it pivots, how far it swings, its rhythm, when it was busiest. A shaken camera is noticed and taken away first. |
| `rhythm` | a rhythm map: which parts of the scene repeat their motion and how often, drawn as a small letter map with one line per region. A rhythm shared by the whole picture is named as the camera's or the light's and left out. |

## Writing a lens

Copy `_example.py` to a new file, rename it and edit. The contract:

```python
NAME = "my_lens"               # used by --motion-lens my_lens
DESCRIPTION = "what it reads"  # shown in --help
UNIT = "degrees"               # optional: unit of the track values

def read(frames, times, params):
    """
    frames: (n, rows, cols) uint8 greyscale, row 0 at the top
    times:  (n,) seconds, each frame's real time (cameras are uneven)
    params: dict of str -> str from --motion-params key=val,key=val
    returns: dict(lines=[str, ...],
                  tracks=[dict(name=str, t=array, value=array, size=float), ...],
                  findings=[dict(name=str, box=(x0, y0, x1, y1), rate=float or None), ...])
    """
```

`findings` is optional. It lets `[both]` compare your lens with the others:
each thing you found, its box as fractions of the picture (0 to 1, top left
first), and how often it repeats per second if you know.

`lines` go into the text summary. Each track becomes a voice: with
`--sound glide` its pitch follows the value (plus or minus `size` maps to
plus or minus seven semitones) and its loudness follows how fast it changes;
with `--sound notes` it plays a note on `--voice` each time it turns around,
the pitch from how far it had swung, in `--scale`.

Test a new lens on a scene where you know the answer before trusting it on
real footage, and include a still scene and a shaken one as controls.
