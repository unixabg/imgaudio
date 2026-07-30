# Lenses

A **lens** is a way of looking at an image — a function that surfaces some
hidden structure in it. Where `imgaudio.py` by default sonifies raw
brightness, a lens first extracts a specific kind of pattern (edges, fractal
roughness, foraging paths, etc.) and sonifies *that* instead.

`imgaudio.py` discovers lenses automatically. Drop a new `.py` file in this
directory and it will appear as an option in `--lens`. No edits to
`imgaudio.py` are required.

## Writing a lens

Copy `_example.py` to a new file, rename, and edit. The contract is three things:

```python
NAME = "my_lens"              # short identifier used by --lens my_lens
DESCRIPTION = "what it does"  # shown in --help

def analyze(grid, params):
    """
    grid:   numpy array, shape (rows, cols), float32 values in [0.0, 1.0]
            (already preprocessed if --auto-prep was used)
    params: dict of str->str from --lens-params key=val,key=val
            Use float(params.get("threshold", "0.3")) to read with a default.
    returns: numpy array, same shape, values in [0.0, 1.0]
    """
    # ... your processing ...
    return new_grid
```

That's it. No classes, no inheritance, no decorators. The lens runs after
adaptive preprocessing but before the gamma curve and audio synthesis, so
returning a different brightness distribution is what changes the resulting
sound.

## Listing available lenses

```bash
python3 imgaudio.py --help
```

The `--lens` line in the help output enumerates every loaded lens with its
description.

## Using a lens

```bash
# Default — no lens, original behavior
python3 imgaudio.py encode photo.jpg out.wav

# Apply a lens
python3 imgaudio.py --lens edges encode photo.jpg out.wav

# Pass parameters to the lens
python3 imgaudio.py --lens edges --lens-params threshold=0.4,blur=2 \
    encode photo.jpg out.wav
```

## Conventions

- One file per lens. Filename is also the module name (e.g. `edges.py` →
  imported as `lenses.edges`).
- Files starting with `_` (like `_example.py`) are skipped by the loader.
  Use this for templates and helpers you don't want exposed as lenses.
- The lens should NOT modify the input grid in place — return a new array.
- The lens should NOT read or write files, network, or environment. It's a
  pure function from grid to grid. This keeps batch processing reliable.
- Keep lenses small and focused. If you find yourself writing 200 lines,
  consider whether you're actually building two lenses.
