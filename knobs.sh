#!/bin/bash
# knobs.sh — sweep anomaly.py parameters against one audio file so you can
# compare the results side by side and pick settings you like.
#
# Usage: ./knobs.sh INPUT.wav [OUT_DIR]

set -euo pipefail

F="${1:-}"
OUT="${2:-knobs}"

if [[ -z "$F" ]]; then
    echo "Usage: $0 INPUT.wav [OUT_DIR]" >&2
    echo "  e.g. $0 out/image-20250101-000101.wav" >&2
    exit 1
fi
if [[ ! -f "$F" ]]; then
    echo "no such file: $F" >&2
    exit 1
fi

mkdir -p "$OUT"
echo "sweeping $F -> $OUT/"

# --- size sweep (detail vs compute; 1600 is ~16x the work of 400) ---
for S in 400 800 1600; do
    python3 anomaly.py --size "$S" recurrence "$F" "$OUT/size_$(printf %04d "$S").png"
done

# --- bands sweep (spectral resolution; sharpens the striping) ---
for B in 32 64 128 256; do
    python3 anomaly.py --size 600 --bands "$B" recurrence "$F" \
        "$OUT/bands_$(printf %03d "$B").png"
done

# --- binary threshold sweep (graphic/print look) ---
for P in 5 10 20 35; do
    python3 anomaly.py --size 600 recurrence "$F" \
        "$OUT/bin_p$(printf %02d "$P").png" --mode binary --percentile "$P"
done

# --- inverted (white field, dark structure) ---
python3 anomaly.py --size 600 recurrence "$F" "$OUT/inverted.png" --invert

# --- contact sheet, if imagemagick is around ---
if command -v montage >/dev/null 2>&1; then
    montage "$OUT"/*.png -tile 4x -geometry 260x -label '%f' "$OUT/sheet.png"
    echo "contact sheet: $OUT/sheet.png"
fi

echo "done — $(ls "$OUT"/*.png | wc -l) images in $OUT/"
