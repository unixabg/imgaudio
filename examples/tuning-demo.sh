#!/bin/bash
# tuning-demo.sh — hear the difference between tuning systems.
#
# Encodes one image, then transcribes it five times with everything held
# constant except the tuning. Same source, same voices, same grid — only
# --base and --scale change.
#
# Usage: ./examples/tuning-demo.sh [IMAGE] [OUT_DIR]

set -euo pipefail

IMG="${1:-input.jpg}"
OUT="${2:-tuning_demo}"

[[ -f "$IMG" ]] || { echo "no such image: $IMG" >&2; exit 1; }
mkdir -p "$OUT"

echo "source: $IMG"
echo

# One encode, shared by every transcription, so the only variable is tuning.
# Square rows/cols because the spectral lens produces a square spectrum.
python3 imgaudio.py --auto-prep --lens spectral --lens-params mode=quadrant \
    --rows 400 --cols 400 encode "$IMG" "$OUT/source.wav"
echo

# base, scale, one-line description
TUNINGS=(
  "10 minor_pent  equal temperament — the modern default, irrational ratios"
  "60 babylonian  stacked fifths, the Mesopotamian tuning tablets (~6 cents off)"
  "60 just_major  classical just intonation, pure thirds and fifths"
  "60 harmonic    partials 8-16 of the harmonic series (~49 cents off — a quarter-tone)"
  "60 just_pent   five degrees only — prettiest, and the worst at carrying signal"
)

for entry in "${TUNINGS[@]}"; do
    read -r base scale desc <<< "$entry"
    f="$OUT/${base}_${scale}.wav"
    echo "--- base $base, $scale"
    echo "    $desc"
    python3 notate.py notes "$OUT/source.wav" "$f" \
        --base "$base" --scale "$scale" --voices 7 --grid 16 --verify \
        | grep -E 'spectral fidelity|deviation from 12-TET' | sed 's/^/    /'
    echo
done

echo "listen in order:"
echo "  for f in $OUT/*.wav; do echo \"\$f\"; play -q \"\$f\"; done"
echo
echo "10_minor_pent and 60_babylonian should sound nearly identical."
echo "60_harmonic should sound audibly other."
