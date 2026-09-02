#!/bin/bash
# voice-stack.sh — transcribe one audio file repeatedly with --voices set to
# each prime in a range, then join the results into a single WAV.
#
# Two ways to join:
#   concat (default)  each version plays in turn, sparse to dense — the
#                     progression becomes audible as a form
#   --layer           all versions mixed simultaneously; same duration,
#                     accumulating density. They stay in tune with each other
#                     because every version shares the root and scale.
#
# Input must be a .wav. To start from an image, encode it first:
#   python3 imgaudio.py --auto-prep encode photo.jpg photo.wav
#
# Usage: ./voice-stack.sh INPUT.wav [OUT_DIR] [MAX_PRIME] [--layer]
#   e.g. ./voice-stack.sh drone.wav stack
#        ./voice-stack.sh drone.wav stack 40 --layer

set -euo pipefail

LAYER=0
args=()
for a in "$@"; do
    case "$a" in
        --layer) LAYER=1 ;;
        *) args+=("$a") ;;
    esac
done
set -- "${args[@]}"

IN="${1:?usage: $0 INPUT.wav [OUT_DIR] [MAX_PRIME] [--layer]}"
OUT="${2:-voice_stack}"
# Default cap is 40, not 100. Note count peaks around voices=37 and declines
# after: low-amplitude peaks jitter between neighbouring bins step to step, so
# they merge into fewer, longer, blurrier notes rather than more of them.
# Primes past 40 give denser but less articulate output.
MAX="${3:-40}"

# Shared musical settings — keep identical so the parts share a key.
BASE=60
SCALE=babylonian
ROOT=A2
BPM=70
GRID=12

# --- input checks ---------------------------------------------------------
[[ -f "$IN" ]] || { echo "no such file: $IN" >&2; exit 1; }

case "$IN" in
    *.wav|*.WAV) ;;
    *)  echo "expected a .wav — got $IN" >&2
        echo "" >&2
        echo "encode an image first, then stack the audio:" >&2
        echo "  python3 imgaudio.py --auto-prep encode \"$IN\" source.wav" >&2
        echo "  $0 source.wav" >&2
        exit 1 ;;
esac

# A bare number in the second slot is almost certainly a MAX_PRIME that
# landed in the OUT_DIR position — catch it rather than creating a "40/" dir.
if [[ "$OUT" =~ ^[0-9]+$ ]]; then
    echo "second argument is OUT_DIR, not MAX_PRIME — got '$OUT'" >&2
    echo "usage: $0 INPUT.wav [OUT_DIR] [MAX_PRIME] [--layer]" >&2
    echo "  e.g. $0 $IN stack $OUT --layer" >&2
    exit 1
fi

mkdir -p "$OUT"

# --- primes below MAX, by trial division ----------------------------------
primes=()
for ((n = 2; n < MAX; n++)); do
    is=1
    for ((d = 2; d * d <= n; d++)); do
        (( n % d == 0 )) && { is=0; break; }
    done
    (( is )) && primes+=("$n")
done

echo "input:  $IN"
echo "primes: ${primes[*]}"
echo "output: $OUT/"
echo "join:   $([ $LAYER -eq 1 ] && echo 'layered (simultaneous)' || echo 'concat (sequential)')"
echo ""

# --- transcribe at each voice count ---------------------------------------
for V in "${primes[@]}"; do
    f="$OUT/v$(printf %03d "$V").wav"
    if [[ -f "$f" ]]; then
        echo "  voices=$V  (exists, skipping)"
        continue
    fi
    n=$(python3 notate.py notes "$IN" "$f" \
            --base "$BASE" --scale "$SCALE" --root "$ROOT" \
            --bpm "$BPM" --grid "$GRID" --voices "$V" \
        | grep -oE '[0-9]+ notes' | grep -oE '[0-9]+')
    echo "  voices=$V  ->  ${n:-?} notes"
done

# --- join -----------------------------------------------------------------
echo ""
if [[ $LAYER -eq 1 ]]; then
    # Mix all versions together. Scale each input down first — summing this
    # many correlated signals clips badly before norm gets a chance to act.
    n_files=$(ls "$OUT"/v*.wav | wc -l)
    gain=$(awk -v n="$n_files" 'BEGIN{printf "%.3f", 1.0/n}')
    mix_args=()
    for f in "$OUT"/v*.wav; do mix_args+=(-v "$gain" "$f"); done
    sox -m "${mix_args[@]}" "$OUT/layered.wav" norm -3
    echo "joined: $OUT/layered.wav  ($(soxi -D "$OUT/layered.wav" | cut -d. -f1)s, $n_files versions mixed)"
else
    # Join in ascending order of voice count: sparse to dense.
    # ffmpeg's concat demuxer resolves paths relative to the list file's own
    # directory, so entries must be bare filenames, not OUT/-prefixed.
    ls "$OUT"/v*.wav | sort | xargs -n1 basename | sed 's|^|file |' \
        > "$OUT/list.txt"
    ffmpeg -y -loglevel error -f concat -safe 0 -i "$OUT/list.txt" \
        -c copy "$OUT/stacked.wav"
    echo "joined: $OUT/stacked.wav  ($(soxi -D "$OUT/stacked.wav" | cut -d. -f1)s)"
fi
