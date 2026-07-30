#!/bin/bash
# lens_recursion.sh - feed an image through a lens N times, saving each step.
# Resumable: skips iterations whose output already exists.
#
# Usage:
#   ./lens_recursion.sh IMAGE N                              (resume by default)
#   ./lens_recursion.sh IMAGE N --lens edges                 (just one lens)
#   ./lens_recursion.sh IMAGE N --out-dir my_results/        (custom output)
#   ./lens_recursion.sh IMAGE N --force                      (redo everything)
#   ./lens_recursion.sh IMAGE N --skip-completed-lenses      (skip whole lens if final step exists)

set -euo pipefail

# --- parse args -----------------------------------------------------------
IMG=""
N=""
LENS=""
OUT_ROOT="recursion_out"
FORCE=0
SKIP_COMPLETED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lens)                    LENS="$2"; shift 2 ;;
        --out-dir)                 OUT_ROOT="$2"; shift 2 ;;
        --force)                   FORCE=1; shift ;;
        --skip-completed-lenses)   SKIP_COMPLETED=1; shift ;;
        -h|--help)
            cat <<EOF
Usage: $0 IMAGE N [OPTIONS]

  IMAGE                       input image (.jpg/.png)
  N                           number of iterations per lens

Options:
  --lens NAME                 run just one lens (default: all)
  --out-dir DIR               parent output directory (default: recursion_out)
  --force                     redo every iteration even if output exists
  --skip-completed-lenses     skip a lens entirely if its final step exists

By default, the script resumes by skipping iterations whose output PNG
already exists. To re-run an iteration, delete its step_NNN.png or use
--force.
EOF
            exit 0 ;;
        *)
            if [[ -z "$IMG" ]]; then IMG="$1"
            elif [[ -z "$N" ]]; then N="$1"
            else echo "Unknown argument: $1" >&2; exit 1
            fi
            shift ;;
    esac
done

if [[ -z "$IMG" || -z "$N" ]]; then
    echo "Usage: $0 IMAGE N [--lens NAME] [--out-dir DIR] [--force]" >&2
    exit 1
fi

# --- discover lenses (reads NAME constant from each file) -----------------
discover_lenses() {
    local d="lenses"
    if [[ ! -d "$d" ]]; then
        echo "no lenses/ directory found" >&2
        exit 1
    fi
    for f in "$d"/*.py; do
        local base name
        base=$(basename "$f" .py)
        [[ "$base" == _* || "$base" == "__init__" ]] && continue
        name=$(grep -E '^NAME[[:space:]]*=' "$f" \
               | head -1 \
               | sed -E 's/^NAME[[:space:]]*=[[:space:]]*["'"'"']([^"'"'"']+).*/\1/')
        echo "${name:-$base}"
    done
}

# --- decide which lenses to run -------------------------------------------
if [[ -n "$LENS" ]]; then
    LENSES=("$LENS")
else
    mapfile -t LENSES < <(discover_lenses)
fi

echo "Image:      $IMG"
echo "Iterations: $N per lens"
echo "Lenses:     ${LENSES[*]}"
echo "Output:     $OUT_ROOT/"
echo "Mode:       $([ $FORCE -eq 1 ] && echo 'force redo' || echo 'resume (skip existing)')"
echo ""

# --- run the recursion for each lens --------------------------------------
ext="${IMG##*.}"
mkdir -p "$OUT_ROOT"

for L in "${LENSES[@]}"; do
    DIR="$OUT_ROOT/$L"
    mkdir -p "$DIR"
    FINAL_STEP=$(printf "%03d" "$N")

    # Lens-level skip: if the final iteration already exists and the flag is set, skip entirely
    if [[ $SKIP_COMPLETED -eq 1 && -f "$DIR/step_${FINAL_STEP}.png" && $FORCE -eq 0 ]]; then
        echo "[$L] all $N iterations already done — skipping lens"
        continue
    fi

    # Make sure step_000 exists (the starting image)
    if [[ ! -f "$DIR/step_000.${ext}" ]]; then
        cp "$IMG" "$DIR/step_000.${ext}"
    fi

    echo "=========================================="
    echo "  LENS: $L"
    echo "=========================================="

    current="$DIR/step_000.${ext}"

    for i in $(seq 1 "$N"); do
        step=$(printf "%03d" "$i")
        audio="$DIR/step_${step}.wav"
        next="$DIR/step_${step}.png"

        # Iteration-level skip: if this step's PNG exists and we're not forcing, skip
        if [[ -f "$next" && $FORCE -eq 0 ]]; then
            echo "[$L] $step  ✓ already exists, skipping"
            current="$next"
            continue
        fi

        echo "[$L] $step  computing..."
        python3 imgaudio.py --auto-prep --rows 400 --cols 800 \
            --lens "$L" encode "$current" "$audio"
        python3 imgaudio.py --rows 400 --cols 800 \
            decode "$audio" "$next"

        current="$next"
    done
    echo ""
done

# --- summary --------------------------------------------------------------
echo "=========================================="
echo "Done. Outputs under $OUT_ROOT/:"
for L in "${LENSES[@]}"; do
    count=$(ls "$OUT_ROOT/$L/"step_*.png 2>/dev/null | wc -l)
    echo "  $L/  ($count iterations)"
done

# --- contact sheets (optional) --------------------------------------------
if command -v montage >/dev/null 2>&1; then
    echo ""
    echo "Generating contact sheets..."
    for L in "${LENSES[@]}"; do
        sheet="$OUT_ROOT/$L/contact_sheet.png"
        montage -tile 4x -geometry 240x -label '%t' \
            "$OUT_ROOT/$L/"step_*.{png,$ext} 2>/dev/null \
            "$sheet" 2>/dev/null || true
        [[ -f "$sheet" ]] && echo "  $sheet"
    done
fi
