#!/bin/bash
# imgaudio-diff-batch.sh - compute per-frame audio diffs to find when things changed.

set -euo pipefail

mkdir -p diffs

prev=""
for f in out/*.wav; do
  if [[ -n "$prev" ]]; then
    name=$(basename "$f" .wav)
    if [[ ! -f "diffs/${name}.wav" ]]; then
      sox -m -v 1.0 "$prev" -v -1.0 "$f" "diffs/${name}.wav" norm -3 2>/dev/null
    fi
    rms=$(sox "diffs/${name}.wav" -n stat 2>&1 | grep "RMS *amplitude" | awk '{print $3}')
    echo "$name $rms"
  fi
  prev="$f"
done > change_log.txt

# Top 20 biggest changes
echo "Biggest changes:"
sort -k2 -nr change_log.txt | head -20
