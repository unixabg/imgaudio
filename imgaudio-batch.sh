#!/bin/bash
# imgaudio-batch.sh - encode all frames as audio
# Resumable: skips frames whose audio already exists.

set -euo pipefail

mkdir -p out_music out_data

for f in frames/*.jpg; do
  name=$(basename "$f" .jpg)

  # Musical version for listening
  if [[ ! -f "out_music/${name}.wav" ]]; then
    python3 imgaudio.py --auto-prep --rows 300 --cols 600 \
        encode "$f" "out_music/${name}.wav"
  fi

  # Data version for guaranteed recovery (uncomment when ready)
  # if [[ ! -f "out_data/${name}.wav" ]]; then
  #   python3 imgaudio.py --lossless encode "$f" "out_data/${name}.wav"
  # fi
done
