#!/bin/bash
# build.sh — assemble the static web app into one folder.
# The page runs the repo's own imgaudio.py, notate.py and lenses/ unchanged,
# so this copies them in rather than keeping a second copy under web/.
#
#   ./web/build.sh _site                     # assemble
#   python3 -m http.server -d _site 8000     # try it at http://localhost:8000

set -euo pipefail

out="${1:-_site}"
root="$(cd "$(dirname "$0")/.." && pwd)"

rm -rf "$out"
mkdir -p "$out/py/lenses"
cp "$root"/web/{index.html,worker.js,bridge.py,manifest.webmanifest} "$out/"
cp -r "$root"/web/icons "$out/"
cp "$root"/imgaudio.py "$root"/notate.py "$out/py/"
cp "$root"/lenses/*.py "$out/py/lenses/"

# The worker reads this list to know which Python files to load.
( cd "$out/py" && ls imgaudio.py notate.py lenses/*.py ) |
  python3 -c 'import sys, json; print(json.dumps([l.strip() for l in sys.stdin]))' \
  > "$out/py/manifest.json"

touch "$out/.nojekyll"
echo "built $out"
