#!/usr/bin/env python3
"""
hidden-tally.py — which faint rhythms keep coming back across many segments?

Run the hidden lens on a folder of segments, then tally its reports:

  mkdir -p out
  for f in segments/*.mp4; do
    python3 motion.py read "$f" "out/$(basename "${f%.mp4}").wav" --motion-lens hidden
  done
  python3 examples/hidden-tally.py out/*.txt

A rhythm that turns up in every segment, whatever the weather and the light,
is most likely the camera's own habit. One that comes and goes with wind, sun
or time of day belongs to the world. Places on the camera's printed clock
are marked (clock).
"""
import collections
import re
import sys

STEP = 0.02                                       # rates closer than this count as one


def main(paths):
    found = collections.defaultdict(list)     # rate -> [(segment, place, channel, kind, clock)]
    shared = collections.defaultdict(set)     # rate -> segments where the whole picture had it
    combs = collections.defaultdict(set)
    for path in paths:
        seg = path.split("/")[-1].rsplit(".", 1)[0]
        for line in open(path, encoding="utf-8"):
            m = re.match(r"found \d+ \(([a-z ,]+)[:)].*?\): (\w+ \w+) \(x.*?repeats ([\d.]+) times"
                         r".*? in ([a-z-]+),", line)
            if m:
                r = round(float(m.group(3)) / STEP) * STEP
                found[r].append((seg, m.group(2), m.group(4), m.group(1),
                                 "printed clock" in line or "edge of the picture" in line))
            if line.startswith("camera or light:"):
                c = re.search(r"([\d.]+) a second and its multiples", line)
                if c:
                    combs[float(c.group(1))].add(seg)
    n = len(paths)
    print(f"{n} segment report{'s' if n != 1 else ''} read\n")
    if combs:
        print("the whole picture's comb of rates (the camera's fresh-picture rhythm):")
        for b, segs in sorted(combs.items()):
            print(f"  {b:.2f} a second and multiples (every {1 / b:.1f} s): in {len(segs)} of {n}")
        print()
    print("faint rhythms found in one place, most widespread first:")
    for r, hits in sorted(found.items(), key=lambda kv: (-len({h[0] for h in kv[1]}), kv[0])):
        segs = {h[0] for h in hits}
        places = sorted({f"{h[1]}{' (clock)' if h[4] else ''} in {h[2]}" for h in hits})
        print(f"  {r:.2f} a second (every {1 / r:.1f} s): in {len(segs)} of {n} segments; "
              + "; ".join(places[:4]) + (f"; and {len(places) - 4} more" if len(places) > 4 else ""))
    if not found:
        print("  none")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
