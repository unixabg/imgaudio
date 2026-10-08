// worker.js — runs Python off the main thread so the page stays responsive.
// Loads Pyodide from jsDelivr, fetches the repo's own .py files, and answers
// "make" requests with WAV, PNG and MIDI bytes.

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/";
importScripts(PYODIDE + "pyodide.js");

let py = null;
const say = (type, data = {}) => postMessage({ type, ...data });

async function boot() {
  say("stage", { text: "Starting Python" });
  py = await loadPyodide({ indexURL: PYODIDE });

  say("stage", { text: "Loading numpy, scipy and Pillow" });
  await py.loadPackage(["numpy", "scipy", "pillow"], {
    messageCallback: (m) => {
      const hit = /Loading ([\w-]+)/.exec(m);
      if (hit) say("stage", { text: "Loading " + hit[1] });
    },
  });

  say("stage", { text: "Loading imgaudio" });
  const files = await (await fetch("py/manifest.json")).json();
  py.FS.mkdirTree("/app/lenses");
  py.FS.mkdirTree("/work");
  await Promise.all(files.map(async (f) => {
    const src = await (await fetch("py/" + f)).text();
    py.FS.writeFile("/app/" + f, src);
  }));
  py.FS.writeFile("/app/bridge.py", await (await fetch("bridge.py")).text());

  py.runPython(`
import os, sys
sys.path.insert(0, "/app")
os.chdir("/app")
import bridge
`);
  say("ready", { catalog: JSON.parse(py.runPython("bridge.catalog()")) });
}

function read(path) {
  try { return py.FS.readFile(path); } catch { return null; }
}

onmessage = async ({ data }) => {
  if (data.type !== "make") return;
  try {
    for (const f of ["sound.wav", "sound.chroma.png", "picture.png", "melody.wav", "melody.mid", "melody.png"]) {
      try { py.FS.unlink("/work/" + f); } catch {}
    }
    py.FS.writeFile("/work/photo.jpg", new Uint8Array(data.photo));
    py.globals.set("opts_json", JSON.stringify(data.opts));
    const log = py.runPython("bridge.make(opts_json)");
    const out = {
      sound: read("/work/sound.wav"),
      picture: read("/work/picture.png"),
      melody: read("/work/melody.wav"),
      midi: read("/work/melody.mid"),
      melodyPicture: read("/work/melody.png"),
    };
    postMessage({ type: "done", log, ...out },
      Object.values(out).filter(Boolean).map((a) => a.buffer));
  } catch (err) {
    say("error", { text: String(err.message || err) });
  }
};

boot().catch((err) => say("error", { text: "Python failed to start. " + (err.message || err), fatal: true }));
