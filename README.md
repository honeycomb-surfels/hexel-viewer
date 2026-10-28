# Hexel viewer

Two viewers for pretrained V1/V2/V3 Hexels: a standalone WebGL2 browser viewer
and a CUDA-server viewer that streams frames to a browser. Optional live
training and controlled training-demo wrappers are included.

## Install and export

The browser site itself needs no Python or NVIDIA GPU. The `hexel-viewer`
Python package is not a standalone wheel: it requires the
[Hexels trainer package](https://github.com/honeycomb-surfels/hexels) (`hexels`),
which is not on PyPI, so install `hexels` from its source checkout first.
Exporting a trusted checkpoint requires Python 3.11, PyTorch and NumPy.

```bash
python -m pip install -e /path/to/hexels
python -m pip install -e /path/to/hexel-viewer
python -m hexel_viewer.viewer_model /path/to/ckpt_030000.pt ./scene-site --site
python -m http.server 8081 --bind 127.0.0.1 --directory scene-site
```

Open `http://127.0.0.1:8081/?model=model.hexview`. Copy the entire exported
site to view on another machine. Its WebGL2 GPU needs enough memory for that
model; a device-independent minimum and real-phone performance are unmeasured.

## CUDA-server and live viewers

```bash
python -m hexel_viewer.viewer --help
python -m hexel_viewer.live_viewer --help
python -m hexel_viewer.training_demo --help
```

Install [hexel-kernel](https://github.com/honeycomb-surfels/hexel-kernel) for
these backends. The tested setup is Linux, Python 3.11, PyTorch 2.5.1 with CUDA 12.1,
CUDA toolkit 12.3, and an RTX A5000 (24 GB). Forward the loopback listener over
SSH when serving from a remote GPU. The server prints its private access URL.
Live viewing is opt-in and adds rendering/encoding overhead while connected.
It does not attach retroactively to another training process.

Both viewers provide navigation, saved cameras and RGB/depth/normal/primitive
inspection. Browser diagnostics and CUDA diagnostics are visualizations, not
interchangeable surface metrics. Display FPS includes the viewer's own
rendering/transport boundary and is separate from the paper benchmark FPS.

## Tests and attribution

`node --test tests/*.mjs` checks browser-side logic and needs Node.js 20.11 or
later. Python tests cover bundle packing, cameras and server behavior; CUDA
checks need an NVIDIA GPU. The browser tests and `scripts/validate_viewers.py`,
which compares browser/CUDA images and exercises navigation and the live server,
use Playwright with Chromium: run `python -m pip install playwright` and then
`python -m playwright install chromium`. Without Playwright the browser tests are
skipped.

Viewer implementation is original Hexels code under [MIT](LICENSE).
Interface ideas were informed by [antimatter15/splat](https://github.com/antimatter15/splat),
[PlayCanvas](https://github.com/playcanvas/engine) and
[Splatter](https://splatter.app/); their source is not bundled here.
