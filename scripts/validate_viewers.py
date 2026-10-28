"""Batch-only CUDA/browser parity and UI checks on synthetic or real checkpoints."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import re
import socket
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image

from hexels.model import Hexels3DGeometry, Hexels3DModel
from hexel_viewer.viewer import render_frame, validate_request
from hexel_viewer.viewer_model import bundle_arrays, colmap_viewpoints, load_viewer_model, write_bundle

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def prepare(output: Path, checkpoints: list[Path]) -> None:
    """Save CUDA references and the exact exported inputs used by the browser."""
    from hexels.renderer import Hexels3DRenderer
    output.mkdir(parents=True, exist_ok=False)
    import shutil
    from hexel_viewer.viewer import WEB_ROOT
    shutil.copytree(WEB_ROOT, output / "web")
    cases = []
    for view_model in ("none", "linear", "sh3"):
        for prefilter in (False, True):
            generator = torch.Generator().manual_seed(123)
            means = torch.randn((24, 3), generator=generator) * 0.4
            means[:, 2] += 3
            rotations = torch.randn((24, 4), generator=generator)
            model = Hexels3DModel(Hexels3DGeometry(means, torch.full((24, 2), 0.25), rotations),
                                  torch.rand((24, 3), generator=generator), 0.85, 8.,
                                  view_basis_dim=15 if view_model == "sh3" else 3)
            with torch.no_grad():
                model.color_grads.copy_(torch.randn(model.color_grads.shape, generator=generator) * 0.15)
                model.view_grads.copy_(torch.randn(model.view_grads.shape, generator=generator) * 0.1)
            config = dict(softness=8., softness_mode="iso", scale_modifier=1., color_model="linear",
                          view_model=view_model, use_view_grads=view_model != "none", debug=False,
                          prefilter_edge_on=prefilter, white_background=False)
            camera = dict(position=[0, 0, 0], forward=[0, 0, 1], up=[0, -1, 0], fov=60,
                          width=320, height=240, mode="rgb", depth_scale=10)
            cases.append((f"synthetic_{view_model}_{int(prefilter)}", model, config, 0, camera))
    angle = math.pi / 4
    model = Hexels3DModel(
        Hexels3DGeometry(torch.tensor([[0.03, -0.02, 1.]]), torch.tensor([[1.4, 0.7]]),
                         torch.tensor([[math.cos(angle / 2), 0., -math.sin(angle / 2), 0.]])),
        torch.tensor([[0.8, 0.5, 0.2]]), 0.8, 10., view_basis_dim=3,
    )
    config = dict(softness=10., softness_mode="iso", scale_modifier=1., color_model="linear",
                  view_model="none", use_view_grads=False, debug=False, prefilter_edge_on=False,
                  conservative_bounds=True, white_background=False)
    camera = dict(position=[0, 0, 0], forward=[0, 0, 1], up=[0, -1, 0], fov=90,
                  width=320, height=240, mode="rgb", depth_scale=10)
    cases.append(("near_plane_corrected", model, config, 0, camera))
    for index, checkpoint in enumerate(checkpoints):
        model, config, iteration = load_viewer_model(checkpoint)
        viewpoints = colmap_viewpoints(Path(config["source_path"]))
        if not viewpoints:
            raise ValueError(f"Real checkpoint validation needs registered poses: {checkpoint}")
        camera = dict(viewpoints[0], width=640, height=480, mode="rgb", depth_scale=10)
        cases.append((f"checkpoint_{index}_{config['view_model']}", model, config, iteration, camera))
    manifest = []
    for name, model, config, iteration, camera in cases:
        metadata, rows = bundle_arrays(model, config, iteration)
        metadata["cameras"] = [dict(camera, name="Validation camera")]
        metadata["name"] = name
        write_bundle(output / f"{name}.hexview", metadata, rows)
        model = model.cuda()
        renderer = Hexels3DRenderer(SimpleNamespace(**config))
        background = torch.tensor(metadata["background"], device="cuda", dtype=torch.float32)
        payload = validate_request(camera)
        render_frame(renderer, model, background, payload)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            image = render_frame(renderer, model, background, payload)
        torch.cuda.synchronize()
        milliseconds = (time.perf_counter() - start) * 100
        rgb = image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        np.save(output / f"{name}_cuda.npy", rgb)
        Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(output / f"{name}_cuda.png")
        manifest.append(dict(name=name, camera=payload, count=metadata["count"], cuda_frame_ms=milliseconds,
                             bundle_bytes=(output / f"{name}.hexview").stat().st_size))
        del model, renderer
    (output / "cases.json").write_text(json.dumps(manifest, indent=2))


def browser_checks(output: Path) -> None:
    """Chromium is software-rendered here; measurements are not mobile GPU FPS."""
    from playwright.sync_api import sync_playwright
    output = output.resolve()
    relative = Path(".")
    errors = []
    results = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(output)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            page = browser.new_page(viewport={"width": 960, "height": 720})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"{base}/web/index.html")
            for case in json.loads((output / "cases.json").read_text()):
                result = page.evaluate("""async ({url, camera}) => {
                    const {parseBundle, sortDepths, viewMatrix} = await import('./core.mjs');
                    const {HexelRenderer} = await import('./renderer.mjs');
                    const {metadata, rows} = parseBundle(await (await fetch(url)).arrayBuffer());
                    const canvas = document.createElement('canvas');
                    const renderer = new HexelRenderer(canvas);
                    renderer.load(metadata, rows);
                    const positions = new Float32Array(metadata.count * 3);
                    for (let index=0; index<metadata.count; index++) positions.set(rows.subarray(index*metadata.texels*4, index*metadata.texels*4+3), index*3);
                    renderer.setOrder(sortDepths(positions, camera.forward, camera.position));
                    renderer.resize(camera.width, camera.height);
                    const pose = {...camera, view: viewMatrix(camera.position, camera.forward, camera.up)};
                    renderer.render(pose, 0, 10);
                    const pixels = canvas.toDataURL('image/png');
                    const gl = renderer.gl;
                    const read = () => {
                        const values = new Uint8Array(camera.width * camera.height * 4);
                        gl.readPixels(0, 0, camera.width, camera.height, gl.RGBA, gl.UNSIGNED_BYTE, values);
                        return values;
                    };
                    const rgb = read();
                    renderer.render(pose, 3, 10);
                    const depth = read();
                    renderer.render(pose, 0, 10);
                    const split = Math.floor(camera.width * 0.4);
                    renderer.render(pose, 3, 10, [split, camera.width - split]);
                    const paired = read();
                    let comparisonError = 0;
                    for (let index = 0; index < paired.length; index++) {
                        const reference = Math.floor(index / 4) % camera.width < split ? rgb : depth;
                        comparisonError = Math.max(comparisonError, Math.abs(paired[index] - reference[index]));
                    }
                    for (let mode=1; mode<6; mode++) renderer.render(pose, mode, 10);
                    const error = gl.getError();
                    const debug = gl.getExtension('WEBGL_debug_renderer_info');
                    const result = {pixels, error, comparisonError, float32: renderer.float32Blend, renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)};
                    gl.getExtension('WEBGL_lose_context').loseContext();
                    return result;
                }""", {"url": f"{base}/{relative}/{case['name']}.hexview", "camera": case["camera"]})
                if result["error"]:
                    raise RuntimeError(f"WebGL error: {result['error']}")
                if result["comparisonError"]:
                    raise AssertionError(f"Comparison changed the camera or pixels: {result['comparisonError']}")
                png = base64.b64decode(result.pop("pixels").split(",", 1)[1])
                (output / f"{case['name']}_web.png").write_bytes(png)
                rendered = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float64) / 255
                reference = np.load(output / f"{case['name']}_cuda.npy")
                difference = rendered - reference
                mse = float(np.mean(difference**2))
                result.update(case, mae=float(np.mean(np.abs(difference))), max_error=float(np.max(np.abs(difference))),
                              psnr_db=-10 * np.log10(max(mse, 1e-20)))
                results.append(result)
                LOGGER.info("%s: MAE %.6f, PSNR %.2f dB", case["name"], result["mae"], result["psnr_db"])
            first = json.loads((output / "cases.json").read_text())[0]
            page.goto(f"{base}/web/index.html?scale=0.5&model=/{relative}/{first['name']}.hexview")
            page.wait_for_function("document.querySelector('#status').textContent.startsWith('Loaded')")
            page.wait_for_function("!document.querySelector('#stats').textContent.includes('No model')")
            for mode in ("wireframe", "primitives", "depth", "normals", "opacity", "rgb"):
                page.select_option("#mode", mode)
                page.wait_for_timeout(100)
            page.locator("#viewport").click(position={"x": 700, "y": 350})
            page.keyboard.down("w"); page.wait_for_timeout(150); page.keyboard.up("w")
            page.click("#reset")
            page.screenshot(path=str(output / "desktop_ui.png"))
            mobile = browser.new_context(viewport={"width": 393, "height": 852}, device_scale_factor=2, is_mobile=True, has_touch=True)
            mobile_page = mobile.new_page()
            mobile_page.on("pageerror", lambda error: errors.append(str(error)))
            mobile_page.goto(f"{base}/web/index.html?model=/{relative}/{first['name']}.hexview")
            mobile_page.wait_for_function("document.querySelector('#status').textContent.startsWith('Loaded')")
            mobile_page.wait_for_function("!document.querySelector('#stats').textContent.includes('No model')")
            assert mobile_page.locator("#panel").is_hidden()
            assert mobile_page.locator("#touch-pad").is_visible()
            mobile_page.locator("[data-key='KeyW']").tap()
            mobile_page.click("#toggle-panel")
            assert mobile_page.locator("#touch-pad").is_hidden()
            mobile_page.screenshot(path=str(output / "mobile_ui.png"))
            mobile.close()
            browser.close()
    finally:
        server.shutdown(); server.server_close(); thread.join()
    (output / "browser_results.json").write_text(json.dumps({"results": results, "page_errors": errors}, indent=2))
    if errors or any(result["mae"] > 0.01 for result in results):
        raise AssertionError(f"Viewer validation failed: {errors}; inspect browser_results.json")


def live_training_check(output: Path, source: Path) -> None:
    """Exercise the real live-training entry, authenticated transport and shutdown."""
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [sys.executable, "-m", "hexel_viewer.live_viewer", "--source_path", str(source),
               "--model_path", str(output / "training"), "--resolution", "8", "--iterations", "200",
               "--viewer_port", str(port), "--viewer_every", "1", "--viewer_max_fps", "10",
               "--render_every", "0", "--save_every", "200", "--log_every", "200",
               "--color_model", "linear", "--view_model", "none", "--use_view_grads", "False",
               "--seed", "0", "--skip_postprocess"]
    log_path = output / "training.log"
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            match = None
            while time.monotonic() < deadline and process.poll() is None:
                match = re.search(r"#token=([A-Za-z0-9_-]+)", log_path.read_text())
                if match:
                    break
                time.sleep(0.05)
            if not match:
                raise RuntimeError(f"Live viewer did not start; inspect {log_path}")
            headers = {"Authorization": f"Bearer {match.group(1)}", "Content-Type": "application/json"}
            base = f"http://127.0.0.1:{port}"
            with urlopen(Request(f"{base}/api/info", headers=headers), timeout=10) as response:
                info = json.load(response)
            payload = dict(info["cameras"][0], width=160, height=120, mode="rgb", depth_scale=10)
            request = Request(f"{base}/api/render", headers=headers, data=json.dumps(payload).encode())
            with urlopen(request, timeout=35) as response:
                image = response.read()
                stats = json.loads(response.headers["X-Hexels-Stats"])
            if not image.startswith(b"\x89PNG"):
                raise AssertionError("Live viewer did not return PNG")
            (output / "live_frame.png").write_bytes(image)
            if process.wait(timeout=180) != 0:
                raise RuntimeError(f"Live training failed; inspect {log_path}")
            summary = json.loads((output / "training/viewer_summary.json").read_text())
            if summary["frames"] < 1:
                raise AssertionError("Training did not service the viewer request")
            (output / "live_results.json").write_text(json.dumps({"frame": stats, "observer": summary,
                                                                   "label": "200-step plumbing check, not quality"}, indent=2))
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--browser-only", action="store_true")
    parser.add_argument("--live-source", type=Path, help="Run only a 200-step live-training plumbing check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.live_source:
        if not torch.cuda.is_available():
            raise RuntimeError("Live smoke requires an NVIDIA GPU with CUDA")
        live_training_check(args.output, args.live_source)
        return
    if not args.browser_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/browser validation requires an NVIDIA GPU with CUDA; --browser-only rechecks a prepared output")
        prepare(args.output, args.checkpoint)
    browser_checks(args.output)


if __name__ == "__main__":
    main()
