"""Browser-level navigation and comparison checks with CPU-only render fixtures."""

import importlib.util
import threading
import time

import numpy as np
import pytest
import torch

from hexel_viewer.training_demo import DemoServer, TrainingControl
from hexel_viewer.viewer_model import scene_metadata
from hexel_viewer.validation import capture_render_frames


@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Uses isolated viewer-tests browser packages")
def test_comparison_and_wasd_after_training_button(monkeypatch: pytest.MonkeyPatch) -> None:
    from playwright.sync_api import sync_playwright

    rendered = []

    def render(renderer: object, model: object, background: torch.Tensor, payload: dict, degree: int) -> torch.Tensor:
        rendered.append(payload)
        color = [0.9, 0.1, 0.1] if payload["mode"] == "rgb" else [0.1, 0.1, 0.9]
        return torch.tensor(color)[:, None, None].expand(3, payload["height"], payload["width"])

    monkeypatch.setattr("hexel_viewer.viewer.render_frame", render)
    metadata = scene_metadata(np.array([[0., 0., 2.], [1., 1., 3.]]), {}, 0)
    metadata.update(count=2, live=True)
    control = TrainingControl()
    stop = threading.Event()
    with DemoServer(metadata, control, "127.0.0.1", 0) as server:
        def pump() -> None:
            while not stop.is_set():
                server.pump(None, None, torch.zeros(3), 50)
                time.sleep(0.005)

        worker = threading.Thread(target=pump)
        worker.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(args=["--no-sandbox", "--disable-gpu"])
                page = browser.new_page(viewport={"width": 960, "height": 720})
                capture_render_frames(page)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server.server_port}/?backend=cuda#token={server.token}")
                page.wait_for_function("document.getElementById('status').textContent.includes('CUDA connected')")
                assert "2 primitives" in page.locator("#model-details").inner_text()
                assert page.locator("#navigation-hint").is_visible()
                page.click("#training-start")
                before = rendered[-1]["position"]
                page.locator("#viewport").click(position={"x": 800, "y": 400})
                assert page.evaluate("document.activeElement.id") == "viewport"
                page.keyboard.down("w")
                page.wait_for_timeout(700)
                page.keyboard.up("w")
                page.wait_for_timeout(600)
                assert rendered[-1]["position"] != before
                page.check("#compare-enabled")
                page.select_option("#mode", "rgb")
                page.select_option("#compare-mode", "depth")
                page.wait_for_function("Boolean(window.hexelsValidationFrames['rgb/depth'])")
                captured = page.evaluate("window.hexelsValidationFrames['rgb/depth']")
                assert bytes(captured["bytes"]).startswith(b"\x89PNG")
                assert captured["stats"]["iteration"] == 50
                page.wait_for_timeout(1200)
                assert "delivered fps" in page.locator("#stats").inner_text()
                pixels = page.evaluate("""() => {
                    const canvas = document.getElementById('viewport'), context = canvas.getContext('2d');
                    return [0.25, 0.75].map(position => Array.from(context.getImageData(Math.floor(canvas.width * position), 10, 1, 1).data));
                }""")
                assert pixels[0][0] > pixels[0][2]
                assert pixels[1][2] > pixels[1][0]
                page.locator("#comparison-handle").focus()
                page.keyboard.press("ArrowRight")
                assert page.locator("#comparison-position").input_value() == "52"
                for left, right in [("rgb", "normals"), ("primitives", "depth")]:
                    page.select_option("#mode", left)
                    page.select_option("#compare-mode", right)
                    page.wait_for_timeout(700)
                    assert any(payload["mode"] == right for payload in rendered[-6:])
                page.click("#help-toggle")
                assert page.locator("#help-dialog").is_visible()
                page.keyboard.press("Escape")
                assert not page.locator("#help-dialog").is_visible()
                assert not errors
                browser.close()
        finally:
            stop.set()
            worker.join(timeout=5)
