"""CPU browser checks for loading UX; the renderer is explicitly a test stub."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import struct
import threading

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def page() -> Iterator[playwright_api.Page]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(ROOT / "hexel_viewer/web")))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with playwright_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=["--no-sandbox", "--disable-gpu"])
            context = browser.new_context(viewport={"width": 1280, "height": 850})
            page = context.new_page()
            page.route("**/renderer.mjs", lambda route: route.fulfill(content_type="text/javascript", body="""
                export class HexelRenderer {
                    constructor(canvas) { this.canvas = canvas; this.floatTarget = true; this.visible = 0; }
                    load() {} setOrder(order) { this.visible = order.length; }
                    resize(width, height) { this.canvas.width = width; this.canvas.height = height; }
                    render() {}
                }
            """))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            page.wait_for_function("document.getElementById('file').onchange !== null")
            yield page
            context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def bundle() -> bytes:
    metadata = {"format": "hexels.viewer", "version": 1, "count": 1, "texels": 6,
                "basis_count": 0, "view_model": "none", "name": "Loading fixture",
                "center": [0, 0, 3], "radius": 1, "cameras": []}
    header = json.dumps(metadata).encode()
    header += b" " * (-len(header) % 4)
    values = [0.0] * 24
    values[2] = 3.0
    return b"HEXVIEW1" + struct.pack("<I", len(header)) + header + struct.pack("<24f", *values)


def test_theme_and_file_loading_rotation(page: playwright_api.Page, tmp_path: Path) -> None:
    assert page.locator("#welcome h2").inner_text() == "Interactive surface reconstruction"
    assert page.evaluate("getComputedStyle(document.documentElement).backgroundColor") == "rgb(25, 22, 21)"
    page.screenshot(path=str(tmp_path / "welcome.png"))
    page.evaluate("""() => {
        const original = File.prototype.arrayBuffer;
        File.prototype.arrayBuffer = async function() {
            await new Promise(resolve => { window.finishFileRead = resolve; });
            return original.call(this);
        };
    }""")
    page.locator("#file").set_input_files({"name": "fixture.hexview", "mimeType": "application/octet-stream", "buffer": bundle()})
    page.wait_for_function("Boolean(window.finishFileRead)")
    assert page.locator("#loading-notice").is_visible()
    assert page.locator("#viewport").get_attribute("aria-busy") == "true"
    assert "Reading fixture.hexview" in page.locator("#status").inner_text()
    assert page.locator("#load-url").is_disabled()
    page.screenshot(path=str(tmp_path / "loading.png"))
    page.wait_for_function("document.querySelector('.loading-message').textContent.includes('Rome')", timeout=7000)
    page.evaluate("window.finishFileRead()")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('Loaded')")
    assert not page.locator("#loading-notice").is_visible()
    assert page.locator("#viewport").get_attribute("aria-busy") == "false"
    assert page.locator("#scene-name").inner_text() == "Loading fixture"
    assert not page.locator("#file").is_disabled()


def test_url_error_cleanup_retry_and_reduced_motion(page: playwright_api.Page, tmp_path: Path) -> None:
    page.emulate_media(reduced_motion="reduce")
    page.set_viewport_size({"width": 390, "height": 844})
    page.evaluate("""() => {
        const original = window.fetch;
        window.fetch = async (...args) => {
            if (String(args[0]).includes('fixture.hexview')) {
                await new Promise(resolve => { window.finishDownload = resolve; });
            }
            return original(...args);
        };
    }""")
    page.route("**/fixture.hexview", lambda route: route.fulfill(status=503, body="Unavailable"))
    page.locator("#model-url").fill(f"{page.url}fixture.hexview")
    page.locator("#load-url").click()
    page.wait_for_function("Boolean(window.finishDownload)")
    assert page.locator("#loading-notice").is_visible()
    assert page.evaluate("getComputedStyle(document.querySelector('.loading-mark')).animationName") == "none"
    rectangle = page.locator("#loading-notice").bounding_box()
    assert rectangle["x"] >= 0 and rectangle["x"] + rectangle["width"] <= 390
    page.screenshot(path=str(tmp_path / "loading_mobile.png"))
    page.evaluate("window.finishDownload()")
    page.wait_for_function("document.getElementById('status').textContent.includes('503')")
    assert not page.locator("#loading-notice").is_visible()
    assert page.locator("#welcome").is_visible()
    assert not page.locator("#load-url").is_disabled()
    page.locator("#file").set_input_files({"name": "bad.hexview", "mimeType": "application/octet-stream", "buffer": b"invalid"})
    page.wait_for_function("document.getElementById('status').textContent.includes('Expected a .hexview')")
    assert not page.locator("#loading-notice").is_visible()
    page.locator("#file").set_input_files({"name": "valid.hexview", "mimeType": "application/octet-stream", "buffer": bundle()})
    page.wait_for_function("document.getElementById('status').textContent.startsWith('Loaded')")
    assert not page.locator("#loading-notice").is_visible()
