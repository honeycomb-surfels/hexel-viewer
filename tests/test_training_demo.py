import importlib.util
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from hexel_viewer.training_demo import DemoOptions, DemoServer, TrainingControl, recording_plan
from hexel_viewer.viewer import WEB_ROOT, make_camera


def test_training_control_transitions() -> None:
    control = TrainingControl()
    assert control.snapshot()["state"] == "waiting"
    assert control.request("start")["state"] == "running"
    assert control.request("pause")["state"] == "paused"
    assert control.request("resume")["state"] == "running"
    with pytest.raises(ValueError):
        control.request("start")
    control.update(30000, 500000, "completed")
    with pytest.raises(ValueError):
        control.request("resume")


def test_control_api_is_authenticated_and_rejects_commands() -> None:
    control = TrainingControl()
    with DemoServer({"count": 1}, control, "127.0.0.1", 0) as server:
        base = f"http://127.0.0.1:{server.server.server_port}"
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/training")
        assert error.value.code == 401
        headers = {"Authorization": f"Bearer {server.token}", "Content-Type": "application/json"}
        request = Request(base + "/api/training", data=b'{"action":"start"}', headers=headers)
        with urlopen(request) as response:
            assert json.load(response)["state"] == "running"
        for payload in [{"action": "exec"}, {"action": "pause", "command": "anything"}, {"action": []}]:
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + "/api/training", data=json.dumps(payload).encode(), headers=headers))
            assert error.value.code == 400
        with urlopen(base) as response:
            assert b"training_controls.mjs" in response.read()
        assert (WEB_ROOT.parent / "viewer.py").is_file()
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/../viewer.py")
        assert error.value.code == 404


def test_demo_options_are_opt_in(tmp_path: Path) -> None:
    options = DemoOptions().parse_args(["--source_path", str(tmp_path), "--model_path", str(tmp_path / "model"),
                                      "--iterations", "100", "--record_frames", "24", "--skip_postprocess"])
    assert options.viewer_port == 0
    assert not options.demo_autostart
    assert options.record_frames == 24


def test_recording_cameras_and_iterations_are_deterministic() -> None:
    import torch

    cameras = []
    for position in ([3, 0, 1], [0, 3, 1], [-3, 0, 1], [0, -3, 1]):
        camera = make_camera({"position": position, "forward": [-float(coordinate) for coordinate in position],
                              "up": [0, 0, 1], "width": 640, "height": 360, "fov": 60}, torch.device("cpu"))
        camera.image_name = str(len(cameras))
        cameras.append(camera)
    first = recording_plan(cameras, 30000, 1440, 640, 360)
    assert first == recording_plan(cameras, 30000, 1440, 640, 360)
    assert first[0]["iteration"] == 0 and first[-1]["iteration"] == 30000
    assert len({row["iteration"] for row in first}) == 1440
    assert first[0]["position"] != first[100]["position"]
    with pytest.raises(ValueError):
        recording_plan(cameras, 5, 7, 640, 360)


@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Browser checks use the isolated viewer-tests packages")
def test_browser_training_buttons_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import time

    import numpy as np
    import torch
    from playwright.sync_api import sync_playwright

    from hexel_viewer.viewer_model import scene_metadata

    monkeypatch.setattr("hexel_viewer.viewer.render_frame", lambda *args: torch.zeros(3, 64, 64))
    metadata = scene_metadata(np.array([[0., 0., 2.], [1., 1., 3.]]), {}, 0)
    metadata.update(count=2, live=True)
    control = TrainingControl()
    stop = threading.Event()
    with DemoServer(metadata, control, "127.0.0.1", 0) as server:
        def pump() -> None:
            while not stop.is_set():
                server.pump(None, None, torch.zeros(3), 0)
                time.sleep(0.01)

        worker = threading.Thread(target=pump)
        worker.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(args=["--no-sandbox", "--disable-gpu"])
                page = browser.new_page(viewport={"width": 960, "height": 720})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server.server_port}/?backend=cuda#token={server.token}")
                page.wait_for_function("document.getElementById('training-state').textContent.includes('waiting')")
                page.click("#training-start")
                page.wait_for_function("document.getElementById('training-state').textContent.includes('running')")
                assert control.snapshot()["state"] == "running"
                page.click("#training-pause")
                page.wait_for_function("document.getElementById('training-state').textContent.includes('paused')")
                assert control.snapshot()["state"] == "paused"
                page.click("#training-pause")
                page.wait_for_function("document.getElementById('training-state').textContent.includes('running')")
                assert not errors
                browser.close()
        finally:
            stop.set()
            worker.join(timeout=5)
