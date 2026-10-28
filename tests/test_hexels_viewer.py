from __future__ import annotations

import json
import io
import struct
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from hexels.model import Hexels3DGeometry, Hexels3DModel
from hexel_viewer.viewer import FrameRequest, ViewerServer, make_camera, validate_request
from hexel_viewer.viewer_model import MAGIC, bundle_arrays, colmap_viewpoints, load_viewer_model, write_bundle


def make_model(basis: int = 3) -> Hexels3DModel:
    return Hexels3DModel(
        Hexels3DGeometry(torch.tensor([[0., 0., 3.], [0.2, 0.1, 4.]]),
                         torch.full((2, 2), 0.4), torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)),
        torch.tensor([[0.3, 0.4, 0.8], [0.7, 0.2, 0.1]]), 0.8, 10., view_basis_dim=basis,
    )


def camera_request() -> dict:
    return {"position": [0, 0, 0], "forward": [0, 0, 1], "up": [0, -1, 0],
            "width": 64, "height": 48, "mode": "rgb", "fov": 60}


@pytest.mark.parametrize("view,basis", [("none", 0), ("linear", 3), ("sh3", 15)])
def test_bundle_layout_and_appearance(tmp_path: Path, view: str, basis: int) -> None:
    model = make_model(15 if basis == 15 else 3)
    with torch.no_grad():
        model.view_grads.copy_(torch.arange(model.view_grads.numel()).reshape_as(model.view_grads))
    config = {"color_model": "linear", "view_model": view, "use_view_grads": True}
    metadata, rows = bundle_arrays(model, config)
    assert rows.shape == (2, 6 + basis, 4)
    np.testing.assert_allclose(rows[:, 1, :3], [[0.4, 0, 0]] * 2)
    np.testing.assert_allclose(rows[:, 2, :3], [[0, 0.4, 0]] * 2)
    if basis == 3:
        np.testing.assert_array_equal(rows[:, 6:, :3], model.view_grads.detach().numpy().transpose(0, 2, 1))
    path = tmp_path / "model.hexview"
    write_bundle(path, metadata, rows)
    payload = path.read_bytes()
    assert payload[:8] == MAGIC
    length, = struct.unpack("<I", payload[8:12])
    assert length % 4 == 0
    header = json.loads(payload[12:12 + length])
    assert header["count"] == 2
    np.testing.assert_array_equal(np.frombuffer(payload[12 + length:], dtype="<f4").reshape(rows.shape), rows)
    with pytest.raises(FileExistsError):
        write_bundle(path, metadata, rows)


def test_export_active_rows_and_anisotropic_softness() -> None:
    model = make_model()
    model.active_mask[0] = False
    model.log_softness = torch.nn.Parameter(torch.tensor([[2., 3., 4.], [3., 4., 5.]]))
    metadata, rows = bundle_arrays(model, {"color_model": "linear", "view_model": "none"})
    assert metadata["count"] == 1
    np.testing.assert_allclose(rows[0, 1:4, 3], np.exp([3., 4., 5.]), rtol=1e-6)


def test_export_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="audited"):
        bundle_arrays(make_model(), {"color_model": "quadratic", "view_model": "none"})
    model = make_model()
    model.allocate_children(1)
    with pytest.raises(ValueError, match="children"):
        bundle_arrays(model, {"color_model": "linear", "view_model": "none"})


def test_checkpoint_load_needs_no_dataset_or_cuda(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints/model.pt"
    checkpoint.parent.mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"color_model": "linear", "view_model": "none"}))
    torch.save({"model": make_model().state_dict(), "iteration": 200}, checkpoint)
    model, config, iteration = load_viewer_model(checkpoint)
    assert model.means3d.device.type == "cpu"
    assert iteration == 200
    assert bundle_arrays(model, config)[0]["count"] == 2


def test_camera_contract() -> None:
    payload = validate_request(camera_request())
    camera = make_camera(payload, torch.device("cpu"))
    torch.testing.assert_close(camera.world_view_transform, torch.eye(4))
    torch.testing.assert_close(camera.full_proj_transform, camera.projection_matrix)
    assert camera.image_width == 64
    assert camera.FoVy == pytest.approx(np.pi / 3)


@pytest.mark.parametrize("changes", [{"width": 100000}, {"width": True}, {"mode": "train"},
                                     {"forward": [0, 0, 0]}, {"up": [0, 0, 1]}, {"fov": float("nan")},
                                     {"position": [float("inf"), 0, 0]}, {"depth_scale": -1}])
def test_network_camera_validation(changes: dict) -> None:
    with pytest.raises(ValueError):
        validate_request({**camera_request(), **changes})


def test_comparison_request_budget() -> None:
    payload = {**camera_request(), "compare_mode": "depth"}
    assert validate_request(payload)["compare_mode"] == "depth"
    with pytest.raises(ValueError, match="combined pixel budget"):
        validate_request(payload, max_pixels=64 * 48)
    with pytest.raises(ValueError, match="comparison mode"):
        validate_request({**payload, "compare_mode": "invalid"})


def test_comparison_renders_one_camera_at_one_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def render(renderer: object, model: object, background: torch.Tensor, payload: dict, degree: int) -> torch.Tensor:
        calls.append(payload)
        return torch.full((3, payload["height"], payload["width"]), 0.25 if payload["mode"] == "rgb" else 0.75)

    monkeypatch.setattr("hexel_viewer.viewer.render_frame", render)
    with ViewerServer({"count": 2}, port=0) as viewer:
        request = FrameRequest(validate_request({**camera_request(), "compare_mode": "depth"}))
        viewer.requests.put(request)
        assert viewer.pump(None, None, torch.zeros(3), 123)
        assert request.completed.is_set()
        assert request.metadata["iteration"] == 123
        assert request.metadata["compare_mode"] == "depth"
        assert calls[0]["position"] == calls[1]["position"]
        assert calls[0]["forward"] == calls[1]["forward"]
        assert [call["mode"] for call in calls] == ["rgb", "depth"]
        pixels = np.asarray(Image.open(io.BytesIO(request.image)))
        assert pixels.shape == (48, 128, 3)
        assert (pixels[:, :64] == 64).all()
        assert (pixels[:, 64:] == 191).all()


def test_authenticated_server_and_main_thread_pump(monkeypatch) -> None:
    monkeypatch.setattr("hexel_viewer.viewer.render_frame", lambda *args: torch.ones((3, 48, 64)))
    with ViewerServer({"count": 2}, port=0) as viewer:
        root = f"http://127.0.0.1:{viewer.server.server_port}"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{root}/api/info")
        assert error.value.code == 401
        headers = {"Authorization": f"Bearer {viewer.token}", "Content-Type": "application/json"}
        request = urllib.request.Request(f"{root}/api/render", data=json.dumps(camera_request()).encode(), headers=headers)
        result = []
        thread = threading.Thread(target=lambda: result.append(urllib.request.urlopen(request).read()))
        thread.start()
        pending = viewer.requests.get(timeout=5)
        viewer.requests.put(pending)
        assert viewer.pump(None, None, torch.zeros(3), 50)
        thread.join(timeout=5)
        assert result[0].startswith(b"\x89PNG")
        assert viewer.metadata["iteration"] == 50
        assert not viewer.pump(None, None, torch.zeros(3), 51)


def simulated_out_of_memory(*args: object) -> torch.Tensor:
    raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")


def test_failed_frame_reaches_client_and_serving_continues(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr("hexel_viewer.viewer.render_frame", simulated_out_of_memory)
    with ViewerServer({"count": 2}, port=0) as viewer:
        root = f"http://127.0.0.1:{viewer.server.server_port}"
        headers = {"Authorization": f"Bearer {viewer.token}", "Content-Type": "application/json"}
        request = urllib.request.Request(f"{root}/api/render", data=json.dumps(camera_request()).encode(), headers=headers)
        replies = []

        def fetch() -> None:
            try:
                urllib.request.urlopen(request)
            except urllib.error.HTTPError as error:
                replies.append((error.code, json.loads(error.read())))

        thread = threading.Thread(target=fetch)
        thread.start()
        viewer.requests.put(viewer.requests.get(timeout=5))
        assert not viewer.pump(None, None, torch.zeros(3), 7)
        thread.join(timeout=5)
        assert replies[0][0] == 503
        assert "OutOfMemoryError" in replies[0][1]["error"]
        monkeypatch.setattr("hexel_viewer.viewer.render_frame", lambda *args: torch.ones((3, 48, 64)))
        retry = FrameRequest(validate_request(camera_request()))
        viewer.requests.put(retry)
        assert viewer.pump(None, None, torch.zeros(3), 8)
        assert retry.error is None and retry.image.startswith(b"\x89PNG")
    assert "CUDA viewer frame failed" in caplog.text


def test_failed_frame_does_not_stop_live_training(monkeypatch: pytest.MonkeyPatch) -> None:
    from hexel_viewer.viewer import TrainingViewer
    monkeypatch.setattr("hexel_viewer.viewer.render_frame", simulated_out_of_memory)
    model = make_model()
    camera = make_camera(validate_request(camera_request()), torch.device("cpu"))
    options = SimpleNamespace(viewer_port=0, viewer_host="127.0.0.1", viewer_every=1, viewer_max_fps=2)
    trainer = SimpleNamespace(opt=options, model=model, renderer=None, bg_color=torch.zeros(3), start_iteration=0,
                              train_cameras=[camera], _render_model=lambda: model)
    with TrainingViewer(trainer) as observer:
        request = FrameRequest(validate_request(camera_request()))
        observer.server.requests.put(request)
        observer.step(1)
        assert request.completed.is_set() and "OutOfMemoryError" in request.error
        assert observer.frames == 0


def test_colmap_pose_export_without_images(tmp_path: Path) -> None:
    path = tmp_path / "sparse/0/images.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(struct.pack("<Q", 1) + struct.pack("<idddddddi", 1, 1, 0, 0, 0, 0, 0, -3, 1)
                     + b"view.png\0" + struct.pack("<Q", 0))
    camera, = colmap_viewpoints(tmp_path)
    assert camera["position"] == [0., 0., 3.]
    assert camera["forward"] == [0., 0., 1.]
    path.with_name("cameras.bin").write_bytes(struct.pack("<QiiQQdddd", 1, 1, 1, 640, 480, 600, 600, 320, 240))
    camera, = colmap_viewpoints(tmp_path)
    assert camera["fov"] == pytest.approx(np.degrees(2 * np.arctan(0.4)))


def test_export_source_path_override_and_missing_pose_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              caplog: pytest.LogCaptureFixture) -> None:
    from hexel_viewer import viewer_model
    checkpoint = tmp_path / "run/checkpoints/ckpt_000100.pt"
    checkpoint.parent.mkdir(parents=True)
    (tmp_path / "run/config.json").write_text(json.dumps({"color_model": "linear", "view_model": "none",
                                                          "source_path": "data/dtu/scan24"}))
    torch.save({"model": make_model().state_dict(), "iteration": 100}, checkpoint)
    poses = tmp_path / "scene/sparse/0/images.bin"
    poses.parent.mkdir(parents=True)
    poses.write_bytes(struct.pack("<Q", 1) + struct.pack("<idddddddi", 1, 1, 0, 0, 0, 0, 0, -3, 1)
                      + b"view.png\0" + struct.pack("<Q", 0))
    monkeypatch.chdir(tmp_path)

    def exported_cameras(output: str, *options: str) -> list[dict]:
        monkeypatch.setattr("sys.argv", ["viewer_model", str(checkpoint), output, *options])
        viewer_model.main()
        payload = (tmp_path / output).read_bytes()
        length, = struct.unpack("<I", payload[8:12])
        return json.loads(payload[12:12 + length])["cameras"]

    assert exported_cameras("default.hexview") == []
    assert "No COLMAP poses at" in caplog.text and "data/dtu/scan24/sparse/0/images.bin" in caplog.text
    cameras = exported_cameras("override.hexview", "--source-path", str(tmp_path / "scene"))
    assert [camera["name"] for camera in cameras] == ["view.png"]


def test_live_observer_is_separate_from_benchmark_trainer() -> None:
    from hexel_viewer.live_viewer import LiveTrainer
    from hexels.trainer import Hexels3DTrainer
    assert "viewer" not in Hexels3DTrainer.train.__code__.co_names
    assert LiveTrainer._update_sh_degree is not Hexels3DTrainer._update_sh_degree


def test_live_options_keep_standard_training_arguments(tmp_path: Path) -> None:
    from hexel_viewer.live_viewer import LiveOptions
    options = LiveOptions().parse_args(["--source_path", str(tmp_path), "--model_path", str(tmp_path / "run"),
                                       "--iterations", "100", "--viewer_every", "20", "--skip_postprocess"])
    assert options.viewer_every == 20
    assert options.viewer_port == 8080
    assert options.skip_postprocess


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_live_cuda_observer_does_not_mutate_training() -> None:
    from hexel_viewer.viewer import TrainingViewer
    from hexels.renderer import Hexels3DRenderer
    options = SimpleNamespace(softness=10., scale_modifier=1., color_model="linear", view_model="none",
                              use_view_grads=False, debug=False, viewer_port=0, viewer_host="127.0.0.1",
                              viewer_every=10, viewer_max_fps=2)
    model = make_model().cuda()
    model.train()
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    payload = validate_request(camera_request())
    camera = make_camera(payload, torch.device("cuda"))
    trainer = SimpleNamespace(opt=options, model=model, renderer=Hexels3DRenderer(options),
                              bg_color=torch.zeros(3, device="cuda"), start_iteration=0,
                              train_cameras=[camera], _render_model=lambda: model)
    with TrainingViewer(trainer) as viewer:
        request = FrameRequest(payload)
        viewer.server.requests.put(request)
        viewer.step(9)
        assert not request.completed.is_set()
        viewer.step(10)
        assert request.completed.is_set() and request.image.startswith(b"\x89PNG")
        assert viewer.frames == 1
        assert model.training
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_viewer_modes_match_reference() -> None:
    from hexels.renderer import Hexels3DRenderer
    from hexel_viewer.viewer import MODES, render_frame
    config = SimpleNamespace(softness=10., scale_modifier=1., color_model="linear", view_model="none",
                             use_view_grads=False, debug=False)
    renderer = Hexels3DRenderer(config)
    model = make_model().cuda()
    background = torch.zeros(3, device="cuda")
    payload = validate_request(camera_request())
    camera = make_camera(payload, background.device)
    with torch.no_grad():
        expected = renderer.render(camera, model, background)["image"]
        torch.testing.assert_close(render_frame(renderer, model, background, payload), expected, rtol=0, atol=0)
        for mode in MODES:
            image = render_frame(renderer, model, background, dict(payload, mode=mode))
            assert image.shape == (3, 48, 64)
            assert torch.isfinite(image).all()
