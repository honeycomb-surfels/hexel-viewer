"""Read-only CUDA viewer, with a loopback HTTP UI and bounded live requests."""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import queue
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np
import torch
from PIL import Image

from .viewer_model import colmap_viewpoints, load_viewer_model, scene_metadata

WEB_ROOT = Path(__file__).resolve().parent / "web"
LOGGER = logging.getLogger(__name__)
MODES = ("rgb", "primitives", "wireframe", "depth", "normals", "opacity", "overdraw")


def validate_request(payload: dict, max_pixels: int = 1920 * 1080) -> dict:
    """Validate network input before GPU allocation; clients cannot change training."""
    width, height = payload.get("width"), payload.get("height")
    if type(width) is not int or type(height) is not int or min(width, height) < 16 or max(width, height) > 4096 or width * height > max_pixels:
        raise ValueError(f"Viewport must be 16..4096 per side and at most {max_pixels} pixels")
    if payload.get("mode") not in MODES:
        raise ValueError("Unsupported viewing mode")
    compare_mode = payload.get("compare_mode")
    if compare_mode is not None and compare_mode not in MODES:
        raise ValueError("Unsupported comparison mode")
    if compare_mode is not None and 2 * width * height > max_pixels:
        raise ValueError("Comparison view exceeds the combined pixel budget")
    vectors = [np.asarray(payload.get(name), dtype=np.float64) for name in ("position", "forward", "up")]
    if any(vector.shape != (3,) or not np.isfinite(vector).all() for vector in vectors):
        raise ValueError("Camera requires finite three-element position, forward and up")
    position, forward, up = vectors
    forward_length, up_length = np.linalg.norm(forward), np.linalg.norm(up)
    if min(forward_length, up_length) < 1e-6:
        raise ValueError("Camera axes cannot be zero")
    forward, up = forward / forward_length, up / up_length
    if abs(np.dot(forward, up)) > 0.999:
        raise ValueError("Camera axes are parallel")
    fov, depth_scale = float(payload.get("fov", 60)), float(payload.get("depth_scale", 10))
    if not 10 <= fov <= 140 or not math.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("Invalid field of view or depth scale")
    return dict(width=width, height=height, mode=payload["mode"], compare_mode=compare_mode, position=position.tolist(),
                forward=forward.tolist(), up=up.tolist(), fov=fov, depth_scale=depth_scale)


def make_camera(payload: dict, device: torch.device) -> SimpleNamespace:
    """Convert common +Z-forward camera into the renderer's transposed matrices."""
    position = np.asarray(payload["position"])
    forward = np.asarray(payload["forward"])
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, payload["up"])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = np.stack([right, down, forward])
    view[:3, 3] = -view[:3, :3] @ position
    fovy = math.radians(payload["fov"])
    tan_y = math.tan(fovy / 2)
    tan_x = tan_y * payload["width"] / payload["height"]
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0], projection[1, 1] = 1 / tan_x, 1 / tan_y
    projection[2, 2], projection[2, 3], projection[3, 2] = 100 / 99.8, -20 / 99.8, 1
    view_tensor = torch.tensor(view.T.copy(), device=device)
    projection_tensor = torch.tensor(projection.T.copy(), device=device)
    return SimpleNamespace(
        image_width=payload["width"], image_height=payload["height"],
        FoVx=2 * math.atan(tan_x), FoVy=fovy, world_view_transform=view_tensor,
        projection_matrix=projection_tensor, full_proj_transform=view_tensor @ projection_tensor,
        camera_center=torch.tensor(position, device=device, dtype=torch.float32),
    )


@torch.no_grad()
def render_frame(renderer: object, model: object, background: torch.Tensor, payload: dict, sh_degree: int = 3) -> torch.Tensor:
    """RGB stays on the reference path; diagnostic modes are explicitly separate."""
    camera = make_camera(payload, background.device)
    mode = payload["mode"]
    if mode in {"primitives", "wireframe"}:
        result = renderer.render_owner(camera, model, background, sh_degree=sh_degree)
        owners = result["owner_id"]
        hashed = owners * 1664525 + 1013904223
        image = torch.stack([(hashed >> shift) & 255 for shift in (0, 8, 16)]).float() / 255
        if mode == "wireframe":
            sdf = result["owner_sdf"]
            horizontal = torch.nn.functional.pad((sdf[:, 1:] - sdf[:, :-1]).abs(), (0, 1))
            vertical = torch.nn.functional.pad((sdf[1:] - sdf[:-1]).abs(), (0, 0, 0, 1))
            width = (horizontal + vertical).clamp_min(0.002) * 1.5
            line = (1 - sdf.abs() / width).clamp(0, 1)
            edge_color = image.new_tensor([0.35, 1, 0.8])[:, None, None]
            image = result["image"].clamp(0, 1) * 0.25 * (1 - line) + edge_color * line
        return image * (owners >= 0)[None]
    result = renderer.render(
        camera, model, background, return_surface=mode in {"depth", "normals", "opacity"},
        return_overdraw_hits=mode == "overdraw", return_contrib=False, return_overdraw=mode == "overdraw",
        sh_degree=sh_degree,
    )
    if mode == "rgb":
        return result["image"]
    if mode == "overdraw":
        value = torch.log1p(result["overdraw_hits"]) / math.log(17)
        return value[None].expand(3, -1, -1)
    alpha = result["surf_alpha"]
    if mode == "opacity":
        return alpha[None].expand(3, -1, -1)
    if mode == "depth":
        depth = result["surf_depth"] / alpha.clamp_min(1e-8)
        return (depth / payload["depth_scale"])[None].expand(3, -1, -1) * (alpha > 1e-4)[None]
    normals = torch.nn.functional.normalize(result["surf_normals"], dim=0)
    return (normals * 0.5 + 0.5) * (alpha > 1e-4)[None]


@dataclass
class FrameRequest:
    payload: dict
    completed: threading.Event = field(default_factory=threading.Event)
    image: bytes | None = None
    metadata: dict = field(default_factory=dict)
    error: str | None = None
    cancelled: bool = False


class ViewerHandler(SimpleHTTPRequestHandler):
    def log_message(self, message: str, *args: object) -> None:
        LOGGER.debug(message, *args)

    def reply(self, status: int, body: bytes, content_type: str = "application/json", headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (headers or {}).items():
            self.send_header(name, str(value))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError) as error:
            LOGGER.debug("Viewer client disconnected: %s", error)

    def authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        if not secrets.compare_digest(supplied, f"Bearer {self.server.viewer.token}"):
            self.reply(401, b'{"error":"Viewer token required"}')
            return False
        return True

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/info":
            if self.authorized():
                self.reply(200, json.dumps(self.server.viewer.metadata).encode())
            return
        if path == "/":
            self.path = "/index.html"
        elif path not in {f"/{file.name}" for file in WEB_ROOT.iterdir() if file.is_file()}:
            self.reply(404, b'{"error":"Not found"}')
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/render":
            self.reply(404, b'{"error":"Not found"}')
            return
        if not self.authorized():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Render request must be at most 8192 bytes")
            self.connection.settimeout(10)
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected camera object")
            request = FrameRequest(validate_request(payload, self.server.viewer.max_pixels))
        except (ValueError, TypeError, TimeoutError) as error:
            LOGGER.warning("Rejected viewer request: %s", error)
            self.reply(400, json.dumps({"error": str(error)}).encode())
            return
        try:
            self.server.viewer.requests.put_nowait(request)
        except queue.Full:
            self.reply(429, b'{"error":"Viewer busy; retry later"}')
            return
        if not request.completed.wait(30):
            request.cancelled = True
            self.reply(503, b'{"error":"Waiting for next training viewer interval"}')
        elif request.error:
            self.reply(503, json.dumps({"error": request.error}).encode())
        else:
            self.reply(200, request.image, "image/png", {"X-Hexels-Stats": json.dumps(request.metadata)})


class ViewerServer:
    """HTTP threads queue one request; only the model-owning thread executes CUDA."""
    def __init__(self, metadata: dict, host: str = "127.0.0.1", port: int = 8080, max_pixels: int = 1920 * 1080):
        self.metadata = dict(metadata, backend="cuda", modes=list(MODES))
        self.max_pixels = max_pixels
        self.token = secrets.token_urlsafe(32)
        self.requests: queue.Queue[FrameRequest] = queue.Queue(maxsize=1)
        self.server = ThreadingHTTPServer((host, port), partial(ViewerHandler, directory=str(WEB_ROOT)))
        self.server.daemon_threads = True
        self.server.viewer = self
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> ViewerServer:
        self.thread.start()
        LOGGER.info("CUDA viewer: http://127.0.0.1:%d/?backend=cuda#token=%s", self.server.server_port, self.token)
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        while not self.requests.empty():
            request = self.requests.get_nowait()
            request.error = "Viewer stopped"
            request.completed.set()
        self.thread.join()

    def pump(self, renderer: object, model: object, background: torch.Tensor, iteration: int, loss: torch.Tensor | None = None, sh_degree: int = 3) -> bool:
        """Serve one queued frame; a failed frame is logged and returned to its client, never raised."""
        if self.requests.empty():
            return False
        request = self.requests.get_nowait()
        if request.cancelled:
            return False
        started = time.perf_counter()
        try:
            image = render_frame(renderer, model, background, request.payload, sh_degree)
            compare_mode = request.payload.get("compare_mode")
            if compare_mode is not None:
                comparison = render_frame(renderer, model, background, {**request.payload, "mode": compare_mode}, sh_degree)
                image = torch.cat((image, comparison), dim=2)
            pixels = image.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            rendered = time.perf_counter()
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG", compress_level=1)
            request.image = buffer.getvalue()
            request.metadata = {"iteration": iteration, "render_transfer_ms": (rendered - started) * 1000,
                                "encode_ms": (time.perf_counter() - rendered) * 1000,
                                "primitive_count": self.metadata["count"],
                                "mode": request.payload["mode"], "compare_mode": compare_mode}
            if background.device.type == "cuda":
                request.metadata["cuda_allocated_mib"] = torch.cuda.memory_allocated(background.device) / 2**20
            if loss is not None:
                request.metadata["loss"] = float(loss.detach().item())
            self.metadata["iteration"] = iteration
        except Exception as error:
            # One failed frame (for example CUDA out of memory) must not stop serving or a live training run.
            LOGGER.exception("CUDA viewer frame failed; continuing")
            request.error = f"Frame failed: {type(error).__name__}: {error}"
            return False
        finally:
            request.completed.set()
        return True


class TrainingViewer:
    """Opt-in observer; closed/disconnected viewers perform no CUDA work."""
    def __init__(self, trainer: object):
        self.trainer = trainer
        self.interval = int(trainer.opt.viewer_every)
        self.next_frame = 0.0
        self.frames = 0
        self.seconds = 0.0
        self.frame_seconds = 1 / float(trainer.opt.viewer_max_fps)
        metadata = scene_metadata(trainer.model.means3d.detach().cpu().numpy(), vars(trainer.opt), trainer.start_iteration)
        camera = trainer.train_cameras[0]
        world_view = camera.world_view_transform.detach().cpu().numpy().T
        metadata["cameras"] = [{"name": "Training view", "position": camera.camera_center.detach().cpu().tolist(),
                                "forward": world_view[2, :3].tolist(), "up": (-world_view[1, :3]).tolist(),
                                "fov": math.degrees(camera.FoVy)}]
        metadata["live"] = True
        metadata["count"] = trainer.model.num_active_base + trainer.model.num_children
        self.server = ViewerServer(metadata, trainer.opt.viewer_host, trainer.opt.viewer_port)

    def __enter__(self) -> TrainingViewer:
        self.server.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.__exit__(*args)

    def step(self, iteration: int, loss: torch.Tensor | None = None) -> None:
        if iteration % self.interval or time.monotonic() < self.next_frame or self.server.requests.empty():
            return
        trainer = self.trainer
        self.server.metadata["count"] = trainer.model.num_active_base + trainer.model.num_children
        started = time.perf_counter()
        served = self.server.pump(trainer.renderer, trainer._render_model(), trainer.bg_color, iteration, loss,
                                  sh_degree=getattr(trainer, "active_sh_degree", 3))
        self.seconds += time.perf_counter() - started
        self.frames += int(served)
        self.next_frame = time.monotonic() + self.frame_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--access-file", type=Path)
    parser.add_argument("--source-path", type=Path,
                        help="COLMAP scene directory for saved views; overrides source_path from config.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA viewer requires an NVIDIA GPU with CUDA; use the static web viewer on CPU-only machines")
    from hexels.renderer import Hexels3DRenderer
    model, config, iteration = load_viewer_model(args.checkpoint, args.config)
    metadata = scene_metadata(model.means3d.detach().numpy(), config, iteration)
    metadata["count"] = model.num_active_base + model.num_children
    metadata["cameras"] = colmap_viewpoints(args.source_path or Path(config.get("source_path", ".")))
    metadata["name"] = args.checkpoint.parent.parent.name
    model = model.cuda()
    renderer = Hexels3DRenderer(SimpleNamespace(**config))
    background = torch.tensor(metadata["background"], dtype=torch.float32, device="cuda")
    with ViewerServer(metadata, args.host, args.port) as server:
        if args.access_file is not None:
            args.access_file.parent.mkdir(parents=True, exist_ok=True)
            with args.access_file.open("x") as handle:
                os.chmod(handle.name, 0o600)
                json.dump({"host": socket.gethostname(), "port": server.server.server_port,
                           "url": f"http://localhost:{server.server.server_port}/?backend=cuda#token={server.token}"}, handle)
        while True:
            server.pump(renderer, model, background, iteration)
            time.sleep(0.002)


if __name__ == "__main__":
    main()
