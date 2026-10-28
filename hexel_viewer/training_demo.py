"""Opt-in controlled training and automatic movies, separate from benchmark code."""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import subprocess
import threading
import time
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch
from PIL import Image, ImageDraw

from hexels.options import Hexels3DOptions
from hexels.trainer import Hexels3DTrainer
from .viewer import ViewerHandler, ViewerServer, WEB_ROOT, render_frame
from .viewer_model import scene_metadata


LOGGER = logging.getLogger(__name__)
CONTROL_SCRIPT = Path(__file__).resolve().parent / "training_controls.mjs"


class TrainingControl:
    """Thread-safe requests; optimizer and CUDA access stay on the training thread."""

    def __init__(self, autostart: bool = False):
        self.lock = threading.Lock()
        self.state = "running" if autostart else "waiting"
        self.iteration = 0
        self.count = 0

    def request(self, action: str) -> dict:
        transitions = {("waiting", "start"): "running", ("running", "pause"): "paused",
                       ("paused", "resume"): "running"}
        with self.lock:
            if (self.state, action) not in transitions:
                raise ValueError(f"Cannot {action} while {self.state}")
            self.state = transitions[self.state, action]
            return self._snapshot()

    def _snapshot(self) -> dict:
        return {"state": self.state, "iteration": self.iteration, "primitive_count": self.count,
                "pause_semantics": "takes effect at the next optimizer-iteration boundary"}

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot()

    def update(self, iteration: int, count: int, state: str | None = None) -> None:
        with self.lock:
            self.iteration, self.count = iteration, count
            if state is not None:
                self.state = state


class DemoHandler(ViewerHandler):
    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/training":
            if self.authorized():
                self.reply(200, json.dumps(self.server.viewer.control.snapshot()).encode())
        elif path in {"/", "/index.html"}:
            page = (WEB_ROOT / "index.html").read_text().replace(
                "</body>", '<script type="module" src="training_controls.mjs"></script></body>')
            self.reply(200, page.encode(), "text/html; charset=utf-8")
        elif path == "/training_controls.mjs":
            self.reply(200, CONTROL_SCRIPT.read_bytes(), "text/javascript; charset=utf-8")
        elif path == "/app.mjs":
            script = (WEB_ROOT / "app.mjs").read_text().replace("read-only viewer", "training demo").replace("read-only", "training controls enabled")
            self.reply(200, script.encode(), "text/javascript; charset=utf-8")
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/training":
            super().do_POST()
            return
        if not self.authorized():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 128:
                raise ValueError("Training control requires a small action object")
            self.connection.settimeout(5)
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or set(payload) != {"action"} or not isinstance(payload["action"], str):
                raise ValueError("Expected a single string action: start, pause or resume")
            result = self.server.viewer.control.request(payload["action"])
        except (ValueError, TypeError, TimeoutError) as error:
            LOGGER.warning("Rejected training control: %s", error)
            self.reply(400, json.dumps({"error": str(error)}).encode())
            return
        self.reply(200, json.dumps(result).encode())


class DemoServer(ViewerServer):
    def __init__(self, metadata: dict, control: TrainingControl, host: str, port: int):
        super().__init__(metadata, host, port)
        self.control = control
        self.server.RequestHandlerClass = partial(DemoHandler, directory=str(WEB_ROOT))


def recording_plan(cameras: list, iterations: int, frames: int, width: int, height: int) -> list[dict]:
    """Two camera-only orbits with gentle dolly and elevation, no image inspection."""
    from hexels.benchmark_metrics import camera_geometry
    from scripts.nvs_trajectory import orbit, normalize

    trajectory = orbit([camera_geometry(camera) for camera in cameras],
                       {"frames": frames, "fps": 24, "codec": "libx264", "crf": 23,
                        "max_width": width, "max_height": height})
    schedule = np.linspace(0, iterations, frames).round().astype(int)
    if len(set(schedule)) != frames:
        raise ValueError("Recording frames cannot exceed training iterations plus one")
    result = []
    positions = np.asarray([frame["position"] for frame in trajectory["frames"]])
    center = positions.mean(0)
    radius = np.median(np.linalg.norm(positions - center, axis=1))
    for index, iteration in enumerate(schedule):
        frame = trajectory["frames"][(index * 2) % frames]
        phase = 2 * math.pi * index / frames
        position = np.asarray(frame["position"])
        forward = np.asarray(frame["forward"])
        up = np.asarray(frame["up"])
        focus = position + forward * np.linalg.norm(position - center)
        position = position + forward * (0.08 * radius * math.sin(phase)) + up * (0.06 * radius * math.sin(2 * phase))
        result.append({"iteration": int(iteration), "position": position.tolist(),
                       "forward": normalize(focus - position).tolist(), "up": up.tolist(),
                       "fov": math.degrees(frame["fov_y"]), "width": width, "height": height,
                       "mode": "rgb", "depth_scale": max(1.0, radius * 4)})
    return result


class TrainingRecorder:
    """Render fixed iteration/camera samples directly to MP4, not browser capture."""

    def __init__(self, trainer: Hexels3DTrainer):
        import imageio_ffmpeg

        self.trainer = trainer
        options = trainer.opt
        self.plan = recording_plan(trainer.train_cameras, options.iterations, options.record_frames,
                                   options.record_width, options.record_height)
        self.rows: list[dict] = []
        self.seconds = 0.0
        self.encoder = None
        self.output = trainer.model_path / "training_progress.mp4"
        self.command = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-n", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-s", f"{2 * options.record_width}x{options.record_height + 40}",
                        "-r", str(options.record_fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                        "-threads", "2", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.output)]
        (trainer.model_path / "recording_plan.json").write_text(json.dumps({
            "fps": options.record_fps, "frames": self.plan, "source": "training cameras only",
            "note": "Time-lapse, not real-time footage. Orbit can reveal uncaptured regions. Right panel is strongest-contributor IDs, not a mesh.",
        }, indent=2) + "\n")

    @torch.no_grad()
    def step(self, iteration: int) -> None:
        if len(self.rows) == len(self.plan) or self.plan[len(self.rows)]["iteration"] != iteration:
            return
        started = time.perf_counter()
        trainer = self.trainer
        payload = self.plan[len(self.rows)]
        images = []
        for mode in ("rgb", "primitives"):
            rendered = render_frame(trainer.renderer, trainer._render_model(), trainer.bg_color,
                                    dict(payload, mode=mode), sh_degree=trainer.active_sh_degree)
            pixels = rendered.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            images.append(Image.fromarray(pixels))
        count = trainer.model.num_active_base + trainer.model.num_children
        canvas = Image.new("RGB", (payload["width"] * 2, payload["height"] + 40), "#10181f")
        for index, image in enumerate(images):
            canvas.paste(image, (index * payload["width"], 40))
        ImageDraw.Draw(canvas).text((12, 12), f"RGB  |  iteration {iteration}/{trainer.opt.iterations}  |  {count:,} hexels  |  TRAINING TIME-LAPSE", fill="white")
        ImageDraw.Draw(canvas).text((payload["width"] + 12, 12), "Strongest-contributor primitive IDs", fill="white")
        if self.encoder is None:
            self.encoder = subprocess.Popen(self.command, stdin=subprocess.PIPE)
        self.encoder.stdin.write(canvas.tobytes())
        self.rows.append({"frame": len(self.rows), "iteration": iteration, "primitives": count})
        self.seconds += time.perf_counter() - started

    def close(self, complete: bool) -> None:
        if self.encoder is not None:
            self.encoder.stdin.close()
            if self.encoder.wait() != 0:
                raise RuntimeError("Training movie encoding failed")
        (self.trainer.model_path / "recording_summary.json").write_text(json.dumps({
            "complete": complete and len(self.rows) == len(self.plan), "frames": self.rows,
            "expected_frames": len(self.plan), "observer_seconds": self.seconds,
            "path": str(self.output), "note": "Includes RGB/diagnostic renders and encoding; not benchmark timing",
        }, indent=2) + "\n")


class DemoOptions(Hexels3DOptions):
    viewer_host: str = "127.0.0.1"
    viewer_port: int = 0
    viewer_every: int = 10
    demo_autostart: bool = False
    demo_wait_seconds: float = 7200.0
    demo_keepalive_seconds: float = 1800.0
    record_frames: int = 1440
    record_fps: int = 24
    record_width: int = 640
    record_height: int = 360

    def process_args(self) -> None:
        super().process_args()
        if not self.skip_postprocess:
            raise ValueError("The demonstration requires --skip_postprocess")
        if self.resume_checkpoint:
            raise ValueError("This demonstration records fresh training, not checkpoint continuation")
        if not 0 <= self.viewer_port <= 65535 or self.viewer_every < 1:
            raise ValueError("Invalid viewer port or iteration interval")
        if not 2 <= self.record_frames <= self.iterations + 1 or not 1 <= self.record_fps <= 60:
            raise ValueError("Invalid recording frame count or rate")
        if any(size < 64 or size > 1280 or size % 2 for size in (self.record_width, self.record_height)):
            raise ValueError("Recording dimensions must be even and between 64 and 1280")
        if not math.isfinite(self.demo_wait_seconds) or self.demo_wait_seconds <= 0 or not math.isfinite(self.demo_keepalive_seconds) or self.demo_keepalive_seconds < 0:
            raise ValueError("Invalid wait/keepalive duration")


class DemoTrainer(Hexels3DTrainer):
    """Controls and observation run only between complete optimizer iterations."""

    def service(self, iteration: int) -> None:
        count = self.model.num_active_base + self.model.num_children
        self.control.update(iteration, count)
        self.server.metadata["count"] = count
        if iteration % self.opt.viewer_every == 0:
            self.server.pump(self.renderer, self._render_model(), self.bg_color, iteration,
                             sh_degree=self.active_sh_degree)
        state = self.control.snapshot()["state"]
        if state in {"waiting", "paused"}:
            started = time.monotonic()
            while self.control.snapshot()["state"] in {"waiting", "paused"}:
                self.server.pump(self.renderer, self._render_model(), self.bg_color, iteration,
                                 sh_degree=self.active_sh_degree)
                if time.monotonic() - started > self.opt.demo_wait_seconds:
                    raise TimeoutError("Training demo timed out waiting for Start/Resume")
                time.sleep(0.02)
            self.wait_seconds += time.monotonic() - started
        self.recorder.step(iteration)

    def _update_sh_degree(self, iteration: int) -> int:
        self.service(iteration - 1)
        return super()._update_sh_degree(iteration)

    def train(self) -> dict:
        metadata = scene_metadata(self.model.means3d.detach().cpu().numpy(), vars(self.opt), 0)
        camera = self.train_cameras[0]
        transform = camera.world_view_transform.detach().cpu().numpy().T
        metadata.update(live=True, name=f"{Path(self.opt.source_path).name}: live V3 training",
                        count=self.model.num_active_base, cameras=[{"name": "Training view",
                        "position": camera.camera_center.detach().cpu().tolist(), "forward": transform[2, :3].tolist(),
                        "up": (-transform[1, :3]).tolist(), "fov": math.degrees(camera.FoVy)}])
        self.control = TrainingControl(self.opt.demo_autostart)
        self.wait_seconds = 0.0
        self.recorder = TrainingRecorder(self)
        complete = False
        with DemoServer(metadata, self.control, self.opt.viewer_host, self.opt.viewer_port) as server:
            self.server = server
            access = {"host": socket.gethostname(), "port": server.server.server_port,
                      "url": f"http://localhost:{server.server.server_port}/?backend=cuda#token={server.token}"}
            with (self.model_path / "viewer_access.json").open("x") as handle:
                os.chmod(handle.name, 0o600)
                json.dump(access, handle, indent=2)
            LOGGER.info("Live training host=%s port=%d; access file=%s", access["host"], access["port"], self.model_path / "viewer_access.json")
            try:
                summary = super().train()
                self.recorder.step(self.opt.iterations)
                complete = True
                self.control.update(self.opt.iterations, self.model.num_active_base + self.model.num_children, "completed")
            except Exception:
                self.control.update(self.control.snapshot()["iteration"], self.model.num_active_base + self.model.num_children, "failed")
                LOGGER.exception("Live training failed")
                raise
            finally:
                self.recorder.close(complete)
                (self.model_path / "demo_summary.json").write_text(json.dumps({
                    **self.control.snapshot(), "wait_seconds": self.wait_seconds,
                    "recording_seconds": self.recorder.seconds, "not_a_timing_benchmark": True,
                }, indent=2) + "\n")
            deadline = time.monotonic() + self.opt.demo_keepalive_seconds
            while time.monotonic() < deadline:
                server.pump(self.renderer, self._render_model(), self.bg_color, self.opt.iterations,
                            sh_degree=self.active_sh_degree)
                time.sleep(0.02)
        return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    DemoTrainer(DemoOptions().parse_args()).train()


if __name__ == "__main__":
    main()
