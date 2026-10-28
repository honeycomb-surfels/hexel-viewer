"""Opt-in live-training entry point, isolated from frozen benchmark trainers."""

from __future__ import annotations

import logging
import json

from hexels.options import Hexels3DOptions
from hexels.trainer import Hexels3DTrainer
from .viewer import TrainingViewer


class LiveOptions(Hexels3DOptions):
    viewer_port: int = 8080
    viewer_host: str = "127.0.0.1"
    viewer_every: int = 100
    viewer_max_fps: float = 2.0

    def process_args(self) -> None:
        super().process_args()
        if not 1 <= self.viewer_port <= 65535 or self.viewer_every < 1 or not 0 < self.viewer_max_fps <= 30:
            raise ValueError("Invalid viewer port, interval or maximum frame rate")
        if not self.skip_postprocess:
            raise ValueError("Pass --skip_postprocess; live viewer trains only, with generic render/evaluation run separately")


class LiveTrainer(Hexels3DTrainer):
    """Observe the iteration boundary before its render, never inside autograd."""
    def train(self) -> dict[str, object]:
        with TrainingViewer(self) as viewer:
            self.live_viewer = viewer
            summary = super().train()
            (self.model_path / "viewer_summary.json").write_text(json.dumps({
                "frames": viewer.frames, "observer_seconds": viewer.seconds,
                "note": "Training wall time includes observer work; not a benchmark timing run",
            }, indent=2))
            return summary

    def _update_sh_degree(self, iteration: int) -> int:
        self.live_viewer.step(iteration - 1)
        return super()._update_sh_degree(iteration)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    options = LiveOptions().parse_args()
    LiveTrainer(options).train()


if __name__ == "__main__":
    main()
