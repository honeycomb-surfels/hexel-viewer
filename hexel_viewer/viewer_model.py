"""CPU checkpoint loading and portable, activated Hexel viewer bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hexels.contact_mesh import rotation_matrices
from hexels.model import Hexels3DGeometry, Hexels3DModel, validate_active_mask_state
from hexels.render import RenderOptions, _checkpoint_view_basis_dim, _prepare_render_model_state

MAGIC = b"HEXVIEW1"
LOGGER = logging.getLogger(__name__)


def load_viewer_model(checkpoint: Path, config_path: Path | None = None) -> tuple:
    """Load trusted local checkpoints without importing a CUDA extension or images."""
    config_path = config_path or checkpoint.parent.parent / "config.json"
    defaults = RenderOptions()
    config = {name: getattr(defaults, name) for name in (
        "softness", "softness_mode", "scale_modifier", "color_model", "view_model",
        "use_view_grads", "prefilter_edge_on", "child_offset_frac", "child_scale_ratio",
        "child_rotation_delta", "children_detach_geometry", "white_background", "debug",
    )}
    config.update(json.loads(config_path.read_text()))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model"]
    geometry = Hexels3DGeometry(state["means3d"], state["log_scales"].exp(), state["rotation_quats"])
    model = Hexels3DModel(
        geometry, state["colors"], 0.1, config["softness"],
        softness_mode=config["softness_mode"],
        view_basis_dim=_checkpoint_view_basis_dim(state, config["view_model"]),
    )
    model.allocate_children(int(state.get("child_parent_idx", torch.empty(0)).numel()))
    prepared = _prepare_render_model_state(payload, model, SimpleNamespace(**config))
    validate_active_mask_state(prepared, model.num_base)
    incompatible = model.load_state_dict(prepared, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) - {"active_mask", "child_rotvec", "child_layer"}:
        raise ValueError(f"Unsupported checkpoint state: {incompatible}")
    return model.eval(), config, int(payload.get("iteration", 0))


def pack_model(model: Hexels3DModel, config: dict) -> dict:
    """Use the same active-row/child packing as the renderer."""
    return model.pack_for_rasterizer(
        child_offset_frac=config.get("child_offset_frac", 0.5),
        child_scale_ratio=config.get("child_scale_ratio", 0.5),
        child_rotation_delta=config.get("child_rotation_delta", False),
        detach_child_geometry=True,
    )


def scene_metadata(centers: np.ndarray, config: dict, iteration: int) -> dict:
    """Robust framing excludes COLMAP outliers, but never removes model rows."""
    lower, upper = np.quantile(centers, [0.05, 0.95], axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.linalg.norm(upper - lower) * 0.5), 0.5)
    return {
        "format": "hexels.viewer", "version": 1, "count": len(centers),
        "iteration": iteration, "center": center.tolist(), "radius": radius,
        "background": [1, 1, 1] if config.get("white_background", False) else [0, 0, 0],
        "near": 0.2, "far": 100.0, "prefilter": bool(config.get("prefilter_edge_on", False)),
        "conservative_bounds": bool(config.get("conservative_bounds", False)),
        "cameras": [],
    }


def colmap_viewpoints(source: Path) -> list[dict]:
    """Read registered poses only; no source images or CUDA allocations."""
    image_file = source / "sparse/0/images.bin"
    if not image_file.exists():
        LOGGER.warning("No COLMAP poses at %s; the viewer will have no saved views. Relative source paths "
                       "resolve against the current directory; pass --source-path to override.", image_file.absolute())
        return []
    fields_of_view = {}
    intrinsic_file = image_file.with_name("cameras.bin")
    if intrinsic_file.exists():
        with intrinsic_file.open("rb") as handle:
            count, = struct.unpack("<Q", handle.read(8))
            for _ in range(count):
                camera_id, camera_model, width, height = struct.unpack("<iiQQ", handle.read(24))
                if camera_model not in {0, 1}:
                    raise ValueError("Viewer camera presets require undistorted SIMPLE_PINHOLE/PINHOLE COLMAP cameras")
                parameters = struct.unpack("<" + "d" * (3 if camera_model == 0 else 4), handle.read(24 if camera_model == 0 else 32))
                focal_y = parameters[0 if camera_model == 0 else 1]
                fields_of_view[camera_id] = math.degrees(2 * math.atan(height / (2 * focal_y)))
    cameras = []
    with image_file.open("rb") as handle:
        count, = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            record = struct.unpack("<idddddddi", handle.read(64))
            name_bytes = bytearray()
            while True:
                character = handle.read(1)
                if not character:
                    raise ValueError("Truncated COLMAP image name")
                if character == b"\0":
                    break
                name_bytes.extend(character)
            points, = struct.unpack("<Q", handle.read(8))
            handle.seek(points * 24, 1)
            rotation = rotation_matrices(np.asarray(record[1:5])[None])[0]
            position = -rotation.T @ np.asarray(record[5:8])
            cameras.append({"name": name_bytes.decode("utf8"), "position": position.tolist(),
                            "forward": rotation[2].tolist(), "up": (-rotation[1]).tolist(),
                            "fov": fields_of_view.get(record[8], 60.0)})
    return sorted(cameras, key=lambda camera: camera["name"])


@torch.no_grad()
def bundle_arrays(model: Hexels3DModel, config: dict, iteration: int = 0) -> tuple[dict, np.ndarray]:
    """Export actual tangent axes, raw appearance and activated boundary parameters."""
    view_model = config.get("view_model", "none")
    if config.get("color_model") != "linear" or view_model not in {"none", "linear", "sh3"}:
        raise ValueError("Browser export supports audited linear-color V1/V2/V3, not lobe/quadratic models")
    if model.num_children:
        raise ValueError("Browser export currently supports fixed-population models without children")
    packed = {name: tensor.detach().cpu().numpy() for name, tensor in pack_model(model, config).items()}
    centers = packed["means3d"]
    basis_count = {"none": 0, "linear": 3, "sh3": 15}[view_model] if config.get("use_view_grads", True) else 0
    rows = np.zeros((len(centers), 6 + basis_count, 4), dtype="<f4")
    matrices = rotation_matrices(packed["rotations"])
    scales = packed["scales"] * config.get("scale_modifier", 1.0)
    softness = packed["softnesses"].reshape(len(centers), -1)
    softness = np.broadcast_to(softness, (len(centers), 3))
    rows[:, 0, :3], rows[:, 0, 3] = centers, packed["opacities"]
    rows[:, 1, :3] = matrices[:, :, 0] * scales[:, 0, None]
    rows[:, 2, :3] = matrices[:, :, 1] * scales[:, 1, None]
    rows[:, 3, :3] = packed["colors"]
    rows[:, 1:4, 3] = np.maximum(softness, 1e-4)
    rows[:, 4:6, :3] = packed["color_grads"].reshape(-1, 2, 3)
    if basis_count:
        rows[:, 6:, :3] = packed["view_grads"].transpose(0, 2, 1) if view_model == "linear" else packed["view_grads"]
    if not np.isfinite(rows).all() or np.any(scales <= 0):
        raise ValueError("Viewer bundle requires finite attributes and positive scales")
    metadata = scene_metadata(centers, config, iteration)
    metadata.update({"texels": rows.shape[1], "view_model": view_model, "basis_count": basis_count})
    return metadata, rows


def write_bundle(path: Path, metadata: dict, rows: np.ndarray) -> None:
    """Write a self-contained little-endian bundle with aligned JSON metadata."""
    metadata = dict(metadata, payload_sha256=hashlib.sha256(rows.tobytes()).hexdigest())
    header = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()
    header += b" " * (-len(header) % 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(MAGIC + struct.pack("<I", len(header)) + header)
        handle.write(rows.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--site", action="store_true", help="Create a new standalone site directory instead of a single bundle")
    parser.add_argument("--source-path", type=Path,
                        help="COLMAP scene directory for saved views; overrides source_path from config.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    model, config, iteration = load_viewer_model(args.checkpoint, args.config)
    metadata, rows = bundle_arrays(model, config, iteration)
    metadata["cameras"] = colmap_viewpoints(args.source_path or Path(config.get("source_path", ".")))
    metadata["name"] = args.output.stem
    metadata["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    output = args.output
    if args.site:
        output.mkdir(parents=True, exist_ok=False)
        web_root = Path(__file__).resolve().parent / "web"
        for source in web_root.iterdir():
            if source.is_file():
                shutil.copy2(source, output / source.name)
        output = output / "model.hexview"
    write_bundle(output, metadata, rows)
    logging.info("Exported %d hexels, %.2f MiB to %s", len(rows), output.stat().st_size / 2**20, output)
    if args.site:
        logging.info("Serve %s as static files, then open /?model=model.hexview", args.output)


if __name__ == "__main__":
    main()
