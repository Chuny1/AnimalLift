"""Portable JSON configuration shared by Python and Blender entry points."""
import argparse
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"
LOCAL_CONFIG = PROJECT_ROOT / "config.local.json"

PATH_KEYS = {
    "shared": {"reference_mesh_path", "vae_local_path"},
    "train": {"base_root", "data_root", "uv_mask_path", "flat_val_image_dir", "infer_input", "output_dir", "resume"},
    "test": {"resume", "flat_val_image_dir", "output_dir"},
    "blender": {"results_dir", "obj_path", "save_blend_path", "hdr_path", "underhair_mask_path", "uv_source_obj_path", "hair_meta_path", "normal_path", "roughness_path", "hair_nodes_path", "density_dir"},
    "dataset_generation": {"base_dir", "input_texture_path"},
}


def resolve_path(value):
    """All relative paths (including CLI paths) are relative to the project root."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A path must be a non-empty string or null.")
    expanded = os.path.expandvars(os.path.expanduser(value))
    if re.search(r"\$[A-Za-z_]|\$\{", expanded):
        raise ValueError(f"Undefined environment variable in path: {value}")
    if os.name != "nt" and (re.match(r"^[A-Za-z]:[\\/]", expanded) or expanded.startswith("\\\\")):
        raise ValueError("A Windows path was used on a non-Windows host; edit the local config.")
    path = Path(expanded)
    return str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def load_config(config_path=None):
    data = _read_json(DEFAULT_CONFIG)
    selected = Path(resolve_path(config_path)) if config_path else LOCAL_CONFIG
    if config_path or selected.is_file():
        if selected.resolve() != DEFAULT_CONFIG.resolve():
            override = _read_json(selected)
            for section, values in override.items():
                if section not in data or not isinstance(values, dict):
                    raise ValueError(f"Unknown or invalid config section: {section}")
                unknown = set(values) - set(data[section])
                if unknown:
                    raise ValueError(f"Unknown keys in {section}: {sorted(unknown)}")
                data[section].update(values)
    for section, keys in PATH_KEYS.items():
        for key in keys:
            if key in data[section]:
                data[section][key] = resolve_path(data[section][key])
    return data


def parse_configured_args(parser, section, argv=None):
    """CLI > selected config (or config.local.json) > portable config.json."""
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    known, _ = probe.parse_known_args(argv)
    parser.add_argument("--config", help="JSON overrides; defaults to config.local.json if present")
    try:
        config = load_config(known.config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    defaults = dict(config[section])
    if section in {"train", "test"}:
        defaults.update(config["shared"])
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    keys = set(PATH_KEYS[section])
    if section in {"train", "test"}:
        keys.update(PATH_KEYS["shared"])
    try:
        for key in keys:
            if hasattr(args, key):
                setattr(args, key, resolve_path(getattr(args, key)))
    except ValueError as exc:
        parser.error(str(exc))
    return args


def require_file(path, label):
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"Missing {label}: {path}. See docs/ASSETS.md and config.json.")


def require_dir(path, label):
    if path is None or not Path(path).is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}. Set its path in your config or CLI.")


def select_result_obj(results_dir, sample=None, obj_path=None):
    """Select a prediction without embedding sample names or device paths."""
    if obj_path:
        require_file(obj_path, "prediction OBJ")
        return str(Path(obj_path).resolve())
    require_dir(results_dir, "results directory")
    mesh_dir = Path(results_dir) / "meshes"
    if sample:
        if Path(sample).name != sample or "/" in sample or "\\" in sample:
            raise ValueError("--sample must be a filename stem, e.g. animal_2")
        name = sample if sample.endswith("_mesh.obj") else sample + "_mesh.obj"
        chosen = mesh_dir / name
        require_file(chosen, "prediction OBJ")
    else:
        candidates = sorted(mesh_dir.glob("*_mesh.obj"))
        if not candidates:
            raise FileNotFoundError(f"No *_mesh.obj predictions found in {mesh_dir}")
        chosen = candidates[0]
        print(f"[Blender] Selected {chosen.name}; use --sample to select another result.")
    return str(chosen.resolve())


def result_paths(obj_path):
    obj = Path(obj_path).resolve()
    if not obj.name.endswith("_mesh.obj"):
        raise ValueError("Prediction OBJ filename must end with _mesh.obj")
    stem = obj.name[:-len("_mesh.obj")]
    root = obj.parent.parent
    return {"dataset_dir": root, "sample_name": stem,
            "textures_dir": root / "textures",
            "texture": root / "textures" / f"{stem}_uv.png",
            "hair_map": root / "hair" / f"{stem}_hair.npz"}


def blend_output_path(obj_path, save_blend_path=None):
    """Choose a portable .blend destination for one prediction."""
    if save_blend_path is not None:
        output = Path(resolve_path(save_blend_path))
    else:
        paths = result_paths(obj_path)
        output = paths["dataset_dir"] / "blender" / (paths["sample_name"] + ".blend")
    if output.suffix.lower() != ".blend":
        raise ValueError("The Blender output path must end with .blend")
    return output
