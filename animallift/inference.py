"""Folder inference and portable result export."""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


def _parse_int_tuple(x: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if x is None:
        return default
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    if isinstance(x, str):
        return tuple(int(v.strip()) for v in x.replace(",", " ").split() if v.strip())
    return default


def _parse_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def import_training_module(module_name_or_path: str) -> Any:
    """Import the training module by dotted name or an explicit .py path."""
    module_path = Path(module_name_or_path).expanduser()
    if module_path.suffix == ".py" or module_path.is_file():
        if not module_path.is_file():
            raise FileNotFoundError(f"Training module not found: {module_path}")
        spec = importlib.util.spec_from_file_location(
            "sd_vae_training_module", module_path.resolve()
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import training module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_name_or_path)


def build_runtime_args(
    cli_args: argparse.Namespace,
    ckpt: dict[str, Any],
    train_mod: Any,
) -> SimpleNamespace:
    """Restore checkpoint model settings and apply runtime-only CLI options."""
    if ckpt.get("vae_config", {}).get("type") != "stable_diffusion_autoencoderkl":
        raise ValueError(
            "Checkpoint is incompatible: expected an SD-VAE multitask "
            "checkpoint containing vae_config. Old custom-AE flow weights "
            "cannot be tested with this script."
        )
    ckpt_args = ckpt.get("args")
    if not isinstance(ckpt_args, dict):
        raise ValueError("Checkpoint is missing its saved args dictionary.")

    runtime_args = SimpleNamespace(**ckpt_args)
    train_mod.restore_checkpoint_model_args(runtime_args, ckpt)

    # These options do not change the learned latent representation.
    runtime_args.resume = cli_args.resume
    runtime_args.flat_val_image_dir = cli_args.flat_val_image_dir
    runtime_args.reference_mesh_path = cli_args.reference_mesh_path
    runtime_args.output_dir = cli_args.output_dir
    runtime_args.device = cli_args.device
    runtime_args.max_images = cli_args.max_images
    runtime_args.batch_size = cli_args.batch_size
    runtime_args.seed = cli_args.seed
    runtime_args.val_nfe = (
        cli_args.val_nfe
        if cli_args.val_nfe is not None
        else int(ckpt_args.get("val_nfe", 50))
    )
    runtime_args.val_cfg_scale = (
        cli_args.val_cfg_scale
        if cli_args.val_cfg_scale is not None
        else float(ckpt_args.get("val_cfg_scale", 1.0))
    )
    runtime_args.vae_local_path = cli_args.vae_local_path
    runtime_args.vae_local_files_only = bool(cli_args.vae_local_files_only)
    runtime_args.vae_slicing = bool(cli_args.vae_slicing)
    runtime_args.hair_norm_scale = float(ckpt_args.get("hair_norm_scale", 0.1))

    runtime_args.flow_ch_mult = _parse_int_tuple(
        runtime_args.flow_ch_mult, (1, 2, 4, 4)
    )
    runtime_args.flow_attn_levels = _parse_int_tuple(
        runtime_args.flow_attn_levels, (2, 3)
    )
    runtime_args.dino_intermediate_layers = _parse_int_tuple(
        runtime_args.dino_intermediate_layers, (3, 7)
    )
    runtime_args.use_uv_cross_attn = _parse_bool(runtime_args.use_uv_cross_attn)
    runtime_args.use_color_embed = _parse_bool(runtime_args.use_color_embed, True)
    runtime_args.use_dino_intermediate_tokens = _parse_bool(
        runtime_args.use_dino_intermediate_tokens, True
    )
    runtime_args.freeze_dino_intermediate = _parse_bool(
        runtime_args.freeze_dino_intermediate, True
    )
    return runtime_args


def load_models(
    runtime_args: SimpleNamespace,
    train_mod: Any,
    device: torch.device,
    ckpt: dict[str, Any],
):
    # This uses the same wrapper as training: frozen fp32 AutoencoderKL plus
    # paired config.scaling_factor handling in encode/decode.
    vae = train_mod.build_texture_vae(runtime_args, device)
    train_mod.validate_checkpoint_vae(ckpt, vae)

    template_verts = ckpt["template_verts"].to(device).float()
    num_verts = int(ckpt["num_verts"])
    hair_channels = int(ckpt["hair_channels"])
    hair_size = tuple(int(x) for x in ckpt["hair_size"])

    model = train_mod.AnimalTextureMeshHairFlowModel(
        ctrl_img_size=runtime_args.ctrl_size,
        latent_size=runtime_args.latent_size,
        latent_ch=runtime_args.latent_ch,
        flow_base_ch=runtime_args.flow_base_ch,
        flow_ch_mult=tuple(runtime_args.flow_ch_mult),
        freeze_encoder=True,
        flow_attn_levels=tuple(runtime_args.flow_attn_levels),
        num_verts=num_verts,
        hair_out_channels=hair_channels,
        hair_size=hair_size,
        mesh_hidden_dim=runtime_args.mesh_hidden_dim,
        hair_hidden_dim=runtime_args.hair_hidden_dim,
        head_dropout=runtime_args.head_dropout,
        mesh_head_depth=runtime_args.mesh_head_depth,
        hair_head_depth=runtime_args.hair_head_depth,
        unfreeze_last_n_blocks=runtime_args.unfreeze_last_n_blocks,
        use_uv_cross_attn=runtime_args.use_uv_cross_attn,
        n_surface_points=runtime_args.n_surface_points,
        uv_attn_heads=runtime_args.uv_attn_heads,
        uv_attn_dropout=runtime_args.uv_attn_dropout,
        use_color_embed=runtime_args.use_color_embed,
        use_dino_intermediate_tokens=runtime_args.use_dino_intermediate_tokens,
        dino_intermediate_layers=tuple(runtime_args.dino_intermediate_layers),
        dino_intermediate_grid=runtime_args.dino_intermediate_grid,
        freeze_dino_intermediate=runtime_args.freeze_dino_intermediate,
        num_hair_regions=len(train_mod.parse_region_names(runtime_args.hair_region_names)),
    ).to(device).eval()
    model.load_state_dict(ckpt.get("ema_model", ckpt["model"]))
    model.val_cfg_scale = float(runtime_args.val_cfg_scale)

    return vae, model, template_verts


@torch.no_grad()
def run_flat_validation(
    runtime_args: SimpleNamespace,
    train_mod: Any,
    ckpt: dict[str, Any],
) -> None:
    if not os.path.isdir(runtime_args.flat_val_image_dir):
        raise FileNotFoundError(f"flat_val_image_dir not found: {runtime_args.flat_val_image_dir}")
    if not os.path.isfile(runtime_args.reference_mesh_path):
        raise FileNotFoundError(f"reference_mesh_path not found: {runtime_args.reference_mesh_path}")

    image_paths = train_mod.collect_flat_images(runtime_args.flat_val_image_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in flat validation folder: {runtime_args.flat_val_image_dir}")

    if runtime_args.max_images is not None and runtime_args.max_images > 0:
        image_paths = image_paths[: runtime_args.max_images]

    stems = [Path(p).stem for p in image_paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Input images must have unique stems (e.g. cat.png and cat.jpg would overwrite outputs).")
    device = torch.device(runtime_args.device if torch.cuda.is_available() else "cpu")
    ref_verts, faces, verts_uvs, faces_uvs = train_mod.load_obj(runtime_args.reference_mesh_path)
    if len(ref_verts) != int(ckpt["num_verts"]):
        raise ValueError("Reference OBJ vertex count does not match the checkpoint; use the original template topology.")
    if verts_uvs is None or faces_uvs is None or (faces_uvs < 0).any():
        raise ValueError("Reference OBJ must have complete face UV indices for texture and hair export.")
    vae, model, template_verts = load_models(runtime_args, train_mod, device, ckpt)

    out_root = Path(runtime_args.output_dir)
    tex_dir = out_root / "textures"
    hair_dir = out_root / "hair"
    mesh_dir = out_root / "meshes"
    input_dir = out_root / "input_images"
    for d in (tex_dir, hair_dir, mesh_dir, input_dir):
        d.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((runtime_args.ctrl_size, runtime_args.ctrl_size), transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["input", "texture_png", "hair_preview_png", "hair_npz", "mesh_obj"],
        )
        writer.writeheader()

        batch_size = max(1, int(runtime_args.batch_size))
        pbar = tqdm(range(0, len(image_paths), batch_size), desc=f"Testing flat validation bs={batch_size}")

        for start in pbar:
            batch_paths = [Path(p) for p in image_paths[start : start + batch_size]]

            def load_rgb_white_bg(img_path):
                img = Image.open(img_path).convert("RGBA")
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img).convert("RGB")
                return img

            ctrl = torch.stack([
                tfm(load_rgb_white_bg(img_path))
                for img_path in batch_paths
            ], dim=0).to(device, non_blocking=True)

            outs = model.infer_all(
                ctrl,
                nfe=int(runtime_args.val_nfe),
                cfg_scale=float(runtime_args.val_cfg_scale),
            )

            expected_latent = (
                len(batch_paths),
                int(runtime_args.latent_ch),
                int(runtime_args.latent_size),
                int(runtime_args.latent_size),
            )
            if tuple(outs["z_pred"].shape) != expected_latent:
                raise RuntimeError(
                    f"Flow returned latent {tuple(outs['z_pred'].shape)}, "
                    f"expected {expected_latent} from the checkpoint/VAE."
                )

            # The training wrapper reverses config.scaling_factor before decode.
            pred_texture = vae.decode(outs["z_pred"]).detach().cpu()
            expected_texture_hw = (
                int(runtime_args.texture_size),
                int(runtime_args.texture_size),
            )
            if tuple(pred_texture.shape[-2:]) != expected_texture_hw:
                raise RuntimeError(
                    f"VAE decoded texture size {tuple(pred_texture.shape[-2:])}, "
                    f"expected {expected_texture_hw}."
                )
            pred_hair = outs["pred_hair"].detach().cpu()
            pred_verts_batch = (template_verts.unsqueeze(0) + outs["pred_offsets"]).detach().cpu().numpy()

            for i, img_path in enumerate(batch_paths):
                stem = img_path.stem

                texture_path = tex_dir / f"{stem}_uv.png"
                save_image((pred_texture[i] * 0.5 + 0.5).clamp(0, 1), texture_path)

                hair_preview_path = hair_dir / f"{stem}_hair.png"
                hair_npz_path = hair_dir / f"{stem}_hair.npz"
                train_mod.save_hair_offset_preview(
                    pred_hair[i],
                    str(hair_preview_path),
                    hair_norm_scale=float(runtime_args.hair_norm_scale),
                )
                train_mod.save_hair_offset_npz_compatible(
                    pred_hair[i],
                    str(hair_npz_path),
                    hair_norm_scale=float(runtime_args.hair_norm_scale),
                )

                mesh_path = mesh_dir / f"{stem}_mesh.obj"
                train_mod.save_obj(str(mesh_path), pred_verts_batch[i], faces, verts_uvs, faces_uvs)

                copied_input = input_dir / img_path.name
                shutil.copy2(img_path, copied_input)

                writer.writerow({
                    "input": copied_input.relative_to(out_root).as_posix(),
                    "texture_png": texture_path.relative_to(out_root).as_posix(),
                    "hair_preview_png": hair_preview_path.relative_to(out_root).as_posix(),
                    "hair_npz": hair_npz_path.relative_to(out_root).as_posix(),
                    "mesh_obj": mesh_path.relative_to(out_root).as_posix(),
                })

            del ctrl, outs, pred_texture, pred_hair, pred_verts_batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"[Done] Saved {len(image_paths)} validation predictions to: {out_root}")
    print(f"[Done] Manifest: {manifest_path}")


def run(cli_args):
    train_mod = import_training_module(cli_args.module_name)
    ckpt = torch.load(cli_args.resume, map_location="cpu")
    runtime_args = build_runtime_args(cli_args, ckpt, train_mod)
    train_mod.set_seed(runtime_args.seed)
    run_flat_validation(runtime_args, train_mod, ckpt)
