#!/usr/bin/env python3
"""Predict texture, mesh and hair for every image in an input folder."""
import argparse
from animallift.config import parse_configured_args, require_file, require_dir

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Test multitask model on flattened validation image folder")
    p.add_argument("--module_name", "--module", default="animallift.model",
                   help="Training module name or path to its .py file")
    p.add_argument("--resume", "--checkpoint", default=None, help="Multitask checkpoint, e.g. output/multitask/checkpoints/latest.pt")
    p.add_argument("--flat_val_image_dir", "--input_dir", default=None, help="Folder containing flat validation images")
    p.add_argument("--reference_mesh_path", default=None, help="OBJ whose faces/UV topology are used for export")
    p.add_argument("--output_dir", "--save_dir", default=None,
                   help="Where predictions are written")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=2,
                   help="Validation batch size; 1024 texture decoding uses more memory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_nfe", "--nfe", type=int, default=None)
    p.add_argument("--val_cfg_scale", "--cfg_scale", type=float, default=None)
    p.add_argument("--vae_local_files_only", action="store_true",
                   help="Load only already-downloaded/local VAE weights")
    p.add_argument("--vae_slicing", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Slice VAE batches to lower memory; disable with --no-vae_slicing")
    p.add_argument("--vae_local_path", help="Local copy of the same pretrained VAE")
    return parse_configured_args(p, "test", argv)


def main():
    args = parse_args()
    require_file(args.resume, "AnimalLift checkpoint (download into ckpt/)")
    require_file(args.reference_mesh_path, "reference OBJ")
    require_dir(args.flat_val_image_dir, "input image folder")
    from animallift.inference import run
    run(args)


if __name__ == "__main__":
    main()
