#!/usr/bin/env python3
"""Train AnimalLift; paths are configured in config.json or --config."""
import argparse
from animallift.config import parse_configured_args, require_file, require_dir

def parse_args(argv=None):
    p = argparse.ArgumentParser("Texture + Mesh + Hair multitask training")

    p.add_argument("--mode", choices=["train_multitask", "infer_multitask"], default="train_multitask")

    # dataset
    p.add_argument("--data_root", default=None)
    p.add_argument("--base_root", default=None)
    p.add_argument("--all_animals", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--animals", nargs="+", default=["dog", "small_cat", "big_cat", "wolf", "fox", "bear"])
    p.add_argument("--infer_input", default=None)
    p.add_argument("--guidance_dirname", default="augmented_images")
    p.add_argument("--texture_dirname", default="textures")
    p.add_argument("--shape_dirname", default="shapes")
    p.add_argument("--hair_dirname", default="hair_maps_single_512_uvlocal")
    p.add_argument("--hair_region_names", nargs="+", default=["face", "ear", "neck", "body", "leg", "tail", "eye_brow", "fore_head", "nose", "jaw"])
    p.add_argument("--recursive_dataset", action="store_true")

    # split
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)

    # mask
    p.add_argument("--uv_mask_path", default=None)
    p.add_argument("--invert_mask", action="store_true")

    # sizes
    p.add_argument("--ctrl_size", type=int, default=256)
    p.add_argument("--texture_size", type=int, default=1024)
    p.add_argument("--latent_size", type=int, default=None,
                   help="Auto: texture_size / 8 (1024 -> 128, 512 -> 64)")
    p.add_argument("--latent_ch", type=int, default=None,
                   help="Auto: pretrained VAE latent_channels (4 for SD)")

    # Frozen pretrained Stable Diffusion VAE; no separate AE training stage.
    p.add_argument("--vae_model", default="stabilityai/sd-vae-ft-mse",
                   help="Hugging Face model ID or local Diffusers VAE directory")
    p.add_argument("--vae_subfolder", default=None,
                   help="Use 'vae' when loading from a complete SD model directory")
    p.add_argument("--vae_revision", default=None,
                   help="Optional Hugging Face model commit/tag")
    p.add_argument("--vae_local_files_only", action="store_true",
                   help="Use only local/cached VAE weights")
    p.add_argument("--vae_slicing", action=argparse.BooleanOptionalAction, default=True,
                   help="Process VAE batches in slices; disable with --no-vae_slicing")
    p.add_argument("--vae_tiling", action="store_true",
                   help="Reduce VAE spatial memory with overlapping tiles; may alter reconstruction")
    p.add_argument("--vae_sample_posterior", action="store_true",
                   help="Sample posterior during training; default uses its deterministic mean")

    # multitask / flow
    p.add_argument("--flow_base_ch", type=int, default=128)
    p.add_argument("--flow_ch_mult", type=int, nargs="+", default=[1, 2, 4, 4])
    p.add_argument("--flow_attn_levels", type=int, nargs="+", default=[2, 3])
    p.add_argument("--unfreeze_last_n_blocks", type=int, default=2)
    p.add_argument("--use_uv_cross_attn", action="store_true", default=False,
                   help="Compatibility flag; this encoder does not implement a separate UV cross-attention module.")
    p.add_argument("--n_surface_points", type=int, default=256)
    p.add_argument("--uv_attn_heads", type=int, default=8)
    p.add_argument("--uv_attn_dropout", type=float, default=0.1)
    p.add_argument("--use_color_embed", action="store_true", default=True)
    p.add_argument("--use_dino_intermediate_tokens", action="store_true", default=True)
    p.add_argument("--dino_intermediate_layers", type=int, nargs="+", default=[3, 7])
    p.add_argument("--dino_intermediate_grid", type=int, default=16)
    p.add_argument("--freeze_dino_intermediate", action="store_true", default=True)
    p.add_argument("--cond_drop_prob", type=float, default=0.10)
    p.add_argument("--val_cfg_scale", type=float, default=3.0)
    p.add_argument("--freeze_encoder", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--flow_max_steps", type=int, default=100_000)
    p.add_argument("--flow_warmup_steps", type=int, default=1000)
    p.add_argument("--flow_val_every", type=int, default=1000)
    p.add_argument("--flow_save_every", type=int, default=5000)

    # mesh/hair heads
    p.add_argument("--mesh_hidden_dim", type=int, default=1024)
    p.add_argument("--hair_hidden_dim", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.1)
    p.add_argument("--mesh_head_depth", type=int, default=2)
    p.add_argument("--hair_head_depth", type=int, default=2)

    # losses
    p.add_argument("--texture_loss_weight", type=float, default=1.0)
    p.add_argument("--geometry_loss_weight", type=float, default=10)
    p.add_argument("--mesh_loss_type", choices=["l1", "l2"], default="l1")
    p.add_argument("--mesh_edge_weight", type=float, default=0.1)
    p.add_argument("--mesh_laplacian_weight", type=float, default=0.05)
    p.add_argument("--mesh_offset_reg_weight", type=float, default=0.001)
    p.add_argument("--total_hair_weight", type=float, default=1.0)
    p.add_argument("--hair_l1_weight", type=float, default=1.0)
    p.add_argument("--hair_mse_weight", type=float, default=0.0)
    p.add_argument("--hair_fg_weight", type=float, default=5.0)
    p.add_argument("--hair_bg_weight", type=float, default=0.0)
    p.add_argument("--hair_norm_scale", type=float, default=0.1)
    p.add_argument("--hair_region_length_scale", type=float, default=0.05)
    p.add_argument("--region_len_weight", type=float, default=1.0)
    p.add_argument("--region_curl_weight", type=float, default=0.5)
    p.add_argument("--fg_weight", type=float, default=5.0)

    # loaders
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--mixed_precision", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ema_decay", type=float, default=0.999)

    # flat-dir validation / inference
    p.add_argument("--val_nfe", type=int, default=50)
    p.add_argument("--reference_mesh_path", default=None)
    p.add_argument("--flat_val_image_dir", default=None,
                   help="Flat directory of real-world validation images")
    p.add_argument("--val_save_count", type=int, default=8,
                   help="How many shuffled flat-dir images to save each validation run")

    # paths
    p.add_argument("--output_dir", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda")

    p.add_argument("--vae_local_path", help="Local copy of the same pretrained VAE; keeps checkpoint identity")
    return parse_configured_args(p, "train", argv)



def main():
    args = parse_args()
    require_file(args.reference_mesh_path, "reference OBJ")
    if args.resume:
        require_file(args.resume, "checkpoint")
    if args.mode == "train_multitask":
        require_file(args.uv_mask_path, "UV occupancy mask")
        require_dir(args.flat_val_image_dir, "validation images")
        if args.all_animals:
            from pathlib import Path
            for animal in args.animals:
                require_dir(Path(args.base_root) / animal / "dataset", f"{animal} training data")
        else:
            require_dir(args.data_root, "training data")
    else:
        require_file(args.resume, "checkpoint")
        require_dir(args.infer_input, "input images")
    from animallift.model import set_seed, train_multitask, infer_multitask
    set_seed(args.seed)
    if args.mode == "train_multitask":
        train_multitask(args)
    else:
        infer_multitask(args)


if __name__ == "__main__":
    main()
