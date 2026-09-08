"""
AnimalLift texture, mesh, and hair prediction with a frozen Stable Diffusion VAE.

The default VAE is stabilityai/sd-vae-ft-mse. A 1024x1024 RGB texture is encoded
as a scaled [B, 4, 128, 128] latent. The VAE is frozen and runs in float32;
latent dimensions are inferred from the VAE and texture resolution.

Encoding uses the posterior mean by default. VAE slicing is enabled, and
optional tiling is recorded in checkpoint metadata. Resume and inference
restore the model architecture and VAE configuration from the checkpoint.
A runtime-only local_path can relocate the same pretrained VAE weights.

References:
https://huggingface.co/stabilityai/sd-vae-ft-mse
https://huggingface.co/docs/diffusers/en/api/models/autoencoderkl
"""

import os
import glob
import re
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# --------------------------------------------------------
# OBJ loader with UV support
# --------------------------------------------------------
def load_obj(path):
    verts = []
    verts_uvs = []
    faces = []
    faces_uvs = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                vals = line.split()[1:4]
                verts.append(list(map(float, vals)))

            elif line.startswith("vt "):
                vals = line.split()[1:]
                if len(vals) < 2:
                    raise ValueError(f"Invalid vt line in {path}: {line}")
                verts_uvs.append([float(vals[0]), float(vals[1])])

            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) < 3:
                    raise ValueError(f"Face has fewer than 3 vertices in {path}: {line}")

                poly_v = []
                poly_vt = []

                for tok in tokens:
                    parts = tok.split("/")

                    if len(parts) < 1 or parts[0] == "":
                        raise ValueError(f"Invalid face token in {path}: {tok}")
                    poly_v.append(int(parts[0]) - 1)

                    if len(parts) > 1 and parts[1] != "":
                        poly_vt.append(int(parts[1]) - 1)
                    else:
                        poly_vt.append(-1)

                for i in range(1, len(poly_v) - 1):
                    faces.append([poly_v[0], poly_v[i], poly_v[i + 1]])
                    faces_uvs.append([poly_vt[0], poly_vt[i], poly_vt[i + 1]])

    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    if len(verts_uvs) > 0:
        verts_uvs = np.asarray(verts_uvs, dtype=np.float32)
    else:
        verts_uvs = None

    if len(faces_uvs) > 0:
        faces_uvs = np.asarray(faces_uvs, dtype=np.int64)
    else:
        faces_uvs = None

    return verts, faces, verts_uvs, faces_uvs


# --------------------------------------------------------
# Mesh normalization
# --------------------------------------------------------
def normalize_verts(verts):
    center = verts.mean(axis=0, keepdims=True)
    verts = verts - center
    return verts.astype(np.float32)




import os
import math
import glob
import re
import random
import argparse
import json
from pathlib import Path
from copy import deepcopy
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")

try:
    from diffusers import AutoencoderKL
except ImportError as exc:
    raise ImportError(
        "Please install the Stable Diffusion VAE dependencies: "
        "pip install -U diffusers transformers accelerate safetensors"
    ) from exc



# ============================================================================
# UTILITY
# ============================================================================

def safe_group_norm(num_channels: int) -> nn.GroupNorm:
    for g in range(min(32, num_channels), 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", str(s))]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_binary_mask(mask_path: str, size: int, invert: bool = False) -> torch.Tensor:
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    img = Image.open(mask_path).convert("L")
    img = img.resize((size, size), Image.NEAREST)
    mask = transforms.ToTensor()(img)
    mask = (mask > 0.5).float()
    if invert:
        mask = 1.0 - mask
    return mask


def save_tensor_image(x: torch.Tensor, path: str):
    save_image((x * 0.5 + 0.5).clamp(0, 1), path)


def collect_flat_images(folder: str):
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = [str(p) for p in Path(folder).iterdir()
             if p.is_file() and p.suffix.lower() in extensions]
    return sorted(paths, key=natural_key)


def build_edges_from_faces(faces: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(faces, torch.Tensor):
        faces_np = faces.detach().cpu().numpy()
    else:
        faces_np = np.asarray(faces)

    edges = set()
    for a, b, c in faces_np:
        tri = [int(a), int(b), int(c)]
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if u > v:
                u, v = v, u
            edges.add((u, v))

    edges = torch.tensor(sorted(edges), dtype=torch.long)
    if edges.numel() == 0:
        raise ValueError("Failed to build mesh edge list from faces.")
    return edges


def build_vertex_neighbors(num_verts: int, edges: torch.Tensor):
    neighbors = [[] for _ in range(num_verts)]
    edges_np = edges.detach().cpu().numpy()
    for u, v in edges_np:
        u = int(u)
        v = int(v)
        neighbors[u].append(v)
        neighbors[v].append(u)
    return neighbors


# ============================================================================
# HAIR IO
# ============================================================================

def load_hair_npz_compatible(path: str, hair_norm_scale: float = 0.1) -> torch.Tensor:
    """
    Legacy helper. Returns normalized hair offsets as [C, H, W] float tensor.
    Kept for compatibility with older utilities, but training now uses the
    quantized int8 path below and dequantizes on GPU.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Hair npz not found: {path}")

    data = np.load(path, allow_pickle=True)
    q = data["hair_offsets_local_q"].astype(np.float32)   # [H, W, S-1, 3]
    scale = data["offset_scale"].astype(np.float32)       # [3]

    offsets = q / 127.0 * scale.reshape(1, 1, 1, 3)
    offsets = offsets.transpose(2, 3, 0, 1)               # [S-1, 3, H, W]
    offsets = offsets.reshape(-1, offsets.shape[-2], offsets.shape[-1])
    offsets = offsets / float(hair_norm_scale)
    return torch.from_numpy(offsets).float()


def load_hair_quantized_pt_or_npz(path: str):
    """
    Fast training loader. Does NOT expand int8 hair to float32 on CPU.

    Prefer same-name .pt cache:
      xxx.npz -> xxx.pt

    .pt format:
      {
        "hair_q": int8 tensor [H, W, S-1, 3],
        "scale": float32 tensor [3],
        "hair_mask": bool tensor [H, W]
      }

    Returns:
      hair_q:     int8 tensor  [H, W, S-1, 3]
      hair_scale: float tensor [3]
      hair_mask:  bool tensor  [1, H, W]
    """
    pt_path = os.path.splitext(path)[0] + ".pt"
    if os.path.isfile(pt_path):
        d = torch.load(pt_path, map_location="cpu")
        hair_q = d["hair_q"].to(torch.int8).contiguous()
        hair_scale = d["scale"].float().contiguous()
        hair_mask = d["hair_mask"].bool().unsqueeze(0).contiguous()
        return hair_q, hair_scale, hair_mask

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Hair file not found: {path}")

    data = np.load(path, allow_pickle=True)
    hair_q_np = data["hair_offsets_local_q"].astype(np.int8)      # [H, W, S-1, 3]
    hair_scale_np = data["offset_scale"].astype(np.float32)       # [3]
    hair_mask_np = (np.abs(hair_q_np).sum(axis=(2, 3)) > 0)        # [H, W]

    hair_q = torch.from_numpy(hair_q_np).to(torch.int8).contiguous()
    hair_scale = torch.from_numpy(hair_scale_np).float().contiguous()
    hair_mask = torch.from_numpy(hair_mask_np).bool().unsqueeze(0).contiguous()
    return hair_q, hair_scale, hair_mask


def dequantize_hair_on_device(
    hair_q: torch.Tensor,
    hair_scale: torch.Tensor,
    hair_norm_scale: float = 0.1,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Dequantize hair on GPU/device.

    Inputs:
      hair_q:     int8 tensor  [B, H, W, S-1, 3]
      hair_scale: float tensor [B, 3]

    Returns:
      hair_offset: tensor [B, C, H, W], C=(S-1)*3
    """
    if hair_q.dim() != 5:
        raise ValueError(f"Expected hair_q [B,H,W,S-1,3], got {tuple(hair_q.shape)}")
    if hair_scale.dim() != 2 or hair_scale.shape[-1] != 3:
        raise ValueError(f"Expected hair_scale [B,3], got {tuple(hair_scale.shape)}")

    B, H, W, S1, XYZ = hair_q.shape
    if XYZ != 3:
        raise ValueError(f"Expected last hair_q dim to be 3, got {XYZ}")

    if dtype is None:
        dtype = torch.float32

    x = hair_q.to(dtype=dtype)
    scale = hair_scale.to(device=hair_q.device, dtype=dtype).view(B, 1, 1, 1, 3)
    x = x / 127.0 * scale
    x = x.permute(0, 3, 4, 1, 2).contiguous()  # [B, S-1, 3, H, W]
    x = x.reshape(B, S1 * 3, H, W)
    x = x / float(hair_norm_scale)
    return x


def load_hair_pt_or_npz_compatible(path: str, hair_norm_scale: float = 0.1):
    """
    Legacy compatibility wrapper. Training should use load_hair_quantized_pt_or_npz()
    plus dequantize_hair_on_device().
    """
    hair_q, hair_scale, hair_mask = load_hair_quantized_pt_or_npz(path)
    hair_offset = dequantize_hair_on_device(
        hair_q.unsqueeze(0),
        hair_scale.unsqueeze(0),
        hair_norm_scale=hair_norm_scale,
        dtype=torch.float32,
    )[0].cpu()
    return hair_offset.float(), hair_mask.float()

def save_hair_offset_preview(x_norm, path, hair_norm_scale=0.1, preview_max_mag=0.05):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = x_norm.detach().cpu() * float(hair_norm_scale)
    c, h, w = x.shape
    if c % 3 != 0:
        raise ValueError(f"Hair offset channels must be divisible by 3, got {c}")

    s_minus_1 = c // 3
    x = x.reshape(s_minus_1, 3, h, w)
    mag = torch.norm(x, dim=1).mean(dim=0, keepdim=True)
    mag = (mag / max(preview_max_mag, 1e-6)).clamp(0, 1)
    save_image(mag, path)


def save_hair_offset_npz_compatible(x_norm, path, hair_norm_scale=0.1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x_np = x_norm.detach().cpu().numpy()
    c, h, w = x_np.shape
    if c % 3 != 0:
        raise ValueError(f"Hair offset channels must be divisible by 3, got {c}")

    s_minus_1 = c // 3
    samples = s_minus_1 + 1
    x_np = x_np * float(hair_norm_scale)
    offsets_local = x_np.reshape(s_minus_1, 3, h, w).transpose(2, 3, 0, 1).astype(np.float32)

    offset_scale = np.max(np.abs(offsets_local), axis=(0, 1, 2)).astype(np.float32)
    offset_scale = np.maximum(offset_scale, 1e-6)

    hair_offsets_local_q = np.round(
        offsets_local / offset_scale.reshape(1, 1, 1, 3) * 127.0
    )
    hair_offsets_local_q = np.clip(hair_offsets_local_q, -127, 127).astype(np.int8)

    np.savez_compressed(
        path,
        hair_offsets_local_q=hair_offsets_local_q,
        offset_scale=offset_scale,
        samples=np.int32(samples),
    )


# ============================================================================
# OBJ EXPORT
# ============================================================================

def save_obj(path, verts, faces, verts_uvs=None, faces_uvs=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        if verts_uvs is not None:
            for vt in verts_uvs:
                f.write(f"vt {vt[0]} {vt[1]}\n")

        if verts_uvs is not None and faces_uvs is not None:
            for face, face_uv in zip(faces, faces_uvs):
                if np.any(np.asarray(face_uv) < 0):
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                else:
                    f.write(
                        f"f "
                        f"{face[0]+1}/{face_uv[0]+1} "
                        f"{face[1]+1}/{face_uv[1]+1} "
                        f"{face[2]+1}/{face_uv[2]+1}\n"
                    )
        else:
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# ============================================================================
# HAIR REGION AUXILIARY LABELS / MASKS
# ============================================================================
DEFAULT_HAIR_REGION_NAMES = (
    "face", "ear", "neck", "body", "leg", "tail",
    "eye_brow", "fore_head", "nose", "jaw",
)
CURL_POSITIVE_TYPES = {"wavy", "curly", "coily", "curl", "curled", "ringlet"}

def parse_region_names(x):
    if x is None:
        return list(DEFAULT_HAIR_REGION_NAMES)
    if isinstance(x, str):
        return [t.strip() for t in x.replace(",", " ").split() if t.strip()]
    return list(x)

def resolve_dataset_local_path(path_value: str, sample_root: str) -> str:
    if not path_value:
        return ""
    path_value = str(path_value)
    if os.path.isfile(path_value):
        return path_value
    sample_root = str(sample_root)
    sample_name = os.path.basename(sample_root)
    parts = Path(path_value).parts
    if sample_name in parts:
        idx = parts.index(sample_name)
        suffix = os.path.join(*parts[idx + 1:]) if idx + 1 < len(parts) else ""
        candidate = os.path.join(sample_root, suffix)
        if os.path.isfile(candidate):
            return candidate
    candidate = os.path.join(sample_root, os.path.basename(path_value))
    return candidate if os.path.isfile(candidate) else path_value

def curl_type_to_binary(region_info: dict) -> float:
    curl_type = str(region_info.get("curl_type", "straight")).lower()
    curl_radius = float(region_info.get("curl_radius", 0.0) or 0.0)
    return 1.0 if (curl_type in CURL_POSITIVE_TYPES or curl_radius > 1e-8) else 0.0

def read_hair_info_labels(hair_info_path: str, object_key: str, region_names):
    """Read region-level length/curl labels from hair_info.json. No region masks are read."""
    R = len(region_names)
    length = torch.zeros(R, dtype=torch.float32)
    curl = torch.zeros(R, dtype=torch.float32)
    valid = torch.zeros(R, dtype=torch.float32)
    if not hair_info_path or not os.path.isfile(hair_info_path):
        return length, curl, valid
    with open(hair_info_path, "r") as f:
        info = json.load(f)
    resolved = info.get("resolved_hair_params", {}) or {}
    obj = (info.get("objects", {}) or {}).get(object_key, {}) or {}
    meta = obj.get("metadata", {}) or {}
    maxima = obj.get("maxima", {}) or {}
    meta_regions = meta.get("regions", {}) or {}
    for i, region in enumerate(region_names):
        src = None
        for cand in (meta_regions.get(region), maxima.get(region), resolved.get(region)):
            if isinstance(cand, dict):
                src = cand
                break
        if src is None:
            continue
        length[i] = float(src.get("length_max", src.get("length", 0.0)) or 0.0)
        curl[i] = curl_type_to_binary(src)
        valid[i] = 1.0
    return length, curl, valid

# ============================================================================
# DATASET
# ============================================================================
class GroupedImageTextureMeshHairDataset(Dataset):
    """
    Expected folder structure:

    root/
      groupXXX_00001/
        render_images/
          *.png|jpg|jpeg|webp
        textures/
          *.png|jpg|jpeg|webp
        shapes/
          *.obj
        hair_maps_single_512_uvlocal/
          *.pt preferred, fallback *.npz
    """

    EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.webp")

    def __init__(
        self,
        root: str,
        ctrl_size: int = 256,
        texture_size: int = 1024,
        hair_dirname: str = "hair_maps_single_512_uvlocal",
        shape_dirname: str = "shapes",
        guidance_dirname: str = "render_images",
        texture_dirname: str = "textures",
        augment: bool = True,
        recursive: bool = False,
        normalize_mesh_flag: bool = True,
        hair_region_names=None,
    ):
        self.root = str(Path(root).resolve())
        self.ctrl_size = ctrl_size
        self.texture_size = texture_size
        self.hair_dirname = hair_dirname
        self.shape_dirname = shape_dirname
        self.guidance_dirname = guidance_dirname
        self.texture_dirname = texture_dirname
        self.augment = augment
        self.recursive = recursive
        self.normalize_mesh_flag = normalize_mesh_flag
        self.hair_region_names = parse_region_names(hair_region_names)

        root_path = Path(self.root)
        self.dataset_name = root_path.name
        self.species = root_path.parent.name if root_path.parent.name != "" else root_path.name

        self.ctrl_resize = transforms.Resize(
            (ctrl_size, ctrl_size),
            transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.tex_resize = transforms.Resize(
            (texture_size, texture_size),
            transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
        self.jitter = transforms.ColorJitter(0.1, 0.1, 0.05, 0.02)

        self.samples = []
        self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(f"Dataset empty: {self.root}")

        first = self.samples[0]

        verts, faces, _, _ = load_obj(first["mesh"])
        if self.normalize_mesh_flag:
            verts = normalize_verts(verts)

        hair_q, _, _ = load_hair_quantized_pt_or_npz(first["hair"])
        if hair_q.dim() != 4 or hair_q.shape[-1] != 3:
            raise ValueError(f"Expected first hair_q [H,W,S-1,3], got {tuple(hair_q.shape)}")

        self.num_verts = verts.shape[0]
        self.hair_channels = int(hair_q.shape[2] * 3)
        self.hair_size = (int(hair_q.shape[0]), int(hair_q.shape[1]))
        self.mesh_edges = build_edges_from_faces(faces)
        self.mesh_neighbors = build_vertex_neighbors(self.num_verts, self.mesh_edges)

        print(f"\n[Dataset] root={self.root}")
        print(f"[Dataset] species={self.species}")
        print(f"[Dataset] samples={len(self.samples)}")
        print(f"[Dataset] control_size={ctrl_size} texture_size={texture_size}")
        print(f"[Dataset] hair_channels={self.hair_channels} hair_size={self.hair_size}")
        print(f"[Dataset] num_verts={self.num_verts}")
        print(f"[Dataset] mesh_edges={self.mesh_edges.shape[0]}")
        print(f"[Dataset] recursive={recursive}")
        print(f"[Dataset] first_hair={first['hair']}")
        print(f"[Dataset] hair_regions={self.hair_region_names}")

    def _glob_images(self, directory):
        paths = []
        for ext in self.EXTENSIONS:
            paths.extend(glob.glob(os.path.join(directory, ext)))
        return sorted(paths, key=natural_key)

    def _glob_objs(self, directory):
        return sorted(glob.glob(os.path.join(directory, "*.obj")), key=natural_key)

    def _glob_npz(self, directory):
        return sorted(glob.glob(os.path.join(directory, "*.npz")), key=natural_key)

    def _glob_pt(self, directory):
        return sorted(glob.glob(os.path.join(directory, "*.pt")), key=natural_key)

    def _glob_hair(self, directory):
        """
        Prefer .pt cache files. If no .pt exists in this hair directory,
        fallback to original .npz files.
        """
        pt_paths = self._glob_pt(directory)
        if len(pt_paths) > 0:
            return pt_paths

        return self._glob_npz(directory)

    def _pick_aux(self, aux_paths, idx, sample_root, kind):
        if len(aux_paths) == 0:
            raise RuntimeError(f"No {kind} files found in {sample_root}")

        if len(aux_paths) == 1:
            return aux_paths[0]

        if idx < len(aux_paths):
            return aux_paths[idx]

        raise RuntimeError(
            f"{kind} count mismatch in {sample_root}: "
            f"need index {idx}, only {len(aux_paths)} files"
        )

    def _build_index(self):
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        if self.recursive:
            candidate_dirs = []
            for dirpath, dirnames, _ in os.walk(self.root):
                needed = {
                    self.guidance_dirname,
                    self.texture_dirname,
                    self.shape_dirname,
                    self.hair_dirname,
                }
                if needed.issubset(set(dirnames)):
                    candidate_dirs.append(dirpath)
            candidate_dirs = sorted(candidate_dirs, key=natural_key)
        else:
            candidate_dirs = sorted(
                [
                    os.path.join(self.root, d)
                    for d in os.listdir(self.root)
                    if os.path.isdir(os.path.join(self.root, d))
                ],
                key=natural_key,
            )

        n_pt = 0
        n_npz = 0

        for sample_root in candidate_dirs:
            img_dir = os.path.join(sample_root, self.guidance_dirname)
            tex_dir = os.path.join(sample_root, self.texture_dirname)
            shape_dir = os.path.join(sample_root, self.shape_dirname)
            hair_dir = os.path.join(sample_root, self.hair_dirname)

            needed_dirs = [img_dir, tex_dir, shape_dir, hair_dir]
            if not all(os.path.isdir(x) for x in needed_dirs):
                continue

            img_paths = self._glob_images(img_dir)
            tex_paths = self._glob_images(tex_dir)
            mesh_paths = self._glob_objs(shape_dir)
            hair_paths = self._glob_hair(hair_dir)

            if len(img_paths) == 0 or len(tex_paths) == 0:
                continue

            pair_count = min(len(img_paths), len(tex_paths))
            folder_name = os.path.basename(sample_root)
            rel_folder = os.path.relpath(sample_root, self.root).replace(os.sep, "__")

            for pair_idx in range(pair_count):
                image_path = img_paths[pair_idx]
                texture_path = tex_paths[pair_idx]
                mesh_path = self._pick_aux(mesh_paths, pair_idx, sample_root, "mesh")
                hair_path = self._pick_aux(hair_paths, pair_idx, sample_root, "hair")

                if hair_path.endswith(".pt"):
                    n_pt += 1
                elif hair_path.endswith(".npz"):
                    n_npz += 1

                image_stem = Path(image_path).stem
                unique_id = f"{self.species}__{rel_folder}__{image_stem}"
                hair_info_path = os.path.join(sample_root, "hair_info.json")
                object_key = Path(mesh_path).stem

                self.samples.append(
                    dict(
                        control=image_path,
                        target=texture_path,
                        mesh=mesh_path,
                        hair=hair_path,
                        hair_info=hair_info_path if os.path.isfile(hair_info_path) else "",
                        sample_root=sample_root,
                        object_key=object_key,
                        species=self.species,
                        cls=folder_name,
                        id=unique_id,
                        name=f"{self.species}/{folder_name}/{image_stem}",
                        image_name=os.path.basename(image_path),
                        texture_name=os.path.basename(texture_path),
                    )
                )

        print(f"[Dataset index] hair .pt samples={n_pt}, hair .npz samples={n_npz}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import time

        t0 = time.time()
        sample = self.samples[idx]

        # -------------------------
        # Images
        # -------------------------
        ctrl_img = Image.open(sample["control"]).convert("RGB")
        tgt_img = Image.open(sample["target"]).convert("RGB")

        ctrl_01 = self.to_tensor(self.ctrl_resize(ctrl_img))
        tgt_01 = self.to_tensor(self.tex_resize(tgt_img))

        if self.augment:
            ctrl_01 = self.jitter(ctrl_01)

        t_img = time.time()

        # -------------------------
        # Mesh
        # -------------------------
        verts, faces, verts_uvs, faces_uvs = load_obj(sample["mesh"])

        if self.normalize_mesh_flag:
            verts = normalize_verts(verts)

        verts = torch.from_numpy(verts).float()
        faces = torch.from_numpy(faces).long()

        if verts_uvs is not None:
            verts_uvs = torch.from_numpy(verts_uvs).float()
        if faces_uvs is not None:
            faces_uvs = torch.from_numpy(faces_uvs).long()

        t_mesh = time.time()

        # -------------------------
        # Hair
        # -------------------------
        # Keep quantized hair on CPU. The training loop dequantizes on GPU.
        hair_q, hair_scale, hair_mask = load_hair_quantized_pt_or_npz(sample["hair"])
        region_length, region_curl, region_valid = read_hair_info_labels(
            sample.get("hair_info", ""),
            sample.get("object_key", Path(sample["mesh"]).stem),
            self.hair_region_names,
        )

        t_hair = time.time()


        return {
            "control": self.normalize(ctrl_01),
            "target_hr": self.normalize(tgt_01),
            "verts": verts,
            "faces": faces,
            "verts_uvs": verts_uvs,
            "faces_uvs": faces_uvs,
            "hair_q": hair_q,
            "hair_scale": hair_scale,
            "hair_mask": hair_mask,
            "hair_region_length": region_length,
            "hair_region_curl": region_curl,
            "hair_region_valid": region_valid,
            "ctrl_path": sample["control"],
            "tgt_path": sample["target"],
            "mesh_path": sample["mesh"],
            "hair_path": sample["hair"],
            "hair_info_path": sample.get("hair_info", ""),
            "object_key": sample.get("object_key", Path(sample["mesh"]).stem),
            "species": sample["species"],
            "id": sample["id"],
            "name": sample["name"],
            "cls": sample["cls"],
            "image_name": sample["image_name"],
            "texture_name": sample["texture_name"],
            "image_path": sample["control"],
        }


def multitask_collate(batch):
    out = {}
    keys_stack = ["control", "target_hr", "verts", "hair_q", "hair_scale", "hair_mask", "hair_region_length", "hair_region_curl", "hair_region_valid"]
    keys_keep = [
        "ctrl_path", "tgt_path", "mesh_path", "hair_path", "hair_info_path", "object_key",
        "species", "id", "name", "cls", "image_name", "texture_name", "image_path"
    ]

    for k in keys_stack:
        out[k] = torch.stack([b[k] for b in batch], dim=0)

    faces_list = [b["faces"] for b in batch]
    if all(torch.equal(faces_list[0], x) for x in faces_list[1:]):
        out["faces"] = faces_list[0]
    else:
        out["faces"] = torch.stack(faces_list, dim=0)

    def same_or_stack(key):
        vals = [b[key] for b in batch]
        if vals[0] is None:
            return None
        if all(torch.equal(vals[0], x) for x in vals[1:]):
            return vals[0]
        return torch.stack(vals, dim=0)

    out["verts_uvs"] = same_or_stack("verts_uvs")
    out["faces_uvs"] = same_or_stack("faces_uvs")

    for k in keys_keep:
        out[k] = [b[k] for b in batch]

    return out


def make_train_val_subsets(dataset, val_ratio=0.1, seed=42):
    n = len(dataset)
    if n < 2:
        raise ValueError(f"Need at least 2 samples to split, got {n}")

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    print(f"[Split] total={n} train={len(train_subset)} val={len(val_subset)}")
    return train_subset, val_subset


# ============================================================================
# DINO ENCODER
# ============================================================================

class DINOv2Encoder(nn.Module):
    """
    DINOv2 encoder exposing final tokens and intermediate backbone features.

    UV cross-attention hooks are compatibility no-ops; this wrapper does not
    add a separate UV cross-attention module.
    """
    def __init__(
        self,
        model_name="vit_small_patch14_dinov2.lvd142m",
        img_size=256,
        freeze=True,
        unfreeze_last_n_blocks: int = 0,
        use_uv_cross_attn: bool = False,
        n_surface_points: int = 256,
        uv_attn_heads: int = 8,
        uv_attn_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.vit = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=img_size
        )
        self.embed_dim = self.vit.embed_dim
        self.use_uv_cross_attn = bool(use_uv_cross_attn)
        self._template_set = False

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)

        # Optionally unfreeze the final backbone blocks.
        if freeze and int(unfreeze_last_n_blocks) > 0 and hasattr(self.vit, "blocks"):
            for blk in self.vit.blocks[-int(unfreeze_last_n_blocks):]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            if hasattr(self.vit, "norm"):
                for p in self.vit.norm.parameters():
                    p.requires_grad_(True)

    def preprocess(self, x):
        # Convert network inputs from [-1, 1] to the DINO input range [0, 1].
        return (x * 0.5 + 0.5).clamp(0, 1)

    def set_template(self, xyz=None, uv=None):
        # Compatibility hook for callers that provide template coordinates.
        self._template_set = True
        self.template_xyz = xyz
        self.template_uv = uv

    def forward(self, x):
        feats = self.vit.forward_features(self.preprocess(x))
        if isinstance(feats, dict):
            return feats["x_norm_patchtokens"]
        return feats[:, 1:, :]


# ============================================================================
# SHARED RESIDUAL BLOCK / PRETRAINED TEXTURE VAE
# ============================================================================

class ConvResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.norm1 = safe_group_norm(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = safe_group_norm(out_ch)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.drop(self.act2(self.norm2(h))))
        return h + self.skip(x)


class StableDiffusionTextureVAE(nn.Module):
    """Frozen SD 1.x/2.x/SDXL AutoencoderKL with scaled tensor-only latents."""

    def __init__(
        self,
        model_id="stabilityai/sd-vae-ft-mse",
        subfolder=None,
        revision=None,
        local_files_only=False,
        use_slicing=True,
        use_tiling=False,
        local_path=None,
    ):
        super().__init__()
        self.model_id = model_id
        self.subfolder = subfolder or None
        self.revision = revision
        self.use_tiling = bool(use_tiling)
        load_kwargs = {
            "torch_dtype": torch.float32,
            "local_files_only": local_files_only,
        }
        if self.subfolder:
            load_kwargs["subfolder"] = self.subfolder
        if revision:
            load_kwargs["revision"] = revision
        # Load pretrained weights; never fall back to random initialization.
        self.vae = AutoencoderKL.from_pretrained(local_path or model_id, **load_kwargs)
        config = self.vae.config
        self.latent_ch = int(config.latent_channels)
        self.downsample_factor = 2 ** (len(config.block_out_channels) - 1)
        self.scaling_factor = float(config.scaling_factor)
        if (
            (config.in_channels, config.out_channels, self.latent_ch,
             self.downsample_factor) != (3, 3, 4, 8)
            or getattr(config, "shift_factor", None) not in (None, 0.0)
            or getattr(config, "latents_mean", None) is not None
            or getattr(config, "latents_std", None) is not None
        ):
            raise ValueError("Use an RGB, 4-channel, 8x SD 1.x/2.x/SDXL VAE.")
        if not math.isfinite(self.scaling_factor) or self.scaling_factor <= 0:
            raise ValueError("The VAE scaling_factor must be finite and positive.")
        if use_slicing:
            self.vae.enable_slicing()
        if use_tiling:
            self.vae.enable_tiling()
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Keep the pretrained VAE frozen/eval even if a caller invokes train().
        return super().train(False)

    def checkpoint_config(self):
        return {
            "type": "stable_diffusion_autoencoderkl",
            "model_id": self.model_id,
            "subfolder": self.subfolder,
            "revision": self.revision,
            "latent_ch": self.latent_ch,
            "downsample_factor": self.downsample_factor,
            "scaling_factor": self.scaling_factor,
            "use_tiling": self.use_tiling,
        }

    @torch.no_grad()
    def encode(self, x, sample_posterior=False):
        """RGB in [-1,1] -> scaled [B,4,H/8,W/8] latent, in float32."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected RGB [B,3,H,W], got {tuple(x.shape)}")
        if any(s % self.downsample_factor for s in x.shape[-2:]):
            raise ValueError("Texture height and width must be divisible by 8.")
        # Dataset normalization already produces [-1,1]; do not normalize twice.
        # Explicitly disable outer AMP to avoid fp16 overflow in the VAE.
        with torch.autocast(device_type=x.device.type, enabled=False):
            posterior = self.vae.encode(x.float()).latent_dist
            z = posterior.sample() if sample_posterior else posterior.mode()
            return z.float() * self.scaling_factor

    @torch.no_grad()
    def decode(self, z):
        """Scaled latent -> decoded RGB; image saving maps [-1,1] to [0,1]."""
        if z.ndim != 4 or z.shape[1] != self.latent_ch:
            raise ValueError(
                f"Expected [B,{self.latent_ch},H,W], got {tuple(z.shape)}"
            )
        with torch.autocast(device_type=z.device.type, enabled=False):
            # AutoencoderKL.decode does not undo latent scaling itself.
            return self.vae.decode(z.float() / self.scaling_factor).sample.float()

    def forward(self, x, sample_posterior=False):
        z = self.encode(x, sample_posterior=sample_posterior)
        return self.decode(z), z


def build_texture_vae(args, device):
    vae = StableDiffusionTextureVAE(
        model_id=args.vae_model,
        subfolder=args.vae_subfolder,
        revision=args.vae_revision,
        local_files_only=args.vae_local_files_only,
        use_slicing=args.vae_slicing,
        use_tiling=args.vae_tiling,
        local_path=getattr(args, "vae_local_path", None),
    ).to(device=device, dtype=torch.float32)
    if args.texture_size <= 0 or args.texture_size % vae.downsample_factor:
        raise ValueError("--texture_size must be positive and divisible by 8.")
    latent_size = args.texture_size // vae.downsample_factor
    if args.latent_ch not in (None, vae.latent_ch):
        raise ValueError(
            f"--latent_ch={args.latent_ch} conflicts with the pretrained VAE "
            f"({vae.latent_ch}); omit --latent_ch to infer it."
        )
    if args.latent_size not in (None, latent_size):
        raise ValueError(
            f"--texture_size={args.texture_size} requires --latent_size={latent_size}, "
            f"got {args.latent_size}; omit --latent_size to infer it."
        )
    if not args.flow_ch_mult or any(c <= 0 for c in args.flow_ch_mult):
        raise ValueError("--flow_ch_mult must contain positive channel multipliers.")
    flow_downsample = 2 ** (len(args.flow_ch_mult) - 1)
    if latent_size % flow_downsample:
        raise ValueError(
            f"Latent size {latent_size} must be divisible by {flow_downsample} "
            "for the flow UNet skip connections."
        )
    args.latent_ch = vae.latent_ch
    args.latent_size = latent_size
    print(
        f"[VAE] {args.vae_model}: frozen float32, "
        f"texture={args.texture_size}, latent=[{args.latent_ch},"
        f"{args.latent_size},{args.latent_size}], scaling={vae.scaling_factor}"
    )
    return vae


# Recover architecture and latent representation before model construction.
# Runtime paths, optimizer settings, sampling CFG/NFE, and memory slicing stay CLI-controlled.
CHECKPOINT_MODEL_ARGS = (
    "vae_model", "vae_subfolder", "vae_revision", "vae_tiling",
    "vae_sample_posterior", "texture_size", "latent_ch", "latent_size",
    "ctrl_size", "flow_base_ch", "flow_ch_mult", "flow_attn_levels",
    "freeze_encoder", "unfreeze_last_n_blocks", "use_uv_cross_attn",
    "n_surface_points", "uv_attn_heads", "uv_attn_dropout",
    "use_color_embed", "use_dino_intermediate_tokens", "dino_intermediate_layers",
    "dino_intermediate_grid", "freeze_dino_intermediate", "mesh_hidden_dim",
    "hair_hidden_dim", "head_dropout", "mesh_head_depth", "hair_head_depth",
    "hair_region_names",
)


def restore_checkpoint_model_args(args, ckpt):
    if ckpt.get("vae_config", {}).get("type") != "stable_diffusion_autoencoderkl":
        raise ValueError(
            "This checkpoint uses the old custom AE or lacks SD-VAE metadata. "
            "Its texture latent space is incompatible. Start a new training run "
            "without --resume, then use the new checkpoint for inference."
        )
    saved_args = ckpt.get("args", {})
    for name in CHECKPOINT_MODEL_ARGS:
        if name not in saved_args:
            raise ValueError(f"SD-VAE checkpoint is missing model setting: {name}")
        setattr(args, name, saved_args[name])
    print("[Checkpoint] Restored model architecture and VAE settings.")


def validate_checkpoint_vae(ckpt, vae):
    if ckpt["vae_config"] != vae.checkpoint_config():
        raise ValueError(
            "The loaded VAE does not match the checkpoint's VAE configuration. "
            "Use the same pretrained model, revision, and tiling settings."
        )


# ============================================================================
# FLOW MODEL
# ============================================================================

def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = safe_group_norm(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))

        self.norm2 = safe_group_norm(out_ch)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        scale, shift = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(self.act2(h)))
        return h + self.shortcut(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=8, head_dim=64):
        super().__init__()
        num_heads = max(1, min(num_heads, query_dim // max(1, head_dim)))
        inner = num_heads * head_dim
        self.nh = num_heads
        self.hd = head_dim

        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.out = nn.Linear(inner, query_dim)

    def forward(self, x, ctx):
        B, HW, _ = x.shape
        h = self.nh
        q = self.to_q(x).reshape(B, HW, h, self.hd).permute(0, 2, 1, 3)
        k = self.to_k(ctx).reshape(B, ctx.shape[1], h, self.hd).permute(0, 2, 1, 3)
        v = self.to_v(ctx).reshape(B, ctx.shape[1], h, self.hd).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.permute(0, 2, 1, 3).reshape(B, HW, -1)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, ch, context_dim, num_heads=8):
        super().__init__()
        head_dim = max(32, ch // num_heads)
        num_heads = max(1, ch // head_dim)

        self.in_norm = safe_group_norm(ch)
        self.norm_sa = nn.LayerNorm(ch)
        self.self_attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

        self.norm_ca = nn.LayerNorm(ch)
        self.cross_attn = CrossAttention(ch, context_dim, num_heads, head_dim)

        self.norm_ff = nn.LayerNorm(ch)
        self.ff = nn.Sequential(nn.Linear(ch, ch * 4), nn.GELU(), nn.Linear(ch * 4, ch))

    def forward(self, x, ctx):
        B, C, H, W = x.shape
        res = x
        h = self.in_norm(x).reshape(B, C, H * W).permute(0, 2, 1)

        hn = self.norm_sa(h)
        sa, _ = self.self_attn(hn, hn, hn, need_weights=False)
        h = h + sa

        h = h + self.cross_attn(self.norm_ca(h), ctx)
        h = h + self.ff(self.norm_ff(h))

        return h.permute(0, 2, 1).reshape(B, C, H, W) + res


class UNetFlowModel(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, base_ch=128, ch_mult=(1, 2, 4, 4),
                 context_dim=384, time_dim=256, dropout=0.1, attn_levels=(2, 3)):
        super().__init__()
        self.time_dim = time_dim
        self.attn_levels = set(attn_levels)
        channels = [base_ch * m for m in ch_mult]
        num_levels = len(channels)
        time_emb_dim = time_dim * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.ctx_proj = nn.Linear(context_dim, context_dim)

        self.enc_init = nn.Conv2d(in_ch, channels[0], 3, padding=1)
        self.enc_res1 = nn.ModuleList()
        self.enc_res2 = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.downs = nn.ModuleList()

        prev = channels[0]
        for i, ch in enumerate(channels):
            self.enc_res1.append(ResBlock(prev, ch, time_emb_dim, dropout))
            self.enc_res2.append(ResBlock(ch, ch, time_emb_dim, dropout))
            self.enc_attn.append(TransformerBlock(ch, context_dim) if i in self.attn_levels else nn.Identity())
            self.downs.append(
                nn.Conv2d(ch, ch, 4, stride=2, padding=1) if i < num_levels - 1 else nn.Identity()
            )
            prev = ch

        bot_ch = channels[-1]
        self.bot_res1 = ResBlock(bot_ch, bot_ch, time_emb_dim, dropout)
        self.bot_attn = TransformerBlock(bot_ch, context_dim)
        self.bot_res2 = ResBlock(bot_ch, bot_ch, time_emb_dim, dropout)

        self.dec_res1 = nn.ModuleList()
        self.dec_res2 = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        self.ups = nn.ModuleList()

        h_ch = bot_ch
        for i in range(num_levels):
            enc_level = num_levels - 1 - i
            skip_ch = channels[enc_level]
            cat_ch = h_ch + skip_ch
            dec_out_ch = channels[enc_level]

            self.dec_res1.append(ResBlock(cat_ch, dec_out_ch, time_emb_dim, dropout))
            self.dec_res2.append(ResBlock(dec_out_ch, dec_out_ch, time_emb_dim, dropout))
            self.dec_attn.append(
                TransformerBlock(dec_out_ch, context_dim) if enc_level in self.attn_levels else nn.Identity()
            )

            if i < num_levels - 1:
                next_ch = channels[enc_level - 1]
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(dec_out_ch, next_ch, 3, padding=1),
                ))
                h_ch = next_ch
            else:
                self.ups.append(nn.Identity())
                h_ch = dec_out_ch

        self.out_norm = safe_group_norm(channels[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(channels[0], out_ch, 3, padding=1)

    def forward(self, x, t, ctx):
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        ctx = self.ctx_proj(ctx)

        h = self.enc_init(x)
        skips = []

        for r1, r2, attn, down in zip(self.enc_res1, self.enc_res2, self.enc_attn, self.downs):
            h = r1(h, t_emb)
            h = r2(h, t_emb)
            if isinstance(attn, TransformerBlock):
                h = attn(h, ctx)
            skips.append(h)
            h = down(h)

        h = self.bot_res1(h, t_emb)
        h = self.bot_attn(h, ctx)
        h = self.bot_res2(h, t_emb)

        for i, (r1, r2, attn, up) in enumerate(zip(self.dec_res1, self.dec_res2, self.dec_attn, self.ups)):
            h = torch.cat([h, skips[-(i + 1)]], dim=1)
            h = r1(h, t_emb)
            h = r2(h, t_emb)
            if isinstance(attn, TransformerBlock):
                h = attn(h, ctx)
            h = up(h)

        return self.out_conv(self.out_act(self.out_norm(h)))


# ============================================================================
# TOKEN / HEAD BUILDING BLOCKS
# ============================================================================

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class TokenMixerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim, mult=4, dropout=dropout)

    def forward(self, x):
        xn = self.norm(x)
        attn_out, _ = self.attn(xn, xn, xn, need_weights=False)
        x = x + attn_out
        x = self.ff(x)
        return x


class AttentivePool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        pooled = (x * w).sum(dim=1)
        return pooled


class MeshTokenHead(nn.Module):
    """
    Stronger than global mean pooling:
    - token mixing over patch tokens
    - attentive global pooling
    - deep residual MLP to per-vertex offsets
    """
    def __init__(self, in_dim=384, hidden_dim=1024, num_verts=5023, depth=2, dropout=0.1):
        super().__init__()
        self.num_verts = num_verts
        self.token_blocks = nn.ModuleList([TokenMixerBlock(in_dim, num_heads=8, dropout=dropout) for _ in range(depth)])
        self.pool = AttentivePool(in_dim)

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_verts * 3),
        )

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tokens):
        x = tokens
        for blk in self.token_blocks:
            x = blk(x)
        pooled = self.pool(x)
        pred = self.mlp(pooled)
        return pred.view(tokens.shape[0], self.num_verts, 3)


class HairDecoderResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = ConvResBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x):
        return self.block(x)


class HairUNetHead(nn.Module):
    """
    Stronger hair head:
    - token mixing in sequence space
    - learned 2D stem from patch tokens
    - progressive upsampling decoder with residual blocks
    """
    def __init__(
        self,
        in_dim=384,
        stem_dim=256,
        decoder_dims=(256, 192, 128, 96, 64),
        hair_out_channels=63,
        dropout=0.1,
        hair_size=(512, 512),
        token_depth=2,
        num_regions: int = 0,
    ):
        super().__init__()
        self.hair_size = tuple(hair_size)
        self.num_regions = int(num_regions)
        self.token_blocks = nn.ModuleList([TokenMixerBlock(in_dim, num_heads=8, dropout=dropout) for _ in range(token_depth)])
        self.token_norm = nn.LayerNorm(in_dim)
        self.stem = nn.Conv2d(in_dim, stem_dim, 3, padding=1)
        if self.num_regions > 0:
            self.aux_pool = AttentivePool(in_dim)
            self.region_len_head = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(in_dim, self.num_regions))
            self.region_curl_head = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(in_dim, self.num_regions))
        else:
            self.aux_pool = None
            self.region_len_head = None
            self.region_curl_head = None

        dims = [stem_dim] + list(decoder_dims)
        self.ups = nn.ModuleList()
        self.res1 = nn.ModuleList()
        self.res2 = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dims[i], dims[i + 1], 3, padding=1),
            ))
            self.res1.append(HairDecoderResBlock(dims[i + 1], dims[i + 1], dropout=dropout))
            self.res2.append(HairDecoderResBlock(dims[i + 1], dims[i + 1], dropout=dropout))

        self.out = nn.Sequential(
            safe_group_norm(dims[-1]),
            nn.SiLU(),
            nn.Conv2d(dims[-1], hair_out_channels, 3, padding=1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, tokens):
        x = tokens
        for blk in self.token_blocks:
            x = blk(x)
        x = self.token_norm(x)

        pred_region_len = None
        pred_region_curl_logits = None
        if self.num_regions > 0:
            pooled = self.aux_pool(x)
            pred_region_len = F.softplus(self.region_len_head(pooled))
            pred_region_curl_logits = self.region_curl_head(pooled)

        B, T, D = x.shape
        patch_hw = int(math.sqrt(T))
        if patch_hw * patch_hw != T:
            raise ValueError(f"Expected square patch-token count, got {T}")

        x = x.transpose(1, 2).reshape(B, D, patch_hw, patch_hw)
        x = self.stem(x)

        for up, r1, r2 in zip(self.ups, self.res1, self.res2):
            x = up(x)
            x = r1(x)
            x = r2(x)

        if tuple(x.shape[-2:]) != self.hair_size:
            x = F.interpolate(x, size=self.hair_size, mode="bilinear", align_corners=False)

        return {
            "pred_hair": self.out(x),
            "pred_region_len": pred_region_len,
            "pred_region_curl_logits": pred_region_curl_logits,
        }


# ============================================================================
# COLOUR EMBEDDING
# ============================================================================

class GlobalColorEncoder(nn.Module):
    """
    Explicit colour statistics token — prepended to ctx before the UNet.
    Captures absolute colour information that DINOv2 is weak at encoding.
    """
    def __init__(self, embed_dim: int, n_bins: int = 16):
        super().__init__()
        self.n_bins = n_bins
        raw_dim = 3 * n_bins + 6 + 15
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        centers = torch.linspace(0, 1, self.n_bins, device=x.device)
        width   = 1.0 / self.n_bins
        x_flat  = x.flatten(1).unsqueeze(2)
        dist    = (x_flat - centers).abs() / width
        weights = (1 - dist).clamp(0, 1)
        hist    = weights.mean(dim=1)
        return hist / (hist.sum(dim=1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] in [-1,1] → [B, embed_dim]"""
        x01   = x * 0.5 + 0.5
        feats = []
        for c in range(3):
            feats.append(self._soft_histogram(x01[:, c]))
        feats.append(x01.mean(dim=[2, 3]))
        feats.append(x01.std(dim=[2, 3]))
        q  = x01.flatten(2)
        qs = torch.quantile(
            q,
            torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=x.device),
            dim=2,
        )
        feats.append(qs.permute(1, 2, 0).flatten(1))
        return self.proj(torch.cat(feats, dim=1))




class DINOIntermediateTokenExtractor(nn.Module):
    """Reuse the backbone inside DINOv2Encoder and expose intermediate ViT tokens."""
    def __init__(self, owner_encoder: nn.Module, embed_dim: int, layers: tuple = (3, 7), grid_size: int = 16, freeze: bool = True):
        super().__init__()
        self.owner_encoder = owner_encoder
        self.layers = tuple(int(x) for x in layers)
        self.grid_size = int(grid_size)
        self.embed_dim = int(embed_dim)
        self.freeze = bool(freeze)
        self.backbone = self._find_intermediate_backbone(owner_encoder)
        if self.backbone is None:
            raise RuntimeError(
                "Could not find a DINOv2 backbone with get_intermediate_layers() inside DINOv2Encoder. "
                "Expose it as self.model/self.backbone/self.dino, or disable --use_dino_intermediate_tokens."
            )
        self.proj = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
            for _ in self.layers
        ])

    @staticmethod
    def _find_intermediate_backbone(module: nn.Module):
        for name in ("model", "backbone", "dinov2", "dino", "vit", "net"):
            child = getattr(module, name, None)
            if child is not None and hasattr(child, "get_intermediate_layers"):
                return child
        for _, child in module.named_children():
            if child is module:
                continue
            if hasattr(child, "get_intermediate_layers"):
                return child
            for _, grandchild in child.named_children():
                if hasattr(grandchild, "get_intermediate_layers"):
                    return grandchild
        return None

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        for name in ("preprocess", "_preprocess", "prepare_input", "_prepare_input"):
            fn = getattr(self.owner_encoder, name, None)
            if callable(fn):
                return fn(x)
        x01 = (x * 0.5 + 0.5).clamp(0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x01 - mean) / std

    @staticmethod
    def _maybe_drop_cls(tokens: torch.Tensor) -> torch.Tensor:
        """Drop a leading CLS/register token if the remaining sequence forms a square patch grid."""
        if tokens.ndim != 3:
            return tokens
        B, N, D = tokens.shape
        side = int(round(math.sqrt(N)))
        if side * side == N:
            return tokens
        side_m1 = int(round(math.sqrt(max(N - 1, 0))))
        if side_m1 * side_m1 == N - 1:
            return tokens[:, 1:, :].contiguous()
        return tokens

    @staticmethod
    def _tokens_from_output(out):
        # Newer DINOv2 may return (patch_tokens, cls_token) when return_class_token=True.
        # Older DINOv2 usually returns only patch tokens and does not accept return_class_token.
        if isinstance(out, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "tokens", "x"):
                if key in out:
                    out = out[key]
                    break
        elif isinstance(out, (tuple, list)):
            tensor_candidates = [z for z in out if torch.is_tensor(z)]
            if len(tensor_candidates) == 0:
                raise RuntimeError("Unsupported DINO intermediate output tuple/list.")
            # Prefer a 3D token tensor with the longest token dimension. This avoids selecting CLS [B,D].
            out = max(tensor_candidates, key=lambda z: z.shape[1] if z.ndim >= 3 else 0)
        if not torch.is_tensor(out):
            raise RuntimeError(f"Unsupported DINO intermediate output type: {type(out)}")
        if out.ndim == 4:
            out = out.flatten(2).transpose(1, 2).contiguous()
        if out.ndim != 3:
            raise RuntimeError(f"Expected DINO tokens [B,N,D], got shape {tuple(out.shape)}")
        return DINOIntermediateTokenExtractor._maybe_drop_cls(out)

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.grid_size <= 0:
            return tokens
        B, N, D = tokens.shape
        h = int(round(math.sqrt(N)))
        if h * h != N:
            return tokens
        feat = tokens.transpose(1, 2).reshape(B, D, h, h)
        feat = F.adaptive_avg_pool2d(feat, (self.grid_size, self.grid_size))
        return feat.flatten(2).transpose(1, 2).contiguous()

    def _get_intermediate_layers_compat(self, x_in: torch.Tensor):
        """Call DINOv2 get_intermediate_layers across multiple API versions."""
        # Newer DINOv2 API.
        try:
            return self.backbone.get_intermediate_layers(
                x_in,
                n=list(self.layers),
                reshape=False,
                return_class_token=False,
            )
        except TypeError as e:
            if "return_class_token" not in str(e):
                raise

        # Older DINOv2 API: no return_class_token kwarg.
        # Most older versions already strip CLS internally when reshape=False.
        try:
            return self.backbone.get_intermediate_layers(
                x_in,
                n=list(self.layers),
                reshape=False,
            )
        except TypeError:
            # Very old/simplified API may only accept integer n.
            max_layer = max(self.layers)
            outs = self.backbone.get_intermediate_layers(x_in, n=max_layer + 1)
            if not isinstance(outs, (tuple, list)):
                outs = [outs]
            selected = []
            for layer in self.layers:
                idx = min(max(layer, 0), len(outs) - 1)
                selected.append(outs[idx])
            return selected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self._prepare_input(x)
        grad_enabled = torch.is_grad_enabled() and not self.freeze
        with torch.set_grad_enabled(grad_enabled):
            outs = self._get_intermediate_layers_compat(x_in)
        if not isinstance(outs, (tuple, list)):
            outs = [outs]
        toks = []
        for i, out in enumerate(outs):
            t = self._tokens_from_output(out)
            t = self._pool_tokens(t)
            toks.append(self.proj[i](t))
        return torch.cat(toks, dim=1)




# ============================================================================
# MULTITASK MODEL
# ============================================================================

class AnimalTextureMeshHairFlowModel(nn.Module):
    """
    Multitask model for texture, mesh, and hair prediction.

    Texture path is now:
      control image -> DINO final tokens
                    + GlobalColorEncoder token
                    + DINO intermediate tokens
                    -> UNet flow in scaled Stable Diffusion VAE latent space
                    -> optional CFG sampling

    Mesh/hair heads remain attached to the clean final DINO tokens so geometry is
    not damaged by classifier-free condition dropout.
    """
    def __init__(
        self,
        ctrl_img_size=256,
        latent_size=128,
        latent_ch=4,
        flow_base_ch=128,
        flow_ch_mult=(1, 2, 4, 4),
        freeze_encoder=True,
        flow_attn_levels=(2, 3),
        num_verts=5023,
        hair_out_channels=63,
        hair_size=(512, 512),
        mesh_hidden_dim=1024,
        hair_hidden_dim=256,
        head_dropout=0.1,
        mesh_head_depth=2,
        hair_head_depth=2,
        unfreeze_last_n_blocks: int = 2,
        use_uv_cross_attn: bool = False,
        n_surface_points: int = 256,
        uv_attn_heads: int = 8,
        uv_attn_dropout: float = 0.1,
        use_color_embed: bool = True,
        use_dino_intermediate_tokens: bool = True,
        dino_intermediate_layers=(3, 7),
        dino_intermediate_grid: int = 16,
        freeze_dino_intermediate: bool = True,
        num_hair_regions: int = 0,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.latent_ch = latent_ch
        self.use_color_embed = bool(use_color_embed)
        self.use_dino_intermediate_tokens = bool(use_dino_intermediate_tokens)

        self.encoder = DINOv2Encoder(
            img_size=ctrl_img_size,
            freeze=freeze_encoder,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            use_uv_cross_attn=use_uv_cross_attn,
            n_surface_points=n_surface_points,
            uv_attn_heads=uv_attn_heads,
            uv_attn_dropout=uv_attn_dropout,
        )
        ctx_dim = self.encoder.embed_dim

        if self.use_color_embed:
            self.color_encoder = GlobalColorEncoder(embed_dim=ctx_dim)

        if self.use_dino_intermediate_tokens:
            self.dino_intermediate = DINOIntermediateTokenExtractor(
                owner_encoder=self.encoder,
                embed_dim=ctx_dim,
                layers=tuple(dino_intermediate_layers),
                grid_size=dino_intermediate_grid,
                freeze=freeze_dino_intermediate,
            )

        self.unet = UNetFlowModel(
            in_ch=latent_ch,
            out_ch=latent_ch,
            base_ch=flow_base_ch,
            ch_mult=flow_ch_mult,
            context_dim=ctx_dim,
            time_dim=256,
            dropout=0.1,
            attn_levels=flow_attn_levels,
        )

        self.vertex_head = MeshTokenHead(
            in_dim=ctx_dim,
            hidden_dim=mesh_hidden_dim,
            num_verts=num_verts,
            depth=mesh_head_depth,
            dropout=head_dropout,
        )
        self.hair_head = HairUNetHead(
            in_dim=ctx_dim,
            stem_dim=hair_hidden_dim,
            decoder_dims=(
                hair_hidden_dim,
                max(hair_hidden_dim // 2 + hair_hidden_dim // 4, 96),
                max(hair_hidden_dim // 2, 96),
                96,
                64,
            ),
            hair_out_channels=hair_out_channels,
            dropout=head_dropout,
            hair_size=hair_size,
            token_depth=hair_head_depth,
            num_regions=num_hair_regions,
        )

    def encode_control_base(self, ctrl):
        """Final DINO patch tokens, used by mesh/hair heads."""
        return self.encoder(ctrl)

    def encode_control(self, ctrl, final_tokens: Optional[torch.Tensor] = None):
        """
        Code2-style texture context for the latent flow UNet.

        If final_tokens are provided, reuse them instead of running the DINO
        encoder again. This avoids the old forward_multitask() path computing
        self.encoder(ctrl) twice per training step.
        """
        if final_tokens is None:
            final_tokens = self.encode_control_base(ctrl)

        ctx_tokens = []
        if self.use_color_embed:
            ctx_tokens.append(self.color_encoder(ctrl).unsqueeze(1))
        if self.use_dino_intermediate_tokens:
            ctx_tokens.append(self.dino_intermediate(ctrl))
        ctx_tokens.append(final_tokens)
        return torch.cat(ctx_tokens, dim=1)

    @staticmethod
    def drop_condition(ctx: torch.Tensor, drop_prob: float) -> torch.Tensor:
        """Per-sample condition dropout for classifier-free guidance training."""
        if drop_prob <= 0:
            return ctx
        keep = (torch.rand(ctx.shape[0], 1, 1, device=ctx.device) >= drop_prob).to(ctx.dtype)
        return ctx * keep

    def forward_flow(self, x_t, t, ctx):
        return self.unet(x_t, t, ctx)

    def forward_heads(self, base_ctx):
        pred_offsets = self.vertex_head(base_ctx)
        hair_out = self.hair_head(base_ctx)
        return pred_offsets, hair_out

    def forward_multitask(self, x_t, t, ctrl, cond_drop_prob: float = 0.0):
        # Compute final DINO patch tokens once. Reuse them both for the texture
        # context and for the mesh/hair heads.
        base_ctx = self.encode_control_base(ctrl)

        tex_ctx = self.encode_control(ctrl, final_tokens=base_ctx)
        flow_ctx = self.drop_condition(tex_ctx, cond_drop_prob)
        v_pred = self.forward_flow(x_t, t, flow_ctx)

        # Keep geometry heads on clean final DINO tokens, not CFG-dropped texture ctx.
        pred_offsets, hair_out = self.forward_heads(base_ctx)
        return {
            "v_pred": v_pred,
            "pred_offsets": pred_offsets,
            "pred_hair": hair_out["pred_hair"],
            "pred_region_len": hair_out.get("pred_region_len"),
            "pred_region_curl_logits": hair_out.get("pred_region_curl_logits"),
        }

    @torch.no_grad()
    def _sample_from_ctx(self, ctx: torch.Tensor, nfe: int = 50, cfg_scale: float = 1.0) -> torch.Tensor:
        device = ctx.device
        B = ctx.shape[0]
        x = torch.randn(B, self.latent_ch, self.latent_size, self.latent_size, device=device)
        dt = 1.0 / nfe
        use_cfg = abs(float(cfg_scale) - 1.0) > 1e-6
        null_ctx = torch.zeros_like(ctx) if use_cfg else None

        for i in range(nfe):
            t_ = torch.full((B,), i * dt, device=device)
            if use_cfg:
                v_cond = self.forward_flow(x, t_, ctx)
                v_uncond = self.forward_flow(x, t_, null_ctx)
                v = v_uncond + float(cfg_scale) * (v_cond - v_uncond)
            else:
                v = self.forward_flow(x, t_, ctx)
            x = x + v * dt
        return x

    @torch.no_grad()
    def sample_texture_latent(self, ctrl: torch.Tensor, nfe: int = 50, cfg_scale: float = 1.0) -> torch.Tensor:
        ctx = self.encode_control(ctrl)
        return self._sample_from_ctx(ctx, nfe=nfe, cfg_scale=cfg_scale)

    @torch.no_grad()
    def infer_all(self, ctrl: torch.Tensor, nfe: int = 50, cfg_scale: float = 1.0):
        # Same reuse path as training: final DINO tokens are shared by texture
        # context and mesh/hair heads.
        base_ctx = self.encode_control_base(ctrl)
        tex_ctx = self.encode_control(ctrl, final_tokens=base_ctx)
        z = self._sample_from_ctx(tex_ctx, nfe=nfe, cfg_scale=cfg_scale)
        pred_offsets, hair_out = self.forward_heads(base_ctx)
        return {
            "z_pred": z,
            "pred_offsets": pred_offsets,
            "pred_hair": hair_out["pred_hair"],
            "pred_region_len": hair_out.get("pred_region_len"),
            "pred_region_curl_logits": hair_out.get("pred_region_curl_logits"),
        }


# ============================================================================
# LOSSES
# ============================================================================

def masked_l1(x, y, mask, fg_weight=5.0):
    abs_err = (x - y).abs()
    pixel_weight = 1.0 + (fg_weight - 1.0) * mask
    return (abs_err * pixel_weight).sum() / (pixel_weight.sum() * x.shape[1])


def masked_mse(x, y, mask, fg_weight=5.0):
    sq_err = (x - y) ** 2
    pixel_weight = 1.0 + (fg_weight - 1.0) * mask
    return (sq_err * pixel_weight).sum() / (pixel_weight.sum() * x.shape[1])






def weighted_masked_l1(pred, target, mask, fg_weight=5.0, bg_weight=0.0):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    weight = bg_weight + mask * (fg_weight - bg_weight)
    loss_map = (pred - target).abs() * weight
    denom = weight.sum().clamp(min=1.0)
    return loss_map.sum() / denom


def weighted_masked_mse(pred, target, mask, fg_weight=5.0, bg_weight=0.0):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    weight = bg_weight + mask * (fg_weight - bg_weight)
    loss_map = (pred - target) ** 2 * weight
    denom = weight.sum().clamp(min=1.0)
    return loss_map.sum() / denom


def mesh_edge_loss(pred_verts, gt_verts, edges):
    e0 = edges[:, 0]
    e1 = edges[:, 1]
    pred_edges = pred_verts[:, e0] - pred_verts[:, e1]
    gt_edges = gt_verts[:, e0] - gt_verts[:, e1]
    return F.l1_loss(pred_edges, gt_edges)


def mesh_laplacian_smoothness_loss(pred_verts, template_verts, neighbors):
    device = pred_verts.device
    num_verts = pred_verts.shape[1]

    lap_pred = []
    lap_template = []
    for i in range(num_verts):
        nbrs = neighbors[i]
        if len(nbrs) == 0:
            continue
        nbr_idx = torch.tensor(nbrs, dtype=torch.long, device=device)
        pred_center = pred_verts[:, i]
        pred_nbr_mean = pred_verts[:, nbr_idx].mean(dim=1)
        lap_pred.append(pred_center - pred_nbr_mean)

        temp_center = template_verts[i].unsqueeze(0).expand(pred_verts.shape[0], -1)
        temp_nbr_mean = template_verts[nbr_idx].mean(dim=0, keepdim=True).expand(pred_verts.shape[0], -1)
        lap_template.append(temp_center - temp_nbr_mean)

    if len(lap_pred) == 0:
        return pred_verts.new_tensor(0.0)

    lap_pred = torch.stack(lap_pred, dim=1)
    lap_template = torch.stack(lap_template, dim=1)
    return F.l1_loss(lap_pred, lap_template)


def mesh_offset_reg_loss(pred_offsets):
    return pred_offsets.abs().mean()


def masked_region_smooth_l1(pred, target, valid, beta=0.1):
    if pred is None:
        return target.new_tensor(0.0)
    valid = valid.to(dtype=pred.dtype)
    target = target.to(dtype=pred.dtype)
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp(min=1.0)

def masked_region_bce_with_logits(logits, target, valid):
    if logits is None:
        return target.new_tensor(0.0)
    valid = valid.to(dtype=logits.dtype)
    target = target.to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp(min=1.0)

def mesh_hair_losses(
    pred_offsets,
    template_verts,
    verts_gt,
    pred_hair,
    hair_gt,
    hair_mask,
    mesh_edges,
    mesh_neighbors,
    mesh_loss_type="l1",
    hair_l1_weight=1.0,
    hair_mse_weight=0.0,
    hair_fg_weight=5.0,
    hair_bg_weight=0.0,
    total_hair_weight=1.0,
    mesh_edge_weight=0.1,
    mesh_laplacian_weight=0.05,
    mesh_offset_reg_weight=0.001,
    pred_region_len=None,
    gt_region_len=None,
    pred_region_curl_logits=None,
    gt_region_curl=None,
    hair_region_valid=None,
    hair_region_length_scale=0.05,
    region_len_weight=0.0,
    region_curl_weight=0.0,
):
    verts_pred = template_verts.unsqueeze(0) + pred_offsets

    if mesh_loss_type == "l2":
        mesh_loss = F.mse_loss(verts_pred, verts_gt)
    else:
        mesh_loss = F.l1_loss(verts_pred, verts_gt)

    mesh_edge = mesh_edge_loss(verts_pred, verts_gt, mesh_edges)
    mesh_laplacian = mesh_laplacian_smoothness_loss(verts_pred, template_verts, mesh_neighbors)
    mesh_offset_reg = mesh_offset_reg_loss(pred_offsets)

    pred_hair = pred_hair * hair_mask
    hair_offset_loss = (
        hair_l1_weight * weighted_masked_l1(pred_hair, hair_gt, hair_mask, fg_weight=hair_fg_weight, bg_weight=hair_bg_weight)
        + hair_mse_weight * weighted_masked_mse(pred_hair, hair_gt, hair_mask, fg_weight=hair_fg_weight, bg_weight=hair_bg_weight)
    )
    region_len_loss = pred_hair.new_tensor(0.0)
    region_curl_loss = pred_hair.new_tensor(0.0)
    if gt_region_len is not None and hair_region_valid is not None:
        gt_len_norm = (gt_region_len / float(max(hair_region_length_scale, 1e-8))).clamp(min=0.0)
        region_len_loss = masked_region_smooth_l1(pred_region_len, gt_len_norm, hair_region_valid, beta=0.1)
        if gt_region_curl is not None:
            region_curl_loss = masked_region_bce_with_logits(pred_region_curl_logits, gt_region_curl, hair_region_valid)

    hair_loss = (
        hair_offset_loss
        + float(region_len_weight) * region_len_loss
        + float(region_curl_weight) * region_curl_loss
    )

    mesh_total = (
        mesh_loss
        + mesh_edge_weight * mesh_edge
        + mesh_laplacian_weight * mesh_laplacian
        + mesh_offset_reg_weight * mesh_offset_reg
    )

    total = mesh_total + total_hair_weight * hair_loss
    return {
        "total": total,
        "mesh_base": mesh_loss,
        "mesh_edge": mesh_edge,
        "mesh_laplacian": mesh_laplacian,
        "mesh_offset_reg": mesh_offset_reg,
        "hair": hair_loss,
        "hair_offset": hair_offset_loss,
        "hair_region_len": region_len_loss,
        "hair_region_curl": region_curl_loss,
        "verts_pred": verts_pred,
        "pred_hair": pred_hair,
    }


def multitask_flow_loss(
    v_pred,
    target,
    mask,
    v_gt,
    fg_weight=5.0,
):
    sq_err = (v_pred - v_gt) ** 2
    pixel_weight = 1.0 + (fg_weight - 1.0) * mask
    loss_mse = (sq_err * pixel_weight).sum() / (pixel_weight.sum() * target.shape[1])

    l1_fg = (v_pred - v_gt).abs() * mask
    loss_l1 = 0.05 * l1_fg.sum() / (mask.sum().clamp(min=1) * target.shape[1])

    return loss_mse + loss_l1


# ============================================================================
# VALIDATION VIS
# ============================================================================

def update_ema(ema, model, decay=0.999):
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.data.mul_(decay).add_(pm.data, alpha=1.0 - decay)


@torch.no_grad()
def validate_flat_image_dir(
    model,
    vae,
    device,
    template_verts,
    val_image_dir,
    reference_mesh_path,
    save_dir,
    step,
    ctrl_size=256,
    nfe=50,
    hair_norm_scale=0.1,
    max_images=8,
    seed=42,
):
    model.eval()
    vae.eval()

    if not os.path.isdir(val_image_dir):
        raise FileNotFoundError(f"val_image_dir not found: {val_image_dir}")
    if not os.path.isfile(reference_mesh_path):
        raise FileNotFoundError(f"reference_mesh_path not found: {reference_mesh_path}")

    _, faces, verts_uvs, faces_uvs = load_obj(reference_mesh_path)

    image_paths = collect_flat_images(val_image_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No validation images found in: {val_image_dir}")

    rng = random.Random(seed + step)
    rng.shuffle(image_paths)
    image_paths = image_paths[:min(max_images, len(image_paths))]

    tfm = transforms.Compose([
        transforms.Resize(
            (ctrl_size, ctrl_size),
            transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    step_dir = os.path.join(save_dir, f"step{step:07d}")
    tex_dir = os.path.join(step_dir, "textures")
    hair_dir = os.path.join(step_dir, "hair")
    mesh_dir = os.path.join(step_dir, "meshes")
    input_dir = os.path.join(step_dir, "input_images")

    os.makedirs(tex_dir, exist_ok=True)
    os.makedirs(hair_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    for img_path in tqdm(image_paths, desc=f"Flat Val {step}", leave=False):
        ctrl = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

        outs = model.infer_all(ctrl, nfe=nfe, cfg_scale=getattr(model, "val_cfg_scale", 1.0))
        pred_hr = vae.decode(outs["z_pred"])

        stem = Path(img_path).stem

        save_image(
            (pred_hr * 0.5 + 0.5).clamp(0, 1),
            os.path.join(tex_dir, f"{stem}_uv.png"),
        )

        save_hair_offset_preview(
            outs["pred_hair"][0],
            os.path.join(hair_dir, f"{stem}_hair.png"),
            hair_norm_scale=hair_norm_scale,
        )
        save_hair_offset_npz_compatible(
            outs["pred_hair"][0],
            os.path.join(hair_dir, f"{stem}_hair.npz"),
            hair_norm_scale=hair_norm_scale,
        )

        pred_verts = (template_verts.unsqueeze(0) + outs["pred_offsets"])[0].detach().cpu().numpy()
        save_obj(
            os.path.join(mesh_dir, f"{stem}_mesh.obj"),
            pred_verts,
            faces,
            verts_uvs,
            faces_uvs,
        )

        Image.open(img_path).convert("RGB").save(os.path.join(input_dir, Path(img_path).name))


# ============================================================================
# DATALOADERS
# ============================================================================

def build_species_dataset(root, args, augment):
    return GroupedImageTextureMeshHairDataset(
        root=root,
        ctrl_size=args.ctrl_size,
        texture_size=args.texture_size,
        hair_dirname=args.hair_dirname,
        shape_dirname=args.shape_dirname,
        guidance_dirname=args.guidance_dirname,
        texture_dirname=args.texture_dirname,
        augment=augment,
        recursive=args.recursive_dataset,
        normalize_mesh_flag=True,
        hair_region_names=args.hair_region_names,
    )


def build_dataloaders(args):
    if args.all_animals:
        datasets_train = []
        ref_specs = None

        for animal in args.animals:
            animal_root = os.path.join(args.base_root, animal, "dataset")
            if not os.path.isdir(animal_root):
                raise FileNotFoundError(f"Dataset root not found for '{animal}': {animal_root}")

            ds_train = build_species_dataset(animal_root, args, augment=True)
            specs = (ds_train.num_verts, ds_train.hair_channels, ds_train.hair_size)
            if ref_specs is None:
                ref_specs = specs
            elif specs != ref_specs:
                raise ValueError(f"Dataset spec mismatch across animals: got {specs}, expected {ref_specs}")

            train_subset, _ = make_train_val_subsets(ds_train, val_ratio=args.val_ratio, seed=args.split_seed)
            datasets_train.append(train_subset)

        train_ds = ConcatDataset(datasets_train)
        num_verts, hair_channels, hair_size = ref_specs
    else:
        dataset_train = build_species_dataset(args.data_root, args, augment=True)
        train_ds, _ = make_train_val_subsets(dataset_train, val_ratio=args.val_ratio, seed=args.split_seed)
        num_verts, hair_channels, hair_size = dataset_train.num_verts, dataset_train.hair_channels, dataset_train.hair_size

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=multitask_collate,
    )

    return train_loader, num_verts, hair_channels, hair_size


def compute_or_load_mean_mesh(datasets, cache_path, force_recompute=False):
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_recompute:
        print(f"[Template] loading cached template verts from: {cache_path}")
        return torch.load(cache_path, map_location="cpu").float()

    verts_list = []
    for ds in datasets:
        base_ds = ds
        if isinstance(base_ds, Subset):
            base_ds = base_ds.dataset
        if isinstance(base_ds, ConcatDataset):
            raise ValueError("Pass base datasets, not ConcatDataset")
        for sample in base_ds.samples:
            verts, _, _, _ = load_obj(sample["mesh"])
            verts = normalize_verts(verts)
            verts_list.append(verts)

    template = torch.from_numpy(np.stack(verts_list, axis=0).mean(axis=0).astype(np.float32)).float()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, cache_path)
    print(f"[Template] saved template verts to: {cache_path}")
    return template


# ============================================================================
# TRAINING SCHEDULER
# ============================================================================

def make_scheduler(optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# TRAIN MULTITASK
# ============================================================================

def build_base_datasets(args):
    base_datasets = []
    if args.all_animals:
        for animal in args.animals:
            animal_root = os.path.join(args.base_root, animal, "dataset")
            base_datasets.append(build_species_dataset(animal_root, args, augment=False))
    else:
        base_datasets.append(build_species_dataset(args.data_root, args, augment=False))
    return base_datasets


def train_multitask(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.flat_val_image_dir is None:
        raise ValueError("--flat_val_image_dir is required for flat-directory validation")
    if args.reference_mesh_path is None:
        raise ValueError("--reference_mesh_path is required for flat-directory validation")

    resume_ckpt = None
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        restore_checkpoint_model_args(args, resume_ckpt)
    # Resolve latent dimensions before masks, loaders, and the flow model.
    vae = build_texture_vae(args, device)
    if resume_ckpt is not None:
        validate_checkpoint_vae(resume_ckpt, vae)

    train_loader, num_verts, hair_channels, hair_size = build_dataloaders(args)
    base_datasets = build_base_datasets(args)

    template_cache_path = os.path.join(args.output_dir, "multitask", "mean_template_verts.pt")
    template_verts = compute_or_load_mean_mesh(base_datasets, template_cache_path).to(device)

    ref_ds = base_datasets[0]
    mesh_edges = ref_ds.mesh_edges.to(device)
    mesh_neighbors = ref_ds.mesh_neighbors

    uv_mask_hr = load_binary_mask(args.uv_mask_path, args.texture_size, invert=args.invert_mask).to(device)
    uv_mask_lat = F.interpolate(
        uv_mask_hr.unsqueeze(0),
        size=(args.latent_size, args.latent_size),
        mode="nearest",
    ).squeeze(0)

    model = AnimalTextureMeshHairFlowModel(
        ctrl_img_size=args.ctrl_size,
        latent_size=args.latent_size,
        latent_ch=args.latent_ch,
        flow_base_ch=args.flow_base_ch,
        flow_ch_mult=tuple(args.flow_ch_mult),
        freeze_encoder=args.freeze_encoder,
        flow_attn_levels=tuple(args.flow_attn_levels),
        num_verts=num_verts,
        hair_out_channels=hair_channels,
        hair_size=hair_size,
        mesh_hidden_dim=args.mesh_hidden_dim,
        hair_hidden_dim=args.hair_hidden_dim,
        head_dropout=args.head_dropout,
        mesh_head_depth=args.mesh_head_depth,
        hair_head_depth=args.hair_head_depth,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        use_uv_cross_attn=args.use_uv_cross_attn,
        n_surface_points=args.n_surface_points,
        uv_attn_heads=args.uv_attn_heads,
        uv_attn_dropout=args.uv_attn_dropout,
        use_color_embed=args.use_color_embed,
        use_dino_intermediate_tokens=args.use_dino_intermediate_tokens,
        dino_intermediate_layers=tuple(args.dino_intermediate_layers),
        dino_intermediate_grid=args.dino_intermediate_grid,
        freeze_dino_intermediate=args.freeze_dino_intermediate,
        num_hair_regions=len(parse_region_names(args.hair_region_names)),
    ).to(device)

    ema = deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    encoder_params = [p for n, p in model.named_parameters() if p.requires_grad and "encoder" in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and "encoder" not in n]
    param_groups = [{"params": other_params, "lr": args.lr}]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": args.lr * 0.1})
        print(f"[Optim] encoder lr={args.lr*0.1:.1e}  others lr={args.lr:.1e}")
    else:
        print(f"[Optim] encoder frozen  lr={args.lr:.1e}")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = make_scheduler(optimizer, args.flow_warmup_steps, args.flow_max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    out_dir = os.path.join(args.output_dir, "multitask")
    val_dir = os.path.join(out_dir, "validation")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    start_step = 0
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        ema.load_state_dict(resume_ckpt.get("ema_model", resume_ckpt["model"]))
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        start_step = resume_ckpt["step"]
        resume_ckpt = None

    model.train()
    data_iter = iter(train_loader)
    running_loss = 0.0
    pbar = tqdm(range(start_step, args.flow_max_steps), initial=start_step, total=args.flow_max_steps, desc="Train MultiTask")

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        ctrl = batch["control"].to(device, non_blocking=True)
        target_hr = batch["target_hr"].to(device, non_blocking=True)
        verts_gt = batch["verts"].to(device, non_blocking=True)
        hair_q = batch["hair_q"].to(device, non_blocking=True)
        hair_scale = batch["hair_scale"].to(device, non_blocking=True)
        hair_mask = batch["hair_mask"].to(device, non_blocking=True)

        with torch.no_grad():
            z_target = vae.encode(
                target_hr, sample_posterior=args.vae_sample_posterior
            )
        expected_shape = (args.latent_ch, args.latent_size, args.latent_size)
        if tuple(z_target.shape[1:]) != expected_shape:
            raise RuntimeError(
                f"VAE returned {tuple(z_target.shape[1:])}, expected {expected_shape}"
            )

        B = ctrl.shape[0]
        t = torch.rand(B, device=device)
        x_noise = torch.randn_like(z_target)
        t_bc = t.reshape(B, 1, 1, 1)
        x_t = (1.0 - t_bc) * x_noise + t_bc * z_target
        v_gt = z_target - x_noise

        mask_lat = uv_mask_lat.unsqueeze(0).expand(B, -1, -1, -1)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            outs = model.forward_multitask(x_t, t, ctrl, cond_drop_prob=args.cond_drop_prob)

            loss_tex = multitask_flow_loss(
                v_pred=outs["v_pred"],
                target=z_target,
                mask=mask_lat,
                v_gt=v_gt,
                fg_weight=args.fg_weight,
            )

            hair_gt = dequantize_hair_on_device(
                hair_q,
                hair_scale,
                hair_norm_scale=args.hair_norm_scale,
                dtype=outs["pred_hair"].dtype,
            )
            hair_mask_for_loss = hair_mask.to(dtype=outs["pred_hair"].dtype)

            geo = mesh_hair_losses(
                pred_offsets=outs["pred_offsets"],
                template_verts=template_verts,
                verts_gt=verts_gt,
                pred_hair=outs["pred_hair"],
                hair_gt=hair_gt,
                hair_mask=hair_mask_for_loss,
                mesh_edges=mesh_edges,
                mesh_neighbors=mesh_neighbors,
                mesh_loss_type=args.mesh_loss_type,
                hair_l1_weight=args.hair_l1_weight,
                hair_mse_weight=args.hair_mse_weight,
                hair_fg_weight=args.hair_fg_weight,
                hair_bg_weight=args.hair_bg_weight,
                total_hair_weight=args.total_hair_weight,
                mesh_edge_weight=args.mesh_edge_weight,
                mesh_laplacian_weight=args.mesh_laplacian_weight,
                mesh_offset_reg_weight=args.mesh_offset_reg_weight,
                pred_region_len=outs.get("pred_region_len"),
                gt_region_len=batch["hair_region_length"].to(device),
                pred_region_curl_logits=outs.get("pred_region_curl_logits"),
                gt_region_curl=batch["hair_region_curl"].to(device),
                hair_region_valid=batch["hair_region_valid"].to(device),
                hair_region_length_scale=args.hair_region_length_scale,
                region_len_weight=args.region_len_weight,
                region_curl_weight=args.region_curl_weight,
            )

            loss = (
                args.texture_loss_weight * loss_tex
                + args.geometry_loss_weight * geo["total"]
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema(ema, model, decay=args.ema_decay)

        running_loss += loss.item()
        if (step + 1) % 20 == 0:
            pbar.set_postfix({
                "total": f"{running_loss / 20:.4f}",
                "tex": f"{loss_tex.item():.4f}",
                "mesh": f"{geo['mesh_base'].item():.4f}",
                "edge": f"{geo['mesh_edge'].item():.4f}",
                "lap": f"{geo['mesh_laplacian'].item():.4f}",
                "hair": f"{geo['hair'].item():.4f}",
                "h_len": f"{geo['hair_region_len'].item():.4f}",
                "h_curl": f"{geo['hair_region_curl'].item():.4f}",
            })
            running_loss = 0.0

        if (step + 1) % args.flow_val_every == 0:
            ema.val_cfg_scale = args.val_cfg_scale
            validate_flat_image_dir(
                model=ema,
                vae=vae,
                device=device,
                template_verts=template_verts,
                val_image_dir=args.flat_val_image_dir,
                reference_mesh_path=args.reference_mesh_path,
                save_dir=val_dir,
                step=step + 1,
                ctrl_size=args.ctrl_size,
                nfe=args.val_nfe,
                hair_norm_scale=args.hair_norm_scale,
                max_images=args.val_save_count,
                seed=args.split_seed,
            )
            ema.train()
            model.train()

        if (step + 1) % args.flow_save_every == 0:
            ckpt = {
                "step": step + 1,
                "model": model.state_dict(),
                "ema_model": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "template_verts": template_verts.cpu(),
                "num_verts": num_verts,
                "hair_channels": hair_channels,
                "hair_size": hair_size,
                "mesh_edges": mesh_edges.cpu(),
                "vae_config": vae.checkpoint_config(),
                "args": vars(args),
            }
            sp = os.path.join(ckpt_dir, f"ckpt_{step+1:07d}.pt")
            torch.save(ckpt, sp)
            torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))

    final_ckpt = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.isfile(final_ckpt):
        torch.save({
            "step": args.flow_max_steps,
            "model": model.state_dict(),
            "ema_model": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "template_verts": template_verts.cpu(),
            "num_verts": num_verts,
            "hair_channels": hair_channels,
            "hair_size": hair_size,
            "mesh_edges": mesh_edges.cpu(),
            "vae_config": vae.checkpoint_config(),
            "args": vars(args),
        }, final_ckpt)

    print("[MultiTask] done")
    return final_ckpt


# ============================================================================
# INFERENCE
# ============================================================================

@torch.no_grad()
def infer_multitask(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not args.resume:
        raise ValueError("--resume required for infer_multitask")
    ckpt = torch.load(args.resume, map_location="cpu")
    restore_checkpoint_model_args(args, ckpt)
    vae = build_texture_vae(args, device)
    validate_checkpoint_vae(ckpt, vae)
    template_verts = ckpt["template_verts"].to(device)
    num_verts = ckpt["num_verts"]
    hair_channels = ckpt["hair_channels"]
    hair_size = tuple(ckpt["hair_size"])

    model = AnimalTextureMeshHairFlowModel(
        ctrl_img_size=args.ctrl_size,
        latent_size=args.latent_size,
        latent_ch=args.latent_ch,
        flow_base_ch=args.flow_base_ch,
        flow_ch_mult=tuple(args.flow_ch_mult),
        freeze_encoder=True,
        flow_attn_levels=tuple(args.flow_attn_levels),
        num_verts=num_verts,
        hair_out_channels=hair_channels,
        hair_size=hair_size,
        mesh_hidden_dim=args.mesh_hidden_dim,
        hair_hidden_dim=args.hair_hidden_dim,
        head_dropout=args.head_dropout,
        mesh_head_depth=args.mesh_head_depth,
        hair_head_depth=args.hair_head_depth,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        use_uv_cross_attn=args.use_uv_cross_attn,
        n_surface_points=args.n_surface_points,
        uv_attn_heads=args.uv_attn_heads,
        uv_attn_dropout=args.uv_attn_dropout,
        use_color_embed=args.use_color_embed,
        use_dino_intermediate_tokens=args.use_dino_intermediate_tokens,
        dino_intermediate_layers=tuple(args.dino_intermediate_layers),
        dino_intermediate_grid=args.dino_intermediate_grid,
        freeze_dino_intermediate=args.freeze_dino_intermediate,
        num_hair_regions=len(parse_region_names(args.hair_region_names)),
    ).to(device).eval()
    model.load_state_dict(ckpt.get("ema_model", ckpt["model"]))

    tfm = transforms.Compose([
        transforms.Resize((args.ctrl_size, args.ctrl_size), transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    os.makedirs(args.output_dir, exist_ok=True)
    tex_dir = os.path.join(args.output_dir, "textures")
    hair_dir = os.path.join(args.output_dir, "hair")
    mesh_dir = os.path.join(args.output_dir, "meshes")
    os.makedirs(tex_dir, exist_ok=True)
    os.makedirs(hair_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    if not args.reference_mesh_path:
        raise ValueError("--reference_mesh_path required for infer_multitask mesh export")
    _, faces, verts_uvs, faces_uvs = load_obj(args.reference_mesh_path)

    paths = collect_flat_images(args.infer_input)

    for p in tqdm(paths, desc="Infer MultiTask"):
        ctrl = tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        outs = model.infer_all(ctrl, nfe=args.val_nfe, cfg_scale=args.val_cfg_scale)
        pred_hr = vae.decode(outs["z_pred"])
        save_image((pred_hr * 0.5 + 0.5).clamp(0, 1), os.path.join(tex_dir, Path(p).stem + "_uv.png"))

        save_hair_offset_preview(
            outs["pred_hair"][0],
            os.path.join(hair_dir, Path(p).stem + "_hair.png"),
            hair_norm_scale=args.hair_norm_scale,
        )
        save_hair_offset_npz_compatible(
            outs["pred_hair"][0],
            os.path.join(hair_dir, Path(p).stem + "_hair.npz"),
            hair_norm_scale=args.hair_norm_scale,
        )

        pred_verts = (template_verts.unsqueeze(0) + outs["pred_offsets"])[0].detach().cpu().numpy()
        save_obj(
            os.path.join(mesh_dir, Path(p).stem + "_mesh.obj"),
            pred_verts,
            faces,
            verts_uvs,
            faces_uvs,
        )

    print(f"[Infer] saved outputs to {args.output_dir}")


