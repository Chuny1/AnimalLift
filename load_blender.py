import bpy
import json
import math
import re
import numpy as np
import bmesh
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
import time
import os
import sys
import argparse

# Blender --python /absolute/path/load_blender.py can run from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from animallift.config import (
    load_config, parse_configured_args, require_file, result_paths, select_result_obj,
    blend_output_path,
)
BLENDER_CONFIG = load_config()["blender"]

IMAGE_PREFIX = "dens_"
VGROUP_PREFIX = "len_"

REGIONS = [
    "face", "ear", "neck", "body", "leg", "tail"
]

CURVE_REGION_MAP = {
    "face_fur": "face",
    "ear_fur": "ear",
    "body_fur": "body",
    "leg_fur": "leg",
    "neck_fur": "neck",
    "neck": "neck",
    "mouse_fur": "face",
    "tail_fur": "tail",
}

IGNORE_NAME_KEYWORDS = ["undercoat"]

MOUSE_LENGTH_VALUE = 0.05

BASE_PART_RESOLUTION = 512
SAMPLES_PER_STRAND = 32
ATLAS_TILE_PADDING = 0.02

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".exr"}


UNDERHAIR_LENGTH = 0.005
UNDERHAIR_INTERP_DENSITY = 500000.0
UNDERHAIR_MASK_GROUP = "underhair_mask"




def log(msg: str):
    print(f"[hair-load] {msg}")


def resolve_paths_from_obj(obj_path: str):
    paths = result_paths(obj_path)
    require_file(obj_path, "prediction OBJ")
    require_file(paths["hair_map"], "predicted hair NPZ")
    paths.update({
        "hair_meta": BLENDER_CONFIG["hair_meta_path"],
        "mesh_normal": BLENDER_CONFIG["normal_path"],
        "mesh_roughness": BLENDER_CONFIG["roughness_path"],
        "hdr_path": BLENDER_CONFIG["hdr_path"],
        "density_dir": BLENDER_CONFIG["density_dir"],
        "hair_map_curl_out": paths["hair_map"],
    })
    return paths


def load_global_hair_meta(meta_path: str):
    meta_path = Path(meta_path).resolve()
    if not meta_path.exists():
        raise FileNotFoundError(f"Global hair meta not found: {meta_path}")

    data = np.load(str(meta_path), allow_pickle=True)

    required_keys = [
        "group_names",
        "guide_group_id_map",
        "uv_face_index_map",
        "uv_bary_map",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in global hair meta: {meta_path}")

    group_names = [str(n) for n in data["group_names"]]
    guide_group_id_map = data["guide_group_id_map"].astype(np.int32)
    uv_face_index_map = data["uv_face_index_map"].astype(np.int32)
    uv_bary_map = data["uv_bary_map"].astype(np.float32)

    log(f"Loaded global hair meta: {meta_path}")
    log(f"  Groups: {group_names}")
    log(f"  guide_group_id_map shape: {guide_group_id_map.shape}")
    log(f"  uv_face_index_map shape: {uv_face_index_map.shape}")
    log(f"  uv_bary_map shape: {uv_bary_map.shape}")

    return group_names, guide_group_id_map, uv_face_index_map, uv_bary_map


def import_obj_as_mesh(obj_path: str, object_name: str = None):
    obj_path = str(Path(obj_path).resolve())

    bpy.ops.object.select_all(action='DESELECT')
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=obj_path)
    else:
        bpy.ops.import_scene.obj(filepath=obj_path)

    imported_meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not imported_meshes:
        raise Exception(f"No mesh imported from {obj_path}")

    mesh_obj = imported_meshes[0]
    if object_name:
        mesh_obj.name = object_name
        mesh_obj.data.name = object_name

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.shade_smooth()

    log(f"Imported mesh: {mesh_obj.name} ({len(mesh_obj.data.vertices)} verts, smooth shading)")
    return mesh_obj


def delete_object(obj):
    if obj is None:
        return
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.delete()


def copy_uvs_from_reference_mesh(target_obj, reference_obj):
    target_mesh = target_obj.data
    reference_mesh = reference_obj.data

    if not reference_mesh.uv_layers:
        raise RuntimeError(f"Reference mesh '{reference_obj.name}' has no UV map.")

    if len(target_mesh.loops) != len(reference_mesh.loops):
        raise RuntimeError(
            f"Cannot copy UVs by topology: loop count mismatch "
            f"(target={len(target_mesh.loops)}, reference={len(reference_mesh.loops)})."
        )

    if len(target_mesh.polygons) != len(reference_mesh.polygons):
        raise RuntimeError(
            f"Cannot copy UVs by topology: polygon count mismatch "
            f"(target={len(target_mesh.polygons)}, reference={len(reference_mesh.polygons)})."
        )

    target_uv = target_mesh.uv_layers.active
    if target_uv is None:
        target_uv = target_mesh.uv_layers.new(name=reference_mesh.uv_layers.active.name or "UVMap")

    ref_uv = reference_mesh.uv_layers.active
    uv_data = np.empty(len(reference_mesh.loops) * 2, dtype=np.float32)
    ref_uv.data.foreach_get("uv", uv_data)
    target_uv.data.foreach_set("uv", uv_data)
    target_mesh.update()

    try:
        target_mesh.uv_layers.active = target_uv
    except Exception:
        pass

    log(
        f"Copied UVs from '{reference_obj.name}' to '{target_obj.name}' "
        f"({len(target_mesh.loops)} loops)"
    )


def try_load_uvs_from_reference_obj(target_obj, uv_source_obj_path: str):
    uv_source_obj_path = Path(uv_source_obj_path).resolve()
    if not uv_source_obj_path.exists():
        raise FileNotFoundError(f"UV source OBJ not found: {uv_source_obj_path}")

    log(f"Imported mesh has no UVs; loading fallback UV source: {uv_source_obj_path}")
    reference_obj = import_obj_as_mesh(str(uv_source_obj_path), object_name="__UVReference__")




def open_image_simple(image_path: str, non_color: bool = True):
    image_path = str(Path(image_path).resolve())
    img = bpy.data.images.load(filepath=image_path, check_existing=True)
    img.colorspace_settings.name = 'Non-Color' if non_color else 'sRGB'
    img.pixels[:]
    return img


def open_image_color(image_path: str):
    return open_image_simple(image_path, non_color=False)


def load_density_images(mesh_obj, density_dir: str):
    density_images = {}
    for region in REGIONS:
        image_path = Path(density_dir) / f"{region}.png"
        if not image_path.exists():
            log(f"  [WARN] Density image not found: {image_path}")
            continue
        img = open_image_simple(str(image_path), non_color=True)
        img.name = f"{IMAGE_PREFIX}{region}_{mesh_obj.name}"
        density_images[region] = img
    return density_images


def create_vertex_group_from_image(obj, image_path: str, group_name: str, scale: float = 1.0):
    mesh = obj.data
    if not mesh.uv_layers:
        raise Exception("Target mesh has no UVs")

    img = open_image_simple(image_path, non_color=True)
    width, height = img.size

    pixel_count = width * height * 4
    pixels_np = np.empty(pixel_count, dtype=np.float32)
    img.pixels.foreach_get(pixels_np)
    pixels_np = pixels_np.reshape((height, width, 4))

    lum = (
        0.2126 * pixels_np[:, :, 0]
        + 0.7152 * pixels_np[:, :, 1]
        + 0.0722 * pixels_np[:, :, 2]
    )

    total_loops = len(mesh.loops)
    loop_uvs = np.empty(total_loops * 2, dtype=np.float32)
    mesh.uv_layers.active.data.foreach_get("uv", loop_uvs)
    loop_uvs = loop_uvs.reshape((total_loops, 2))

    loop_verts = np.empty(total_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_verts)

    u = np.clip(loop_uvs[:, 0], 0.0, 1.0)
    v = np.clip(loop_uvs[:, 1], 0.0, 1.0)
    xi = (u * (width - 1)).astype(np.int32)
    yi = (v * (height - 1)).astype(np.int32)
    loop_weights = lum[yi, xi] * float(scale)

    num_verts = len(mesh.vertices)
    weight_sum = np.zeros(num_verts, dtype=np.float64)
    weight_cnt = np.zeros(num_verts, dtype=np.int32)
    np.add.at(weight_sum, loop_verts, loop_weights)
    np.add.at(weight_cnt, loop_verts, 1)

    mask = weight_cnt > 0
    avg_weights = np.zeros(num_verts, dtype=np.float64)
    avg_weights[mask] = weight_sum[mask] / weight_cnt[mask]

    old_vg = obj.vertex_groups.get(group_name)
    if old_vg is not None:
        obj.vertex_groups.remove(old_vg)
    vg = obj.vertex_groups.new(name=group_name)

    indices = np.where(mask)[0]
    for vid in indices:
        vg.add([int(vid)], float(avg_weights[vid]), 'REPLACE')

    return vg, img


def load_hair_info(json_path):
    json_path = Path(json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"hair_info.json not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)





def ensure_triangulated_bmesh(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    return bm


def compute_root_uvs(roots, mesh_obj):
    mesh = mesh_obj.data
    if not mesh.uv_layers.active:
        raise RuntimeError("Mesh has no active UV layer")

    bm = ensure_triangulated_bmesh(mesh)
    uv_layer = bm.loops.layers.uv.get(mesh.uv_layers.active.name)
    if uv_layer is None:
        bm.free()
        raise RuntimeError("Could not access active UV layer")

    bvh = BVHTree.FromBMesh(bm)
    uvs = []

    for root in roots:
        hit = bvh.find_nearest(Vector(root))
        if hit is None or hit[2] is None:
            uvs.append((0.5, 0.5))
            continue

        location, normal, face_index, dist = hit
        face = bm.faces[face_index]
        if len(face.verts) != 3:
            uvs.append((0.5, 0.5))
            continue

        v0, v1, v2 = [f.co for f in face.verts]
        uv0 = face.loops[0][uv_layer].uv
        uv1 = face.loops[1][uv_layer].uv
        uv2 = face.loops[2][uv_layer].uv

        uv = barycentric_transform(
            location, v0, v1, v2,
            uv0.to_3d(), uv1.to_3d(), uv2.to_3d()
        )
        uvs.append((float(uv.x), float(uv.y)))

    bm.free()
    return uvs


def build_face_geom(mesh_obj):
    mesh = mesh_obj.data
    if not mesh.uv_layers.active:
        raise RuntimeError("Mesh has no active UV layer")

    bm = ensure_triangulated_bmesh(mesh)
    uv_layer = bm.loops.layers.uv.get(mesh.uv_layers.active.name)
    if uv_layer is None:
        bm.free()
        raise RuntimeError("Could not access active UV layer")

    num_faces = len(bm.faces)
    face_pos = np.empty((num_faces, 3, 3), dtype=np.float32)
    face_uvs = np.empty((num_faces, 3, 2), dtype=np.float32)

    for fi, face in enumerate(bm.faces):
        for vi in range(3):
            co = face.verts[vi].co
            uv = face.loops[vi][uv_layer].uv
            face_pos[fi, vi] = (co.x, co.y, co.z)
            face_uvs[fi, vi] = (uv.x, uv.y)

    bm.free()
    return face_pos, face_uvs


def reconstruct_surface_points_and_frames(mesh_obj, uv_face_index_map, uv_bary_map):
    face_pos, face_uvs = build_face_geom(mesh_obj)

    height, width = uv_face_index_map.shape
    roots = np.zeros((height, width, 3), dtype=np.float32)
    Tu = np.zeros((height, width, 3), dtype=np.float32)
    Tv = np.zeros((height, width, 3), dtype=np.float32)
    N = np.zeros((height, width, 3), dtype=np.float32)

    ys, xs = np.where(uv_face_index_map >= 0)
    fids = uv_face_index_map[ys, xs].astype(np.int32)
    bary = uv_bary_map[ys, xs].astype(np.float32)

    p0 = face_pos[fids, 0]
    p1 = face_pos[fids, 1]
    p2 = face_pos[fids, 2]

    uv0 = face_uvs[fids, 0]
    uv1 = face_uvs[fids, 1]
    uv2 = face_uvs[fids, 2]

    roots_pts = (
        bary[:, 0:1] * p0 +
        bary[:, 1:2] * p1 +
        bary[:, 2:3] * p2
    )

    dp1 = p1 - p0
    dp2 = p2 - p0
    duv1 = uv1 - uv0
    duv2 = uv2 - uv0

    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    safe = np.abs(det) > 1e-12
    det_safe = det.copy()
    det_safe[~safe] = 1.0

    Tu_pts = np.empty_like(dp1)
    Tv_pts = np.empty_like(dp1)

    Tu_pts[:, 0] = ( dp1[:, 0] * duv2[:, 1] - dp2[:, 0] * duv1[:, 1]) / det_safe
    Tu_pts[:, 1] = ( dp1[:, 1] * duv2[:, 1] - dp2[:, 1] * duv1[:, 1]) / det_safe
    Tu_pts[:, 2] = ( dp1[:, 2] * duv2[:, 1] - dp2[:, 2] * duv1[:, 1]) / det_safe

    Tv_pts[:, 0] = (-dp1[:, 0] * duv2[:, 0] + dp2[:, 0] * duv1[:, 0]) / det_safe
    Tv_pts[:, 1] = (-dp1[:, 1] * duv2[:, 0] + dp2[:, 1] * duv1[:, 0]) / det_safe
    Tv_pts[:, 2] = (-dp1[:, 2] * duv2[:, 0] + dp2[:, 2] * duv1[:, 0]) / det_safe

    N_pts = np.cross(dp1, dp2)

    def normalize(v):
        lens = np.linalg.norm(v, axis=1, keepdims=True)
        lens[lens < 1e-12] = 1.0
        return v / lens

    Tu_pts = normalize(Tu_pts)
    Tv_pts = normalize(Tv_pts)
    N_pts = normalize(N_pts)

    if np.any(~safe):
        Tu_pts[~safe] = normalize(dp1[~safe])
        Tv_pts[~safe] = normalize(np.cross(N_pts[~safe], Tu_pts[~safe]))
        N_pts[~safe] = normalize(np.cross(Tu_pts[~safe], Tv_pts[~safe]))

    roots[ys, xs] = roots_pts
    Tu[ys, xs] = Tu_pts
    Tv[ys, xs] = Tv_pts
    N[ys, xs] = N_pts

    return roots, Tu, Tv, N


def decode_hair_map_npz(hair_map_path: str, meta_path: str, mesh_obj):
    hair_map_path = str(Path(hair_map_path).resolve())
    data = np.load(hair_map_path, allow_pickle=True)

    required_keys = [
        "hair_offsets_local_q",
        "offset_scale",
        "samples",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in hair map: {hair_map_path}")

    hair_offsets_local_q = data["hair_offsets_local_q"].astype(np.float32)
    offset_scale = data["offset_scale"].astype(np.float32)
    samples = int(data["samples"])

    group_names, guide_group_id_map, uv_face_index_map, uv_bary_map = load_global_hair_meta(meta_path)

    if samples < 1:
        raise ValueError(f"Invalid samples={samples} in hair map")

    height, width = guide_group_id_map.shape[:2]

    if hair_offsets_local_q.shape[0] != height or hair_offsets_local_q.shape[1] != width:
        raise ValueError(
            f"Shape mismatch between hair map and global meta: "
            f"hair_offsets_local_q shape={hair_offsets_local_q.shape[:2]}, "
            f"guide_group_id_map shape={guide_group_id_map.shape[:2]}"
        )

    if uv_face_index_map.shape[:2] != (height, width):
        raise ValueError(
            f"Shape mismatch between uv_face_index_map and guide_group_id_map: "
            f"{uv_face_index_map.shape[:2]} vs {(height, width)}"
        )

    if uv_bary_map.shape[:2] != (height, width):
        raise ValueError(
            f"Shape mismatch between uv_bary_map and guide_group_id_map: "
            f"{uv_bary_map.shape[:2]} vs {(height, width)}"
        )

    offsets_local = (hair_offsets_local_q / 127.0) * offset_scale.reshape(1, 1, 1, 3)
    strands_by_group = {name: [] for name in group_names}

    roots_map, Tu_map, Tv_map, N_map = reconstruct_surface_points_and_frames(
        mesh_obj,
        uv_face_index_map,
        uv_bary_map,
    )

    ys, xs = np.where(guide_group_id_map >= 0)
    gids = guide_group_id_map[ys, xs].astype(np.int32)

    for y, x, gid in zip(ys, xs, gids):
        gid = int(gid)
        if gid < 0 or gid >= len(group_names):
            continue

        root = roots_map[y, x]
        Tu = Tu_map[y, x]
        Tv = Tv_map[y, x]
        N = N_map[y, x]

        local = offsets_local[y, x]  # [samples-1, 3]
        world_offsets = (
            local[:, 0:1] * Tu[None, :] +
            local[:, 1:2] * Tv[None, :] +
            local[:, 2:3] * N[None, :]
        )

        strand = np.empty((samples, 3), dtype=np.float32)
        strand[0] = root
        if samples > 1:
            strand[1:] = root[None, :] + world_offsets

        group_name = group_names[gid]
        strands_by_group[group_name].append(strand)

    for group_name in list(strands_by_group.keys()):
        if len(strands_by_group[group_name]) == 0:
            strands_by_group[group_name] = np.zeros((0, samples, 3), dtype=np.float32)
        else:
            strands_by_group[group_name] = np.stack(strands_by_group[group_name], axis=0)

    total_strands = sum(v.shape[0] for v in strands_by_group.values())
    log(f"Decoded hair map: {hair_map_path}")
    log(f"  Samples per strand: {samples}")
    log(f"  Total guide strands: {total_strands}")

    return group_names, strands_by_group, samples


def sanitize_name(name):
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s if s else "HairPart"


def build_curve_object_from_strands(obj_name, strands, root_uvs, mesh_obj):
    if len(strands) == 0:
        return None

    curve_data = bpy.data.curves.new(obj_name, type='CURVE')
    curve_data.dimensions = '3D'

    curve_obj = bpy.data.objects.new(obj_name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

    curve_obj.parent = mesh_obj
    curve_obj.location = (0, 0, 0)
    curve_obj.rotation_euler = (0, 0, 0)
    curve_obj.scale = (1, 1, 1)

    for strand in strands:
        spline = curve_data.splines.new('POLY')
        spline.points.add(len(strand) - 1)
        for i, p in enumerate(strand):
            spline.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='DESELECT')
    curve_obj.select_set(True)
    bpy.context.view_layer.objects.active = curve_obj
    bpy.ops.object.convert(target='CURVES')

    curve_obj = bpy.context.view_layer.objects.active
    curves = curve_obj.data

    curves.surface = mesh_obj
    curves.surface_uv_map = mesh_obj.data.uv_layers.active.name

    if "surface_uv_coordinate" in curves.attributes:
        uv_attr = curves.attributes["surface_uv_coordinate"]
    else:
        uv_attr = curves.attributes.new(
            name="surface_uv_coordinate",
            type='FLOAT2',
            domain='CURVE'
        )

    for i in range(len(root_uvs)):
        uv_attr.data[i].vector = (float(root_uvs[i][0]), float(root_uvs[i][1]))

    log(f"  Created '{curve_obj.name}' with {len(strands)} guide strands")
    return curve_obj


def build_curve_object_from_strands_no_uv(obj_name, strands, mesh_obj):
    if len(strands) == 0:
        return None

    curve_data = bpy.data.curves.new(obj_name, type='CURVE')
    curve_data.dimensions = '3D'

    curve_obj = bpy.data.objects.new(obj_name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

    curve_obj.matrix_world = mesh_obj.matrix_world.copy()

    for strand in strands:
        spline = curve_data.splines.new('POLY')
        spline.points.add(len(strand) - 1)
        for i, p in enumerate(strand):
            spline.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='DESELECT')
    curve_obj.select_set(True)
    bpy.context.view_layer.objects.active = curve_obj
    bpy.ops.object.convert(target='CURVES')

    curve_obj = bpy.context.view_layer.objects.active
    curve_obj.matrix_world = mesh_obj.matrix_world.copy()
    curve_obj.parent = mesh_obj
    curve_obj.matrix_parent_inverse = mesh_obj.matrix_world.inverted()

    curves = curve_obj.data
    curves.surface = mesh_obj

    log(f"  Created '{curve_obj.name}' with {len(strands)} guide strands (no UV attr)")
    return curve_obj


def describe_node_group_inputs(mod):
    if getattr(mod, "node_group", None) is None:
        return []
    items = []
    try:
        for item in mod.node_group.interface.items_tree:
            if getattr(item, "in_out", None) == 'INPUT':
                items.append((item.name, item.identifier, getattr(item, "socket_type", "")))
    except Exception:
        pass
    return items


def get_node_input_identifier_by_name(mod, display_name):
    for name, identifier, socket_type in describe_node_group_inputs(mod):
        if name == display_name:
            return identifier
    return None


def set_gn_input(mod, display_name, value):
    identifier = get_node_input_identifier_by_name(mod, display_name)
    if identifier is None:
        log(f"    [WARN] GN input '{display_name}' not found on '{mod.name}'")
        return False
    try:
        mod[identifier] = value
        return True
    except Exception as e:
        log(f"    [WARN] Failed to set '{display_name}' on '{mod.name}': {e}")
        return False


def find_gn_input_identifier_contains(mod, keywords):
    keywords = [k.lower() for k in keywords]
    for name, identifier, socket_type in describe_node_group_inputs(mod):
        lname = (name or "").lower()
        if all(k in lname for k in keywords):
            return identifier, name
    return None, None


def set_gn_input_by_keywords(mod, keywords, value):
    identifier, display_name = find_gn_input_identifier_contains(mod, keywords)
    if identifier is None:
        log(f"    [WARN] No GN input matching keywords {keywords} on '{mod.name}'")
        return False
    try:
        mod[identifier] = value
        log(f"    + {mod.name} '{display_name}' = {value}")
        return True
    except Exception as e:
        log(f"    [WARN] Failed setting '{display_name}' on '{mod.name}': {e}")
        return False


def debug_modifier_inputs(mod):
    log(f"--- Modifier inputs for {mod.name} ---")
    for name, identifier, socket_type in describe_node_group_inputs(mod):
        log(f"    name='{name}' identifier='{identifier}' socket_type='{socket_type}'")


def select_and_activate(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def get_essentials_blend_path():
    override = BLENDER_CONFIG.get("hair_nodes_path")
    if override:
        require_file(override, "Blender procedural hair node library")
        return Path(override)
    candidates = []

    try:
        local_root = Path(bpy.utils.resource_path('LOCAL'))
        candidates.append(local_root / "datafiles" / "assets" / "geometry_nodes" / "procedural_hair_node_assets.blend")
    except Exception:
        pass

    try:
        system_root = Path(bpy.utils.resource_path('SYSTEM'))
        candidates.append(system_root / "datafiles" / "assets" / "geometry_nodes" / "procedural_hair_node_assets.blend")
    except Exception:
        pass

    try:
        user_datafiles = Path(bpy.utils.user_resource('DATAFILES'))
        candidates.append(user_datafiles / "assets" / "geometry_nodes" / "procedural_hair_node_assets.blend")
    except Exception:
        pass

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not locate procedural_hair_node_assets.blend. Install Blender Essentials "
        "or set blender.hair_nodes_path in your config."
    )


def ensure_node_group_loaded_from_blend(node_group_name, blend_path):
    ng = bpy.data.node_groups.get(node_group_name)
    if ng is not None:
        return ng

    blend_path = Path(blend_path)
    if not blend_path.exists():
        raise FileNotFoundError(f"Blend library not found: {blend_path}")

    with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
        if node_group_name not in data_from.node_groups:
            raise RuntimeError(f"Node group '{node_group_name}' not found in {blend_path}")
        data_to.node_groups = [node_group_name]

    ng = bpy.data.node_groups.get(node_group_name)
    if ng is None:
        raise RuntimeError(f"Failed to append node group '{node_group_name}' from {blend_path}")
    return ng


def add_gn_modifier(curves_obj, node_group_name, modifier_name):
    select_and_activate(curves_obj)

    blend_path = get_essentials_blend_path()
    node_group = ensure_node_group_loaded_from_blend(node_group_name, blend_path)

    mod = curves_obj.modifiers.new(name=modifier_name, type='NODES')
    mod.node_group = node_group
    return mod


def try_add_gn_modifier(curves_obj, node_group_names, modifier_name):
    last_err = None
    for ng_name in node_group_names:
        try:
            mod = add_gn_modifier(curves_obj, ng_name, modifier_name)
            log(f"    + Loaded node group '{ng_name}' as '{modifier_name}'")
            return mod
        except Exception as e:
            last_err = e
    log(f"    [WARN] Could not load any of {node_group_names} for '{modifier_name}': {last_err}")
    return None


def add_custom_post_duplicate_modifier(curves_obj):
    select_and_activate(curves_obj)

    blend_path = get_essentials_blend_path()
    noise_ng = ensure_node_group_loaded_from_blend("Hair Curves Noise", blend_path)
    trim_ng = ensure_node_group_loaded_from_blend("Trim Hair Curves", blend_path)

    bpy.ops.object.modifier_add(type='NODES')
    mod = curves_obj.modifiers[-1]
    mod.name = "Custom Post Duplicate Trim"

    bpy.ops.node.new_geometry_node_group_assign()
    ng = mod.node_group
    ng.name = f"{curves_obj.name}_CustomPostDuplicateTrim"

    nodes = ng.nodes
    links = ng.links

    group_in = None
    group_out = None
    for node in nodes:
        if node.bl_idname == "NodeGroupInput":
            group_in = node
        elif node.bl_idname == "NodeGroupOutput":
            group_out = node

    if group_in is None or group_out is None:
        raise RuntimeError("Failed to get default GN group input/output nodes")

    for node in list(nodes):
        if node not in {group_in, group_out}:
            nodes.remove(node)

    group_in.location = (-1000, 0)
    group_out.location = (500, 0)

    try:
        id_node = nodes.new("GeometryNodeInputID")
    except RuntimeError:
        id_node = nodes.new("GeometryNodeInputIndex")
    id_node.location = (-1000, -220)

    random_value = nodes.new("FunctionNodeRandomValue")
    random_value.location = (-760, -220)
    random_value.data_type = 'BOOLEAN'
    random_value.inputs["Probability"].default_value = 0.08
    if "Seed" in random_value.inputs:
        random_value.inputs["Seed"].default_value = 0

    separate = nodes.new("GeometryNodeSeparateGeometry")
    separate.location = (-520, 0)
    separate.domain = 'CURVE'

    noise = nodes.new("GeometryNodeGroup")
    noise.location = (-220, -40)
    noise.node_tree = noise_ng

    trim = nodes.new("GeometryNodeGroup")
    trim.location = (40, -40)
    trim.node_tree = trim_ng

    join = nodes.new("GeometryNodeJoinGeometry")
    join.location = (260, 20)

    if "Factor" in noise.inputs:
        noise.inputs["Factor"].default_value = 1.0
    if "Distance" in noise.inputs:
        noise.inputs["Distance"].default_value = 0.002
    if "Shape" in noise.inputs:
        noise.inputs["Shape"].default_value = 0.5
    if "Scale" in noise.inputs:
        noise.inputs["Scale"].default_value = 1.0
    if "Scale Along Length" in noise.inputs:
        noise.inputs["Scale Along Length"].default_value = 1.0
    if "Offset" in noise.inputs:
        noise.inputs["Offset"].default_value = 0.0
    if "Cumulative Offset" in noise.inputs:
        noise.inputs["Cumulative Offset"].default_value = False
    if "Seed" in noise.inputs:
        noise.inputs["Seed"].default_value = 0

    if "Mask" in trim.inputs:
        trim.inputs["Mask"].default_value = 1.0
    if "Random Offset" in trim.inputs:
        trim.inputs["Random Offset"].default_value = 0.01
    if "Pin at Parameter" in trim.inputs:
        trim.inputs["Pin at Parameter"].default_value = 0.0
    if "Seed" in trim.inputs:
        trim.inputs["Seed"].default_value = 0
    if "Replace Length" in trim.inputs:
        trim.inputs["Replace Length"].default_value = False

    if "ID" in random_value.inputs:
        if "ID" in id_node.outputs:
            links.new(id_node.outputs["ID"], random_value.inputs["ID"])
        else:
            links.new(id_node.outputs["Index"], random_value.inputs["ID"])

    links.new(group_in.outputs["Geometry"], separate.inputs["Geometry"])
    links.new(random_value.outputs["Value"], separate.inputs["Selection"])

    links.new(separate.outputs["Selection"], noise.inputs["Geometry"])
    links.new(noise.outputs["Geometry"], trim.inputs["Geometry"])

    links.new(separate.outputs["Inverted"], join.inputs["Geometry"])
    links.new(trim.outputs["Geometry"], join.inputs["Geometry"])
    links.new(join.outputs["Geometry"], group_out.inputs["Geometry"])

    log(f"    + Custom Post Duplicate Trim on {curves_obj.name}")
    return mod


def setup_modifiers(curves_obj, group_name, mesh_obj):
    is_mouse = group_name == "mouse_fur"
    is_face = group_name == "face"

    profile_mod = add_gn_modifier(curves_obj, "Set Hair Curve Profile", "Set Hair Curve Profile")
    set_gn_input(profile_mod, "Radius", 0.0001)
    log("    + Set Hair Curve Profile (radius=0.0001)")

    if (not is_mouse) and (not is_face):
        mod = add_gn_modifier(curves_obj, "Duplicate Hair Curves", "Duplicate Hair Curves")
        set_gn_input(mod, "Amount", 8)
        set_gn_input(mod, "Radius", 0.007)
        set_gn_input(mod, "Distribution Shape", 1.0)
        log("    + Duplicate Hair Curves (radius=0.005, distribution_shape=1.0)")
    else:
        log("    + Skip Duplicate Hair Curves for mouse_fur/face")

    if is_face:
        mod = add_gn_modifier(curves_obj, "Duplicate Hair Curves", "Duplicate Hair Curves")
        set_gn_input(mod, "Amount", 2)
        set_gn_input(mod, "Radius", 0.005)
        set_gn_input(mod, "Distribution Shape", 1.0)


def delete_default_cube():
    cube = bpy.data.objects.get("Cube")
    if cube is None:
        return

    bpy.ops.object.select_all(action='DESELECT')
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.delete()

    log("Deleted default Cube")


def clear_material_slots(obj):
    if not hasattr(obj.data, "materials"):
        return
    obj.data.materials.clear()


def assign_material(obj, material):
    if not hasattr(obj.data, "materials"):
        return
    obj.data.materials.clear()
    obj.data.materials.append(material)
    if hasattr(obj, "active_material"):
        obj.active_material = material


def extract_numeric_suffix(name: str):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.stem.lower())
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    key.append(path.suffix.lower())
    return key


def find_texture_for_obj(obj_path: str, textures_dir: str):
    name = result_paths(obj_path)["texture"].name
    chosen = Path(textures_dir) / name
    require_file(chosen, "predicted UV texture")
    return chosen


def set_node_input_if_exists(node, input_name, value):
    if input_name in node.inputs:
        node.inputs[input_name].default_value = value


def set_enum_attr_if_exists(node, attr_names, value):
    for attr in attr_names:
        if hasattr(node, attr):
            try:
                setattr(node, attr, value)
                return True
            except Exception:
                pass
    return False


def create_simple_gray_material(obj, material_name="FallbackGrayMaterial"):
    mat = bpy.data.materials.get(material_name)
    if mat is None:
        mat = bpy.data.materials.new(material_name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    set_node_input_if_exists(bsdf, "Base Color", (0.6, 0.6, 0.6, 1.0))
    set_node_input_if_exists(bsdf, "Roughness", 0.5)
    set_node_input_if_exists(bsdf, "IOR", 1.5)

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    assign_material(obj, mat)
    log(f"Assigned fallback material: {mat.name}")
    return mat


def create_mesh_material(mesh_obj, texture_path: str, normal_path: str, roughness_path: str, material_name="MeshMaterial"):
    texture_path = Path(texture_path).resolve()
    normal_path = Path(normal_path).resolve()
    roughness_path = Path(roughness_path).resolve()

    if not texture_path.exists():
        raise FileNotFoundError(f"Mesh texture not found: {texture_path}")
    if not normal_path.exists():
        raise FileNotFoundError(f"Normal map not found: {normal_path}")
    if not roughness_path.exists():
        raise FileNotFoundError(f"Roughness map not found: {roughness_path}")

    mat = bpy.data.materials.get(material_name)
    if mat is None:
        mat = bpy.data.materials.new(material_name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (350, 0)
    set_node_input_if_exists(bsdf, "IOR", 1.5)

    tex_color = nodes.new("ShaderNodeTexImage")
    tex_color.location = (-450, 150)
    tex_color.image = open_image_color(str(texture_path))
    tex_color.interpolation = 'Linear'
    tex_color.extension = 'REPEAT'

    tex_rough = nodes.new("ShaderNodeTexImage")
    tex_rough.location = (-450, -50)
    tex_rough.image = open_image_simple(str(roughness_path), non_color=True)
    tex_rough.interpolation = 'Linear'
    tex_rough.extension = 'REPEAT'

    tex_normal = nodes.new("ShaderNodeTexImage")
    tex_normal.location = (-450, -250)
    tex_normal.image = open_image_simple(str(normal_path), non_color=True)
    tex_normal.interpolation = 'Linear'
    tex_normal.extension = 'REPEAT'

    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.location = (-100, -250)
    set_node_input_if_exists(normal_map, "Strength", 1.0)

    links.new(tex_color.outputs["Color"], bsdf.inputs["Base Color"])
    if "Alpha" in tex_color.outputs and "Alpha" in bsdf.inputs:
        links.new(tex_color.outputs["Alpha"], bsdf.inputs["Alpha"])
    links.new(tex_rough.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(tex_normal.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    try:
        mat.blend_method = 'HASHED'
    except Exception:
        pass
    try:
        mat.shadow_method = 'HASHED'
    except Exception:
        pass

    assign_material(mesh_obj, mat)
    log(f"Assigned mesh material: {mat.name}")
    return mat


def create_hair_material(texture_path: str, material_name="HairMaterial"):
    texture_path = Path(texture_path).resolve()
    if not texture_path.exists():
        raise FileNotFoundError(f"Hair texture not found: {texture_path}")

    mat = bpy.data.materials.get(material_name)
    if mat is None:
        mat = bpy.data.materials.new(material_name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    tex_mesh = nodes.new("ShaderNodeTexImage")
    tex_mesh.location = (-900, 200)
    tex_mesh.image = open_image_color(str(texture_path))
    tex_mesh.interpolation = 'Linear'
    tex_mesh.extension = 'REPEAT'

    tex_hair = nodes.new("ShaderNodeTexImage")
    tex_hair.location = (-900, -200)
    tex_hair.image = open_image_color(str(texture_path))
    tex_hair.interpolation = 'Linear'
    tex_hair.extension = 'REPEAT'

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (-500, 180)
    set_node_input_if_exists(bsdf, "IOR", 1.5)
    set_node_input_if_exists(bsdf, "Roughness", 0.5)
    set_node_input_if_exists(bsdf, "Alpha", 1.0)

    hair_bsdf = nodes.new("ShaderNodeBsdfHairPrincipled")
    hair_bsdf.location = (-500, -220)

    set_enum_attr_if_exists(hair_bsdf, ["model", "distribution"], 'CHIANG')
    set_enum_attr_if_exists(hair_bsdf, ["parametrization"], 'COLOR')

    set_node_input_if_exists(hair_bsdf, "Roughness", 0.4)
    set_node_input_if_exists(hair_bsdf, "Radial Roughness", 0.6)
    set_node_input_if_exists(hair_bsdf, "Coat", 0.0)
    set_node_input_if_exists(hair_bsdf, "IOR", 1.5)
    set_node_input_if_exists(hair_bsdf, "Offset", math.radians(3.0))
    set_node_input_if_exists(hair_bsdf, "Random Roughness", 0.25)

    mix_main = nodes.new("ShaderNodeMixShader")
    mix_main.location = (-120, 20)
    if "Fac" in mix_main.inputs:
        mix_main.inputs["Fac"].default_value = 0.6

    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (-120, -260)

    light_path = nodes.new("ShaderNodeLightPath")
    light_path.location = (-360, 320)

    mix_shadow = nodes.new("ShaderNodeMixShader")
    mix_shadow.location = (180, 50)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (420, 50)

    links.new(tex_mesh.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex_hair.outputs["Color"], hair_bsdf.inputs["Color"])
    links.new(bsdf.outputs["BSDF"], mix_main.inputs[1])
    links.new(hair_bsdf.outputs["BSDF"], mix_main.inputs[2])
    links.new(light_path.outputs["Is Shadow Ray"], mix_shadow.inputs["Fac"])
    links.new(mix_main.outputs["Shader"], mix_shadow.inputs[1])
    links.new(transparent.outputs["BSDF"], mix_shadow.inputs[2])
    links.new(mix_shadow.outputs["Shader"], out.inputs["Surface"])

    try:
        mat.blend_method = 'HASHED'
    except Exception:
        pass
    try:
        mat.shadow_method = 'HASHED'
    except Exception:
        pass
    try:
        mat.use_backface_culling = False
    except Exception:
        pass

    log(f"Created hair material: {mat.name}")
    return mat


def setup_cycles_render(scene):
    scene.render.engine = 'CYCLES'
    # try:
    #     scene.cycles.device = 'GPU'
    # except Exception:
    #     pass
    try:
        scene.render.film_transparent = True
    except Exception:
        pass
    log("Set render engine to Cycles")


def setup_world_hdr(hdr_path: str):
    hdr_path = Path(hdr_path).resolve()
    if not hdr_path.exists():
        raise FileNotFoundError(f"HDR not found: {hdr_path}")

    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world

    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-900, 0)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-700, 0)

    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.location = (-500, 0)
    env_tex.image = bpy.data.images.load(filepath=str(hdr_path), check_existing=True)

    background = nodes.new("ShaderNodeBackground")
    background.location = (-220, 0)
    background.inputs["Strength"].default_value = 1.0

    world_out = nodes.new("ShaderNodeOutputWorld")
    world_out.location = (300, 0)

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    links.new(env_tex.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], world_out.inputs["Surface"])

    scene.render.film_transparent = True
    log(f"Set HDR world lighting: {hdr_path}")



def sample_image_luminance_at_uv(img, uv, pixels_np=None):
    width, height = img.size

    if pixels_np is None:
        pixel_count = width * height * 4
        pixels_np = np.empty(pixel_count, dtype=np.float32)
        img.pixels.foreach_get(pixels_np)
        pixels_np = pixels_np.reshape((height, width, 4))

    u = float(np.clip(uv[0], 0.0, 1.0))
    v = float(np.clip(uv[1], 0.0, 1.0))
    xi = int(u * (width - 1))
    yi = int(v * (height - 1))

    rgba = pixels_np[yi, xi]
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return float(lum)


def remove_curve_uv_attributes(curves_obj):
    curves = curves_obj.data

    try:
        curves.surface_uv_map = ""
    except Exception:
        pass

    for attr_name in ["surface_uv_coordinate", "uv_map", "UVMap", "UV", "map1"]:
        attr = curves.attributes.get(attr_name)
        if attr is not None:
            try:
                curves.attributes.remove(attr)
                log(f"  Removed curve attribute: {attr_name}")
            except Exception as e:
                log(f"  [WARN] Failed removing curve attribute '{attr_name}': {e}")


def create_underhair_guides_from_mesh(mesh_obj, mask_image_path, strand_length=0.01, threshold=0.05):
    mesh = mesh_obj.data
    if not mesh.uv_layers.active:
        raise RuntimeError("Mesh has no active UV layer; underhair mask sampling requires UVs.")

    mask_img = open_image_simple(str(mask_image_path), non_color=True)
    width, height = mask_img.size

    pixel_count = width * height * 4
    pixels_np = np.empty(pixel_count, dtype=np.float32)
    mask_img.pixels.foreach_get(pixels_np)
    pixels_np = pixels_np.reshape((height, width, 4))

    total_loops = len(mesh.loops)
    loop_uvs = np.empty(total_loops * 2, dtype=np.float32)
    mesh.uv_layers.active.data.foreach_get("uv", loop_uvs)
    loop_uvs = loop_uvs.reshape((total_loops, 2))

    loop_verts = np.empty(total_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_verts)

    num_verts = len(mesh.vertices)
    uv_sum = np.zeros((num_verts, 2), dtype=np.float64)
    uv_cnt = np.zeros(num_verts, dtype=np.int32)

    np.add.at(uv_sum[:, 0], loop_verts, loop_uvs[:, 0])
    np.add.at(uv_sum[:, 1], loop_verts, loop_uvs[:, 1])
    np.add.at(uv_cnt, loop_verts, 1)

    valid = uv_cnt > 0
    avg_uv = np.zeros((num_verts, 2), dtype=np.float64)
    avg_uv[valid] = uv_sum[valid] / uv_cnt[valid][:, None]

    strands = []
    weights = np.zeros(num_verts, dtype=np.float32)

    for vid, v in enumerate(mesh.vertices):
        if not valid[vid]:
            continue

        uv = avg_uv[vid]
        w = sample_image_luminance_at_uv(mask_img, uv, pixels_np=pixels_np)
        weights[vid] = w

        if w <= threshold:
            continue

        root = v.co.copy()
        normal = v.normal.normalized()
        tip = root + normal * float(strand_length)

        strand = np.stack([
            np.array((root.x, root.y, root.z), dtype=np.float32),
            np.array((tip.x, tip.y, tip.z), dtype=np.float32),
        ], axis=0)
        strands.append(strand)

    old_vg = mesh_obj.vertex_groups.get(UNDERHAIR_MASK_GROUP)
    if old_vg is not None:
        mesh_obj.vertex_groups.remove(old_vg)
    vg = mesh_obj.vertex_groups.new(name=UNDERHAIR_MASK_GROUP)

    for vid, w in enumerate(weights):
        if w > 0.0:
            vg.add([vid], float(w), 'REPLACE')

    log(f"Created underhair guides: {len(strands)} strands")
    return strands, vg, mask_img


def setup_underhair_modifiers(curves_obj, mesh_obj, mask_img, trim_length=0.01):
    profile_mod = add_gn_modifier(curves_obj, "Set Hair Curve Profile", "UnderHair Profile")
    set_gn_input(profile_mod, "Radius", 0.0001)
    log("    + UnderHair Profile (radius=0.0001)")

    interp_mod = add_gn_modifier(curves_obj, "Interpolate Hair Curves", "UnderHair Interpolate")

    try:
        interp_mod["Input_2"] = mesh_obj
        log(f'    + UnderHair Interpolate Input_2 = {mesh_obj.name}')
    except Exception as e:
        log(f'    [WARN] Failed setting UnderHair Interpolate Input_2: {e}')

    set_gn_input(interp_mod, "Surface", mesh_obj)
    set_gn_input(interp_mod, "Surface Object", mesh_obj)
    set_gn_input(interp_mod, "Mesh", mesh_obj)

    density_set = False
    density_set |= set_gn_input(interp_mod, "Surface Density", UNDERHAIR_INTERP_DENSITY)
    density_set |= set_gn_input(interp_mod, "Density", UNDERHAIR_INTERP_DENSITY)
    density_set |= set_gn_input_by_keywords(interp_mod, ["density"], UNDERHAIR_INTERP_DENSITY)
    density_set |= set_gn_input_by_keywords(interp_mod, ["surface", "density"], UNDERHAIR_INTERP_DENSITY)

    if not density_set:
        log("    [WARN] Could not find interpolation density socket; value may remain default")

    set_gn_input(interp_mod, "Interpolation Quality", 6)
    set_gn_input(interp_mod, "Variation Level", 0.15)
    set_gn_input(interp_mod, "Guide Mask", 0.0)
    set_gn_input(interp_mod, "Use Guide Mask", False)

    interp_tex_id = get_node_input_identifier_by_name(interp_mod, "Mask Texture")
    if interp_tex_id is not None:
        try:
            interp_mod[interp_tex_id] = mask_img
            log("    + UnderHair Interpolate mask texture assigned")
        except Exception as e:
            log(f"    [WARN] Failed to assign interpolate mask texture: {e}")

    log("    + UnderHair Interpolate configured")

    noise_mod = add_gn_modifier(curves_obj, "Hair Curves Noise", "UnderHair Noise")
    set_gn_input(noise_mod, "Factor", 1.0)
    set_gn_input(noise_mod, "Distance", 0.0015)
    set_gn_input(noise_mod, "Shape", 0.5)
    set_gn_input(noise_mod, "Scale", 18.0)
    set_gn_input(noise_mod, "Scale Along Length", 1.0)
    set_gn_input(noise_mod, "Offset", 0.0)
    set_gn_input(noise_mod, "Cumulative Offset", False)
    set_gn_input(noise_mod, "Seed", 0)
    log("    + UnderHair Noise")

    frizz_mod = try_add_gn_modifier(
        curves_obj,
        ["Frizz Hair Curves", "Hair Curves Frizz"],
        "UnderHair Frizz"
    )
    if frizz_mod is not None:
        set_gn_input(frizz_mod, "Amount", 0.6)
        set_gn_input(frizz_mod, "Radius", 0.001)
        set_gn_input(frizz_mod, "Frequency", 12.0)
        set_gn_input(frizz_mod, "Seed", 0)
        set_gn_input(frizz_mod, "Mask", 1.0)

        frizz_tex_id = get_node_input_identifier_by_name(frizz_mod, "Mask Texture")
        if frizz_tex_id is not None:
            try:
                frizz_mod[frizz_tex_id] = mask_img
                log("    + UnderHair Frizz mask texture assigned")
            except Exception as e:
                log(f"    [WARN] Failed to assign frizz mask texture: {e}")

        log("    + UnderHair Frizz")

    trim_mod = add_gn_modifier(curves_obj, "Trim Hair Curves", "UnderHair Trim")
    set_gn_input(trim_mod, "Mask", 1.0)
    set_gn_input(trim_mod, "Random Offset", 0.0)
    set_gn_input(trim_mod, "Pin at Parameter", 0.0)
    set_gn_input(trim_mod, "Replace Length", True)
    set_gn_input(trim_mod, "Length", float(trim_length))

    trim_tex_id = get_node_input_identifier_by_name(trim_mod, "Mask Texture")
    if trim_tex_id is not None:
        try:
            trim_mod[trim_tex_id] = mask_img
            log("    + UnderHair Trim mask texture assigned")
        except Exception as e:
            log(f"    [WARN] Failed to assign trim mask texture: {e}")

    log(f"    + UnderHair Trim (length={trim_length})")


def create_underhair_on_mesh(mesh_obj, mesh_material, mask_image_path):
    log("Creating underhair from mesh surface...")

    strands, mask_vg, mask_img = create_underhair_guides_from_mesh(
        mesh_obj=mesh_obj,
        mask_image_path=str(mask_image_path),
        strand_length=UNDERHAIR_LENGTH,
        threshold=0.05,
    )

    if len(strands) == 0:
        log("  [WARN] No underhair strands created from mask.")
        return None

    underhair_obj = build_curve_object_from_strands_no_uv(
        obj_name="underhair",
        strands=strands,
        mesh_obj=mesh_obj,
    )
    if underhair_obj is None:
        return None

    remove_curve_uv_attributes(underhair_obj)
    assign_material(underhair_obj, mesh_material)

    setup_underhair_modifiers(
        curves_obj=underhair_obj,
        mesh_obj=mesh_obj,
        mask_img=mask_img,
        trim_length=UNDERHAIR_LENGTH,
    )

    log("Underhair object created successfully")
    return underhair_obj


def save_prediction_scene(obj_path, save_blend_path=None):
    """Save a prediction scene, packing its loaded image assets by default."""
    if not BLENDER_CONFIG["save_blend"]:
        return None
    output = blend_output_path(obj_path, save_blend_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if BLENDER_CONFIG["pack_assets"]:
        packed = bpy.ops.file.pack_all()
        if "FINISHED" not in packed:
            raise RuntimeError("Blender could not pack the scene's external assets")
    result = bpy.ops.wm.save_as_mainfile(filepath=str(output))
    if "FINISHED" not in result or not output.is_file():
        raise RuntimeError(f"Blender did not save the scene to {output}")
    log(f"Saved blend file: {output}")
    return output


def load_hair_scene(
    obj_path: str,
    mesh_object_name: str = "DeformedMesh",
    save_blend_path: str = None,
):
    t_start = time.perf_counter()
    if BLENDER_CONFIG["save_blend"]:
        blend_output_path(obj_path, save_blend_path)
    paths = resolve_paths_from_obj(obj_path)
    # Check inputs before changing the current Blender scene.
    require_file(paths["hair_meta"], "global hair anchor metadata")
    require_file(BLENDER_CONFIG["underhair_mask_path"], "underhair mask")
    if BLENDER_CONFIG["enable_texture_loading"]:
        require_file(paths["texture"], "predicted texture")
        require_file(paths["mesh_normal"], "normal map")
        require_file(paths["mesh_roughness"], "roughness map")
    if BLENDER_CONFIG["use_hdri"]:
        require_file(paths["hdr_path"], "HDRI")
    get_essentials_blend_path()
    scene = bpy.context.scene

    delete_default_cube()
    setup_cycles_render(scene)
    if BLENDER_CONFIG["use_hdri"]:
        setup_world_hdr(str(paths["hdr_path"]))


    log("Importing mesh...")
    mesh_obj = import_obj_as_mesh(obj_path, mesh_object_name)

    if not mesh_obj.data.uv_layers:
        require_file(BLENDER_CONFIG["uv_source_obj_path"], "UV fallback OBJ")
        try_load_uvs_from_reference_obj(mesh_obj, BLENDER_CONFIG["uv_source_obj_path"])

    if not mesh_obj.data.uv_layers:
        raise RuntimeError("Imported mesh has no UV map even after UV fallback. Hair placement requires UVs.")

    texture_path = None
    if BLENDER_CONFIG["enable_texture_loading"]:
        log("Resolving dataset texture...")
        texture_path = find_texture_for_obj(obj_path, str(paths["textures_dir"]))
        log(f"Using texture: {texture_path}")

        log("Creating mesh material...")
        mesh_mat = create_mesh_material(
            mesh_obj=mesh_obj,
            texture_path=str(texture_path),
            normal_path=str(paths["mesh_normal"]),
            roughness_path=str(paths["mesh_roughness"]),
            material_name=f"{mesh_object_name}_MeshMaterial",
        )

        log("Creating hair material...")
        hair_mat = create_hair_material(
            texture_path=str(texture_path),
            material_name=f"{mesh_object_name}_HairMaterial",
        )
    else:
        log("Texture loading disabled; using a gray material.")
        mesh_mat = create_simple_gray_material(
            mesh_obj,
            material_name=f"{mesh_object_name}_MeshMaterial_DisabledTexture",
        )
        hair_mat = mesh_mat

    log("Creating underhair...")
    underhair_obj = create_underhair_on_mesh(
        mesh_obj=mesh_obj,
        mesh_material=mesh_mat,
        mask_image_path=BLENDER_CONFIG["underhair_mask_path"],
    )

    if paths["density_dir"]:
        log("Loading optional density images...")
        load_density_images(mesh_obj, str(paths["density_dir"]))


    hair_map_path = str(paths["hair_map"])
    log(f"Loading UV-local 512 hair map: {hair_map_path}")
    log(f"Loading global hair meta: {paths['hair_meta']}")
    group_names, strands_by_group, samples = decode_hair_map_npz(
        hair_map_path,
        str(paths["hair_meta"]),
        mesh_obj,
    )
    log(f"  Samples per strand: {samples}")

    created_objects = []

    for group_name in group_names:
        if any(k in group_name.lower() for k in IGNORE_NAME_KEYWORDS):
            log(f"  [SKIP] Group '{group_name}' ignored by keyword")
            continue

        strands = strands_by_group.get(group_name)
        if strands is None or strands.shape[0] == 0:
            log(f"  [SKIP] Group '{group_name}' has 0 strands")
            continue

        log(f"  Processing group: {group_name} ({strands.shape[0]} strands)")

        roots = strands[:, 0, :]
        root_uvs = compute_root_uvs(roots, mesh_obj)

        curves_obj = build_curve_object_from_strands(
            sanitize_name(group_name),
            strands,
            root_uvs,
            mesh_obj,
        )
        if curves_obj is None:
            continue

        assign_material(curves_obj, hair_mat)

        log(f"  Setting up modifiers for: {curves_obj.name}")
        setup_modifiers(curves_obj, group_name, mesh_obj)

        created_objects.append(curves_obj)

    if underhair_obj is not None:
        created_objects.append(underhair_obj)

    loaded_hair_map = str(paths["hair_map_curl_out"])
    log(f"Loaded hair map from: {loaded_hair_map}")

    save_prediction_scene(obj_path, save_blend_path)

    elapsed = time.perf_counter() - t_start
    log(f"Done. Created {len(created_objects)} curve objects in {elapsed:.2f}s")
    for obj in created_objects:
        log(f"  {obj.name}")

    return mesh_obj, created_objects





def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Load an AnimalLift prediction into Blender")
    parser.add_argument("--results_dir", help="Result folder containing meshes/, textures/, and hair/")
    parser.add_argument("--sample", help="Input image stem, e.g. animal_2; defaults to first result")
    parser.add_argument("--obj_path", help="Explicit *_mesh.obj; overrides folder/sample selection")
    parser.add_argument("--mesh_object_name", default="DeformedMesh")
    parser.add_argument("--save_blend_path", help="Output .blend file; default: <results>/blender/<sample>.blend")
    parser.add_argument("--save_blend", action=argparse.BooleanOptionalAction,
                        help="Save the loaded scene automatically (enabled by default)")
    parser.add_argument("--use_hdri", action=argparse.BooleanOptionalAction)
    parser.add_argument("--pack_assets", action=argparse.BooleanOptionalAction)
    return parse_configured_args(parser, "blender", argv)


def main():
    global BLENDER_CONFIG
    user_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(user_args)
    BLENDER_CONFIG = vars(args)
    obj_path = select_result_obj(args.results_dir, args.sample, args.obj_path)
    log(f"Loading {obj_path}")
    load_hair_scene(obj_path, args.mesh_object_name, args.save_blend_path)


if __name__ == "__main__":
    main()
