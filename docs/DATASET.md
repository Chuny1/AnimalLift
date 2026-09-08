# Training data layout

The input examples in `input_images/` are for inference and qualitative validation. Full training requires paired controls, UV textures, meshes, and hair maps.

`train.base_root` defaults to `data/Animal_Textures`. For each configured animal (`dog`, `small_cat`, `big_cat`, `wolf`, `fox`, `bear`), the loader uses:

```text
data/Animal_Textures/<animal>/dataset/<group>/
    augmented_images/
    textures/
    shapes/
    hair_maps_single_512_uvlocal/
    hair_info.json
```

Each of the four subdirectories contains the files described below. `hair_info.json` is adjacent to them. Set `train.guidance_dirname` to `render_images` if that is how your controls are organized; the default is `augmented_images`. The other directory names are configurable too.

| Item | Format and matching behavior |
| --- | --- |
| `augmented_images/` | Control PNG/JPG/JPEG/WEBP images. Training image extensions must be lowercase. |
| `textures/` | Paired UV PNG/JPG/JPEG/WEBP images. Controls and textures are independently naturally sorted and paired by index, **not by filename**. The loader uses the smaller count if they differ; provide equal counts and matching ordering to avoid silently dropping pairs. |
| `shapes/` | Reference-topology OBJ meshes. A single OBJ is reused for the group; otherwise files are naturally sorted and selected by pair index. All samples must use the same vertex order/topology for the shared output head and mean mesh. |
| `hair_maps_single_512_uvlocal/` | Quantized hair `.pt` caches, or `.npz` maps. The loader prefers `.pt` files whenever any exist in the directory, so avoid incomplete cache sets. One file can be reused for the group; otherwise pair by natural index. |
| `hair_info.json` | Region length/curl labels. Missing files/regions produce invalid-label masks and skip that supervision. Provide these annotations to supervise the region heads. |

A hair NPZ provides `hair_offsets_local_q` (int8, `[H,W,S-1,3]`) and `offset_scale` (three-component scale). The exported inference NPZ also carries `samples` for Blender. A cached PT provides `hair_q` (int8 tensor), `scale` (float tensor), and `hair_mask` (boolean `[H,W]` tensor).

Training region labels are read from `objects[<OBJ stem>].metadata.regions`, then `objects[<OBJ stem>].maxima`, with `resolved_hair_params` as a fallback. Each region uses `length_max` (or `length`), `curl_type`, and/or `curl_radius`. The default ten regions are `face`, `ear`, `neck`, `body`, `leg`, `tail`, `eye_brow`, `fore_head`, `nose`, and `jaw`.

The loader scans immediate group directories by default. Use `--recursive_dataset` if groups are nested deeper. For a single dataset root use:

```bash
python train.py --no-all_animals --data_root data/Animal_Textures/dog/dataset
```

Set `train.uv_mask_path` to the UV occupancy image, `shared.reference_mesh_path` to the matching OBJ, and `train.flat_val_image_dir` to a folder of qualitative validation images (defaults to the included examples). Static Blender hair anchors are not read during training; training directly reads the quantized per-sample hair maps.
