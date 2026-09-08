"""Check release-critical portability without requiring torch or Blender."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from animallift.config import load_config, resolve_path, result_paths, select_result_obj
import train
import test as test_entry


class PortabilityTests(unittest.TestCase):
    def test_portable_defaults(self):
        args = test_entry.parse_args(["--config", "config.json"])
        self.assertEqual(args.resume, str(ROOT / "ckpt/latest.pt"))
        self.assertEqual(args.flat_val_image_dir, str(ROOT / "input_images"))
        self.assertEqual(args.reference_mesh_path, str(ROOT / "assets/reference.obj"))
        self.assertEqual(args.module_name, "animallift.model")

    def test_explicit_config_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "server.json"
            config.write_text(json.dumps({
                "test": {"flat_val_image_dir": "alternate_images", "resume": "ckpt/released.pt"},
                "shared": {"vae_local_path": "assets/vae", "device": "cpu"},
            }))
            args = test_entry.parse_args(["--config", str(config), "--input_dir", "my_images"])
            self.assertEqual(args.flat_val_image_dir, str(ROOT / "my_images"))
            self.assertEqual(args.resume, str(ROOT / "ckpt/released.pt"))
            self.assertEqual(args.vae_local_path, str(ROOT / "assets/vae"))
            self.assertEqual(args.device, "cpu")
            self.assertEqual(args.output_dir, str(ROOT / "outputs/test"))

    def test_automatic_local_config_and_explicit_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "local.json"
            local.write_text(json.dumps({"test": {"resume": "ckpt/local.pt"}}))
            with patch("animallift.config.LOCAL_CONFIG", local):
                self.assertEqual(load_config()["test"]["resume"], str(ROOT / "ckpt/local.pt"))
                self.assertEqual(load_config("config.json")["test"]["resume"], str(ROOT / "ckpt/latest.pt"))

    def test_environment_paths_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"ANIMALLIFT_TEST_ROOT": tmp}):
                self.assertEqual(resolve_path("${ANIMALLIFT_TEST_ROOT}/data"), str(Path(tmp) / "data"))
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    resolve_path("${UNDEFINED_ANIMALLIFT_PATH}/data")
            config = Path(tmp) / "bad.json"
            config.write_text('{"test": {"typo_path": "x"}}')
            with self.assertRaises(ValueError):
                load_config(str(config))

    def test_train_single_species_and_hair_regions(self):
        args = train.parse_args(["--config", "config.json", "--no-all_animals", "--data_root", "dataset"])
        self.assertFalse(args.all_animals)
        self.assertEqual(args.data_root, str(ROOT / "dataset"))
        self.assertEqual(len(args.hair_region_names), 10)
        self.assertEqual(args.hair_region_names[-4:], ["eye_brow", "fore_head", "nose", "jaw"])
        self.assertIsNone(args.resume)

    def test_paths_from_other_working_directory(self):
        script = "import sys;sys.path.insert(0,sys.argv[1]);import test;a=test.parse_args(['--config','config.json','--input_dir','input_images']);print(a.flat_val_image_dir)"
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, "-c", script, str(ROOT)], cwd=tmp, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.strip(), str(ROOT / "input_images"))
            for name in ("train.py", "test.py", "dataset_generation/gen_dog_dataset.py"):
                subprocess.run([sys.executable, str(ROOT / name), "--help"], cwd=tmp, check=True, capture_output=True)

    def test_result_folder_move_and_sample_selection(self):
        with tempfile.TemporaryDirectory(prefix="meshes_parent_") as tmp:
            result = Path(tmp) / "copied results"
            meshes = result / "meshes"
            meshes.mkdir(parents=True)
            for name in ("animal_2", "animal_92"):
                (meshes / (name + "_mesh.obj")).touch()
            chosen = select_result_obj(str(result), "animal_92")
            paths = result_paths(chosen)
            self.assertEqual(paths["hair_map"], result / "hair/animal_92_hair.npz")
            self.assertEqual(paths["texture"], result / "textures/animal_92_uv.png")
            with contextlib.redirect_stdout(io.StringIO()):
                first = select_result_obj(str(result))
            self.assertEqual(Path(first).name, "animal_2_mesh.obj")
            with self.assertRaises(FileNotFoundError):
                select_result_obj(str(result), "missing")
            with self.assertRaises(ValueError):
                select_result_obj(str(result), "../outside")
            self.assertEqual(select_result_obj(None, obj_path=chosen), chosen)

    def test_missing_checkpoint_message_without_ml_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / "test.py"), "--config", "config.json", "--checkpoint", str(Path(tmp) / "missing.pt")], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("download into ckpt/", result.stderr)
            self.assertNotIn("No module named 'torch'", result.stderr)


if __name__ == "__main__":
    unittest.main()
