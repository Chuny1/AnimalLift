"""Test Blender save orchestration with a mocked Blender API."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from animallift.config import blend_output_path


def load_script():
    bpy = ModuleType("bpy")
    mathutils = ModuleType("mathutils")
    mathutils.Vector = Mock()
    bvhtree = ModuleType("mathutils.bvhtree")
    bvhtree.BVHTree = Mock()
    geometry = ModuleType("mathutils.geometry")
    geometry.barycentric_transform = Mock()
    modules = {"bpy": bpy, "bmesh": ModuleType("bmesh"), "mathutils": mathutils,
               "mathutils.bvhtree": bvhtree, "mathutils.geometry": geometry,
               "numpy": ModuleType("numpy")}
    spec = importlib.util.spec_from_file_location("blender_save_under_test", ROOT / "load_blender.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    module.log = Mock()
    return module


class BlenderSaveTests(unittest.TestCase):
    def setUp(self):
        self.loader = load_script()
        self.loader.BLENDER_CONFIG = {"save_blend": True, "pack_assets": True}
        self.pack = Mock(return_value={"FINISHED"})
        self.save = Mock()
        self.loader.bpy.ops = SimpleNamespace(file=SimpleNamespace(pack_all=self.pack),
                                             wm=SimpleNamespace(save_as_mainfile=self.save))

    def test_default_and_cli_disable(self):
        self.assertTrue(self.loader.parse_args(["--config", "config.json"]).save_blend)
        self.assertFalse(self.loader.parse_args(["--config", "config.json", "--no-save_blend"]).save_blend)

    def test_automatic_save_for_moved_result_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = Path(tmp) / "copied results/meshes/animal_2_mesh.obj"
            expected = Path(tmp) / "copied results/blender/animal_2.blend"
            events = []
            self.pack.side_effect = lambda: events.append("pack") or {"FINISHED"}
            def save(filepath):
                events.append("save")
                Path(filepath).write_bytes(b"mock scene")
                return {"FINISHED"}
            self.save.side_effect = save
            output = self.loader.save_prediction_scene(str(obj))
            self.assertEqual(output, expected)
            self.assertTrue(expected.is_file())
            self.assertEqual(events, ["pack", "save"])
            self.save.assert_called_once_with(filepath=str(expected))

    def test_explicit_output_without_packing(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp) / "custom/scene.blend"
            self.loader.BLENDER_CONFIG["pack_assets"] = False
            def save(filepath):
                Path(filepath).touch()
                return {"FINISHED"}
            self.save.side_effect = save
            self.assertEqual(self.loader.save_prediction_scene("unused", str(expected)), expected)
            self.pack.assert_not_called()

    def test_disabled_save_has_no_filesystem_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.loader.BLENDER_CONFIG["save_blend"] = False
            output = Path(tmp) / "unused/scene.blend"
            self.assertIsNone(self.loader.save_prediction_scene("unused", str(output)))
            self.assertFalse(output.parent.exists())
            self.pack.assert_not_called()
            self.save.assert_not_called()

    def test_cancelled_pack_and_save_are_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "scene.blend")
            self.pack.return_value = {"CANCELLED"}
            with self.assertRaisesRegex(RuntimeError, "pack"):
                self.loader.save_prediction_scene("unused", output)
            self.save.assert_not_called()
            self.pack.return_value = {"FINISHED"}
            for status in ({"CANCELLED"}, {"FINISHED"}):
                self.save.return_value = status
                with self.assertRaisesRegex(RuntimeError, "did not save"):
                    self.loader.save_prediction_scene("unused", output)

    def test_invalid_output_extension(self):
        with self.assertRaisesRegex(ValueError, "end with .blend"):
            blend_output_path("unused", "outputs/scene.obj")


if __name__ == "__main__":
    unittest.main()
