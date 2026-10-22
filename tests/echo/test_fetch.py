"""A fresh asset download must produce the scene paths used by evaluation."""
from io import BytesIO
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from echo.data.fetch import main, unpack_assets
from echo.io import atomic_json


def write_archive(directory, extra=None, omit_state=False):
    payloads = {}
    for suite in ("spatial", "object", "goal", "10", "90"):
        stem = f"selected_bddl_scenes_{suite}/task/episode_001/step_0001"
        payloads[stem + ".bddl"] = b"scene definition"
        if not omit_state:
            payloads[stem + "_sim_state.npy"] = b"recorded state"
    with tarfile.open(directory / "scenes.tar.gz", "w:gz") as archive:
        for name, content in payloads.items():
            info = tarfile.TarInfo("scenes/" + name)
            info.size = len(content)
            archive.addfile(info, BytesIO(content))
        if extra is not None:
            archive.addfile(extra, BytesIO(b"x") if extra.isfile() else None)
    return payloads


class AssetDownloads(unittest.TestCase):
    def test_fetch_cli_prepares_all_suites_and_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "data.json"
            atomic_json(config, {"assets": {"local_dir": str(root)}})
            payloads = write_archive(root)
            with patch("echo.data.fetch.fetch_source") as download, patch.object(
                    sys, "argv", ["fetch", "--config", str(config), "--source", "assets"]):
                main()
                download.assert_called_once_with({"local_dir": str(root)})
            for name, content in payloads.items():
                self.assertEqual((root / name).read_bytes(), content)
            missing = root / next(iter(payloads))
            missing.unlink()
            unpack_assets(root)
            self.assertTrue(missing.is_file())
            missing.write_bytes(b"local modification")
            with self.assertRaisesRegex(ValueError, "Refuse replacing"):
                unpack_assets(root)
            self.assertEqual(missing.read_bytes(), b"local modification")
            self.assertFalse(list(root.rglob(".scene-*")))

    def test_unsafe_entries_and_incomplete_pairs_are_rejected(self):
        extras = []
        traversal = tarfile.TarInfo("scenes/../escaped.bddl")
        traversal.size = 1
        extras.append(traversal)
        symlink = tarfile.TarInfo("scenes/selected_bddl_scenes_spatial/link.bddl")
        symlink.type = tarfile.SYMTYPE
        symlink.linkname = "../../escaped.bddl"
        extras.append(symlink)
        for extra in extras:
            with self.subTest(entry=extra.name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_archive(root, extra)
                with self.assertRaises(ValueError):
                    unpack_assets(root)
                self.assertEqual(list(root.iterdir()), [root / "scenes.tar.gz"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_archive(root, omit_state=True)
            with self.assertRaisesRegex(ValueError, "pairs"):
                unpack_assets(root)


if __name__ == "__main__":
    unittest.main()
