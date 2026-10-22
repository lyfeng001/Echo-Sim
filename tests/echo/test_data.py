"""Read and validate published failure episodes; reject altered downloads."""
from io import BytesIO
from pathlib import Path
import tarfile
import tempfile
import unittest

import imageio.v2 as imageio
import numpy as np

from echo.data.chunks import ChunkDataset
from echo.data.failure_archive import unpack_failure
from echo.data.validation import validate, verify_receipt
from echo.io import atomic_json, sha256


class DataContracts(unittest.TestCase):
    def test_failure_reader_validation_and_content_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root/"episodes/failure_000000"
            episode.mkdir(parents=True)
            actions = np.arange(28, dtype=np.float32).reshape(4, 7)/28
            states = np.arange(32, dtype=np.float32).reshape(4, 8)/32
            np.savez(episode/"trajectory.npz", actions=actions, states=states)
            frames = np.random.default_rng(42).integers(0, 256, (4, 16, 16, 3), dtype=np.uint8)
            for name in ("agentview.mp4", "wrist.mp4"):
                imageio.mimwrite(episode/name, frames, fps=20, codec="libx264", macro_block_size=1)
            instruction = "put the object in the bowl"
            atomic_json(episode/"episode.json", dict(length=4, success=False,
                        terminal_reason="timeout", source_lang=instruction))
            atomic_json(episode/"files.json", {
                name: dict(sha256=sha256(episode/name), bytes=(episode/name).stat().st_size)
                for name in ("trajectory.npz", "episode.json", "agentview.mp4", "wrist.mp4")})
            segment = dict(segment_id="s00000", segment_type="failure", source="failure",
                           storage="rollout_npz", episode_path="episodes/failure_000000",
                           boundary=dict(start=0, end=4, reason="timeout"),
                           source_lang=instruction, canonical_prompt=instruction,
                           source_gripper_convention="libero_env_signed")
            manifest, config, statistics, receipt = [root/name for name in
                ("manifest.json", "data.json", "statistics.json", "validation.json")]
            atomic_json(manifest, dict(schema_version=1, horizon=50, segments=[segment]))
            atomic_json(config, {"sources": {"failure": dict(local_dir=str(root), episodes=1, frames=4)}})
            stats = {"actions": dict(q01=[0]*7, q99=[1]*7), "state": dict(q01=[0]*8, q99=[1]*8)}
            atomic_json(statistics, {"openpi_pi05_libero_franka": {"action": stats["actions"], "state": stats["state"]}})
            validate(manifest, config, statistics, receipt)
            verify_receipt(receipt, manifest, config, statistics, "echo")
            dataset = ChunkDataset(manifest, config, stats, load_images=False)
            tail = dataset[3]
            np.testing.assert_array_equal(tail["echo_ss"], [0]+[1]*49)
            self.assertFalse(tail["echo_action_mask"].any())
            with (episode/"wrist.mp4").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(ValueError):
                verify_receipt(receipt, manifest, config, statistics, "echo")

    def test_downloaded_shards_preserve_bytes_and_reject_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {"episodes/failure_000000/trajectory.npz": b"numeric bytes",
                        "episodes/failure_000000/agentview.mp4": b"video bytes"}
            import hashlib
            inventory = {name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                         for name, data in payloads.items()}
            atomic_json(root/"inventory.json", inventory)
            shard = root/"shards/failure_00000.tar"
            shard.parent.mkdir()
            with tarfile.open(shard, "w", format=tarfile.USTAR_FORMAT) as archive:
                for name, data in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, BytesIO(data))
            atomic_json(root/"archives.json", dict(schema_version=1, format="echo_failure_shards_v1",
                        inventory_sha256=sha256(root/"inventory.json"), objects=len(payloads),
                        archives=[dict(path="shards/failure_00000.tar", bytes=shard.stat().st_size,
                                       sha256=sha256(shard), objects=len(payloads))]))
            unpack_failure(root)
            unpack_failure(root)  # Repeat downloads can reuse an identical extraction.
            for name, data in payloads.items():
                self.assertEqual((root/name).read_bytes(), data)
            with shard.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                unpack_failure(root)


if __name__ == "__main__":
    unittest.main()
