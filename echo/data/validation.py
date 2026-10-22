"""Validate every selected episode, decode every image, and freeze input bytes."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from echo.data.chunks import ChunkDataset, convert_actions
from echo.io import atomic_json, inside, read_json, sha256
from echo.data.video import _VideoHandle


def episode_files(segment, sources):
    root = Path(sources[segment["source"]]["local_dir"])
    if segment["storage"] == "rollout_npz":
        names = [segment["episode_path"]+"/"+name for name in
                 ("trajectory.npz", "episode.json", "agentview.mp4", "wrist.mp4", "files.json")]
    else:
        info = read_json(root/"meta/info.json")
        episode = segment["episode_index"]
        params = dict(episode_chunk=episode//info["chunks_size"], episode_index=episode)
        names = ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl", info["data_path"].format(**params)]
        if segment["storage"] == "lerobot_video":
            names += [info["video_path"].format(**params, video_key=view)
                      for view in ("observation.images.image", "observation.images.wrist_image")]
    return {name: inside(root, name) for name in names}


def validation_code():
    return {name: sha256(Path(__file__).parent/name)
            for name in ("validation.py", "chunks.py", "success_data.py", "video.py")}


def validate(manifest, data_config, statistics, output, method="echo"):
    # This reader uses only NumPy/Arrow/Pillow/video codecs; no CUDA context.
    raw = read_json(statistics)
    key = "openpi_pi05_libero_franka"
    if key not in raw:
        if len(raw) != 1:
            raise ValueError("Ambiguous PI05 normalization statistics")
        key = next(iter(raw))
    # Public PI05 checkpoints store singular "action"; the reader uses the
    # same plural "actions" interface as the upstream statistics loader.
    stats = {"state": raw[key]["state"], "actions": raw[key]["action"]}
    dataset = ChunkDataset(manifest, data_config, stats, method=method, cache_episodes=1, load_images=False)
    episodes, frames, files = Counter(), Counter(), {}
    for index, segment in enumerate(dataset.segments):
        states, actions, images = dataset._load(segment)
        start, end = segment["boundary"]["start"], segment["boundary"]["end"]
        convert_actions(actions[start:end], segment["source_gripper_convention"])
        if isinstance(images, tuple):
            reader, table = images
            for frame in range(start, end):
                for view in ("image", "wrist_image"):
                    reader._decode_image(table[view][frame].as_py(), view)
        paths = episode_files(segment, dataset.sources)
        if not isinstance(images, tuple):
            for name, path in paths.items():
                if path.suffix == ".mp4":
                    handle = _VideoHandle(path, len(actions), prefer_pyav=True)
                    try:
                        # The PyAV/fallback path decodes the entire video and
                        # rejects a short stream, including corrupt tail frames.
                        for frame in (start, (start+end-1)//2, end-1):
                            pixels = handle.read(frame)
                            if pixels.ndim != 3 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
                                raise ValueError("Invalid decoded RGB frame")
                    finally:
                        handle.release()
        source = segment["source"]
        if segment["storage"] == "rollout_npz":
            expected = read_json(paths[segment["episode_path"]+"/files.json"])
            if set(expected) != {"trajectory.npz", "episode.json", "agentview.mp4", "wrist.mp4"}:
                raise ValueError("Unexpected failure export inventory")
            for name, record in expected.items():
                path = paths[segment["episode_path"]+"/"+name]
                if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                    raise ValueError("Failure export checksum differs")
        for name, path in paths.items():
            identity = source+"/"+name
            if identity not in files:
                files[identity] = dict(source=source, path=name, bytes=path.stat().st_size, sha256=sha256(path))
        episodes[source] += 1
        frames[source] += end-start
        if (index+1) % 100 == 0:
            print(f"Validated {index+1} / {len(dataset.segments)} segments", flush=True)
    for source in episodes:
        expected = dataset.sources[source]
        if episodes[source] != expected["episodes"] or frames[source] != expected["frames"]:
            raise ValueError(f"Frozen counts differ for {source}")
    receipt = dict(schema_version=1, method=method, manifest_sha256=sha256(manifest),
                   data_config_sha256=sha256(data_config), statistics_sha256=sha256(statistics),
                   validation_code=validation_code(), episodes=dict(episodes), frames=dict(frames), files=files,
                   scope="all selected numeric arrays, languages, boundaries, decoded frames and source file SHA-256")
    atomic_json(output, receipt)
    print(f"Validated {sum(episodes.values())} segments | {sum(frames.values())} frames | {len(files)} files")
    return receipt


def verify_receipt(path, manifest, data_config, statistics, method):
    receipt = read_json(path)
    if (receipt["schema_version"] != 1 or
            receipt["method"] not in ({"echo"} if method == "echo" else {"echo", "baseline"}) or
            receipt["manifest_sha256"] != sha256(manifest) or
            receipt["data_config_sha256"] != sha256(data_config) or
            receipt["statistics_sha256"] != sha256(statistics) or receipt["validation_code"] != validation_code()):
        raise ValueError("Data validation receipt differs; rerun echo.data.validation")
    sources = read_json(data_config)["sources"]
    for record in receipt["files"].values():
        target = inside(sources[record["source"]]["local_dir"], record["path"])
        if target.stat().st_size != record["bytes"] or sha256(target) != record["sha256"]:
            raise ValueError("Validated source bytes changed; rerun echo.data.validation")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/training.json.gz")
    parser.add_argument("--config", default="configs/data.json")
    parser.add_argument("--statistics", default="data/pretrained/pi05_libero_starvla/dataset_statistics.json")
    parser.add_argument("--output", default="data/training_validation.json")
    parser.add_argument("--method", choices=["echo", "baseline"], default="echo")
    args = parser.parse_args()
    validate(args.manifest, args.config, args.statistics, args.output, args.method)


if __name__ == "__main__":
    main()
