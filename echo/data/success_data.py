"""Read nominal LIBERO LeRobot-v2 frames as PI05 action-chunk examples.

Images retain their stored orientation. At episode tails, zero *raw* actions
before checkpoint quantile normalization. No torch, model, or CUDA imports.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from io import BytesIO
import json
import operator
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_quantile(x, stats):
    """Match pi_libero_common.normalize_quantile, including epsilon/no clip."""
    q01 = stats["q01"][..., : x.shape[-1]]
    q99 = stats["q99"][..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def _jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _positive_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


class SuccessChunkDataset:
    """Map-style dataset: one item per real frame, including terminal tails.

    ``norm_stats`` must already be parsed as ``{'actions': {...}, 'state':
    {...}}`` with q01/q99 vectors; checkpoint files are never read implicitly.
    ``padding_mask`` is True only for artificial future action rows. Padded
    actions remain training targets; this class does not create a loss mask.
    """

    def __init__(self, dataset_root, norm_stats, horizon=50,
                 episode_indices=None, cache_episodes=2, column_layout="physical_intelligence"):
        import pyarrow.parquet as pq  # Imported only when constructing a reader.

        self._pq = pq
        layouts = {
            "physical_intelligence": {
                "version": "v2.0",
                "columns": {},
            },
            "lerobot_embedded": {
                "version": "v2.1",
                "columns": {
                    "image": "observation.images.image",
                    "wrist_image": "observation.images.wrist_image",
                    "state": "observation.state",
                    "actions": "action",
                },
            },
        }
        if column_layout not in layouts:
            raise ValueError(f"Unsupported success column layout: {column_layout}")
        self.column_layout = column_layout
        self.column_names = layouts[column_layout]["columns"].copy()
        expected_version = layouts[column_layout]["version"]
        self.root = Path(dataset_root).resolve(strict=True)
        self.horizon = _positive_int(horizon, "horizon")
        self.cache_episodes = _positive_int(cache_episodes, "cache_episodes")
        self._cache = OrderedDict()
        self.norm_stats = {}
        for name, dims in (("actions", 7), ("state", 8)):
            stats = {}
            for key in ("q01", "q99"):
                try:
                    value = np.asarray(norm_stats[name][key], dtype=np.float32)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Missing/invalid norm_stats[{name!r}][{key!r}]") from exc
                if value.ndim != 1 or len(value) < dims or not np.isfinite(value).all():
                    raise ValueError(f"{name}.{key} must be a finite vector of at least {dims} values")
                stats[key] = value[:dims].copy()
            if np.any(stats["q99"] < stats["q01"]):
                raise ValueError(f"{name}: q99 must not be below q01")
            self.norm_stats[name] = stats

        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info.get("codebase_version") != expected_version:
            raise ValueError(f"Expected {column_layout} dataset version {expected_version}")
        features = {
            key: self.info["features"].get(self.column_names.get(key, key), {})
            for key in ("state", "actions", "image", "wrist_image")
        }
        for key, shape, dtype in (("state", [8], "float32"), ("actions", [7], "float32")):
            if features.get(key, {}).get("shape") != shape or features[key].get("dtype") != dtype:
                raise ValueError(f"Invalid metadata feature {key}")
        self.image_shapes = {}
        for key in ("image", "wrist_image"):
            feature = features.get(key, {})
            shape = feature.get("shape", [])
            if feature.get("dtype") != "image" or len(shape) != 3 or shape[-1] != 3 or any(s <= 0 for s in shape):
                raise ValueError(f"Invalid metadata image feature {key}")
            self.image_shapes[key] = tuple(shape)
        self.chunk_size = _positive_int(self.info["chunks_size"], "chunks_size")
        tasks = _jsonl(self.root / "meta/tasks.jsonl")
        self.tasks = {}
        for task in tasks:
            task_id = task["task_index"]
            if type(task_id) is not int or task_id < 0 or task_id in self.tasks or not isinstance(task["task"], str) or not task["task"].strip():
                raise ValueError("Invalid or duplicate task metadata")
            self.tasks[task_id] = task["task"]
        if len(tasks) != self.info["total_tasks"]:
            raise ValueError("Task metadata count disagrees with info.json")
        episodes = _jsonl(self.root / "meta/episodes.jsonl")
        if len(episodes) != self.info["total_episodes"]:
            raise ValueError("Episode metadata count disagrees with info.json")
        self.episode_meta = {}
        for episode in episodes:
            ep = episode["episode_index"]
            if type(ep) is not int or ep < 0 or ep in self.episode_meta:
                raise ValueError("Invalid or duplicate episode index")
            _positive_int(episode["length"], "episode length")
            if not episode.get("tasks") or any(t not in self.tasks.values() for t in episode["tasks"]):
                raise ValueError(f"Unknown/empty task for episode {ep}")
            self.episode_meta[ep] = episode
        if set(self.episode_meta) != set(range(self.info["total_episodes"])):
            raise ValueError("Episode metadata indices must be consecutive from zero")
        if sum(ep["length"] for ep in episodes) != self.info["total_frames"]:
            raise ValueError("Episode lengths disagree with total_frames")
        selected = sorted(self.episode_meta) if episode_indices is None else list(episode_indices)
        if not selected or any(isinstance(ep, bool) or not isinstance(ep, (int, np.integer)) for ep in selected):
            raise ValueError("episode_indices must contain integers and be nonempty")
        self.episode_indices = [int(ep) for ep in selected]
        if len(set(selected)) != len(selected) or any(ep not in self.episode_meta for ep in selected):
            raise ValueError("Duplicate or unknown selected episode index")
        self.paths, self.offsets = {}, [0]
        for ep in self.episode_indices:
            relative = self.info["data_path"].format(episode_chunk=ep // self.chunk_size, episode_index=ep)
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError(f"Dataset path escapes its root: {relative}")
            if not path.is_file():
                raise FileNotFoundError(path)
            self.paths[ep] = path
            self.offsets.append(self.offsets[-1] + self.episode_meta[ep]["length"])

    def __len__(self):
        return self.offsets[-1]

    def __getstate__(self):
        result = self.__dict__.copy()
        result.pop("_pq")
        result["_cache"] = OrderedDict()
        return result

    def __setstate__(self, state):
        import pyarrow.parquet as pq
        self.__dict__.update(state)
        self._pq = pq

    def _load_episode(self, ep):
        if ep in self._cache:
            self._cache.move_to_end(ep)
            return self._cache[ep]
        columns = ["image", "wrist_image", "state", "actions", "episode_index", "frame_index", "task_index"]
        source_columns = [self.column_names.get(key, key) for key in columns]
        table = self._pq.read_table(self.paths[ep], columns=source_columns)
        # Rename Arrow columns in memory; image bytes and numeric values are
        # unchanged. This preserves the native action and image convention.
        table = table.rename_columns(columns)
        length = self.episode_meta[ep]["length"]
        if len(table) != length:
            raise ValueError(f"Episode {ep}: parquet length does not match metadata")
        for key, expected in (("episode_index", np.full(length, ep)), ("frame_index", np.arange(length))):
            actual = table[key].to_numpy()
            if actual.dtype.kind not in "iu" or not np.array_equal(actual, expected):
                raise ValueError(f"Episode {ep}: invalid {key}")
        arrays = {}
        for key, dims in (("state", 8), ("actions", 7)):
            array = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if array.shape != (length, dims) or not np.isfinite(array).all():
                raise ValueError(f"Episode {ep}: invalid {key} values/shape")
            arrays[key] = array
        task_ids = table["task_index"].to_numpy()
        if task_ids.dtype.kind not in "iu" or any(int(t) not in self.tasks for t in task_ids):
            raise ValueError(f"Episode {ep}: invalid task_index")
        actual_tasks = {self.tasks[int(t)] for t in task_ids}
        if actual_tasks != set(self.episode_meta[ep]["tasks"]):
            raise ValueError(f"Episode {ep}: parquet task labels disagree with metadata")
        result = (table, arrays["state"], arrays["actions"], task_ids)
        self._cache[ep] = result
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return result

    def _decode_image(self, value, key):
        # Hugging Face Image is a struct {bytes, path}. The pinned LIBERO
        # release embeds image bytes; external/relative image paths are rejected.
        if not isinstance(value, dict) or not value.get("bytes"):
            raise ValueError(f"{key} must contain embedded image bytes")
        with Image.open(BytesIO(value["bytes"])) as source:
            image = source.convert("RGB").copy()
        h, w, _ = self.image_shapes[key]
        if image.size != (w, h):
            raise ValueError(f"{key} image shape disagrees with metadata: {image.size}")
        return image

    def __getitem__(self, index):
        if isinstance(index, bool):
            raise TypeError("Dataset index must be an integer")
        index = operator.index(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        position = bisect_right(self.offsets, index) - 1
        ep = self.episode_indices[position]
        frame = index - self.offsets[position]
        table, states, actions, task_ids = self._load_episode(ep)
        valid = min(self.horizon, len(actions) - frame)
        raw_chunk = np.zeros((self.horizon, 7), dtype=np.float32)
        raw_chunk[:valid] = actions[frame:frame + valid]
        return {
            "image": [self._decode_image(table[key][frame].as_py(), key) for key in ("image", "wrist_image")],
            "lang": self.tasks[int(task_ids[frame])],
            "state": normalize_quantile(states[frame:frame + 1], self.norm_stats["state"]),
            "action": normalize_quantile(raw_chunk, self.norm_stats["actions"]),
            "episode_index": ep,
            "frame_index": frame,
            "valid_action_count": valid,
            "padding_mask": np.arange(self.horizon) >= valid,
        }
