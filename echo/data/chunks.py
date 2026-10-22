"""Three-type training data, segment boundaries, native actions and SS/FS targets."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from echo.io import inside, read_json
from echo.data.success_data import SuccessChunkDataset, normalize_quantile
from echo.data.video import _VideoHandle


def signal_targets(kind, length, local, horizon=50):
    """Return padding, targets, masks and bounds with historical reductions."""
    if kind not in {"success", "failure", "recovery"} or not 0 <= local < length:
        raise ValueError("Invalid data type or local frame")
    padding = np.arange(horizon) >= min(horizon, length-local)
    ss = padding.astype(np.float32)
    ss_mask = np.ones(horizon, bool)
    fs = np.zeros(horizon, np.float32)
    fs_mask = np.ones(horizon, bool)
    if kind == "recovery":
        zeros = horizon if length <= 1 else round(horizon * local / (length-1))
        fs[zeros:] = 1
    elif kind == "failure":
        fs = padding.astype(np.float32)
        fs_mask[:] = padding.any()
    action_mask = np.full((horizon, 7), kind != "failure", dtype=bool)
    action_bound = ~padding if kind == "recovery" else np.ones(horizon, bool)
    return (padding, ss, ss_mask, fs, fs_mask, action_mask,
            action_bound, np.ones(horizon, bool), np.ones(horizon, bool))


def convert_actions(actions, convention):
    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("Source actions must be finite [N,7]")
    if convention == "dataset_binary_01":
        if np.any((actions[:, 6] < 0) | (actions[:, 6] > 1)):
            raise ValueError("Binary gripper must be in [0,1]")
        actions[:, 6] = 1 - 2 * actions[:, 6]
    elif convention != "libero_env_signed":
        raise ValueError("Unknown source gripper convention")
    return actions


class ChunkDataset:
    """Map-style reader; one index per real frame, never crossing a segment."""

    def __init__(self, manifest, data_config, norm_stats, method="echo",
                 cache_episodes=2, load_images=True):
        self.horizon = 50
        self.method = method
        if method not in {"echo", "baseline"}:
            raise ValueError("Expected echo or baseline")
        self.sources = read_json(data_config)["sources"]
        content = read_json(manifest)
        if content["schema_version"] != 1 or content["horizon"] != 50:
            raise ValueError("Unsupported training manifest")
        self.segments = [s for s in content["segments"]
                         if method == "echo" or s["segment_type"] != "failure"]
        self.offsets = [0]
        self.norm_stats = {}
        for field, width in (("actions", 7), ("state", 8)):
            values = {key: np.asarray(norm_stats[field][key], dtype=np.float32)[:width]
                      for key in ("q01", "q99")}
            if any(v.shape != (width,) or not np.isfinite(v).all() for v in values.values()):
                raise ValueError("Invalid checkpoint quantile statistics")
            if np.any(values["q99"] < values["q01"]):
                raise ValueError("Invalid quantile order")
            self.norm_stats[field] = values
        ids = set()
        for segment in self.segments:
            if segment["segment_id"] in ids or segment["source"] not in self.sources:
                raise ValueError("Duplicate segment or unknown source")
            ids.add(segment["segment_id"])
            start, end = segment["boundary"]["start"], segment["boundary"]["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end:
                raise ValueError("Invalid segment boundary")
            if not segment["source_lang"] or not segment["canonical_prompt"]:
                raise ValueError("Missing source/canonical instruction")
            if segment["segment_type"] == "success" and segment["source_lang"] != segment["canonical_prompt"]:
                raise ValueError("Success instruction must remain unchanged")
            self.offsets.append(self.offsets[-1] + end-start)
        if not self.segments or type(cache_episodes) is not int or cache_episodes <= 0:
            raise ValueError("Need nonempty data and a positive cache size")
        self.cache_episodes = cache_episodes
        self.load_images = load_images
        self._success = {}
        self._cache = OrderedDict()

    def __len__(self):
        return self.offsets[-1]

    def _load(self, segment):
        key = segment["segment_id"]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        storage = segment["storage"]
        root = Path(self.sources[segment["source"]]["local_dir"])
        if storage in {"success_parquet", "success_parquet_lerobot"}:
            source = segment["source"]
            if source not in self._success:
                self._success[source] = SuccessChunkDataset(
                    root, self.norm_stats, horizon=50, cache_episodes=self.cache_episodes,
                    episode_indices=sorted({s["episode_index"] for s in self.segments if s["source"] == source}),
                    column_layout="physical_intelligence" if storage == "success_parquet" else "lerobot_embedded")
            reader = self._success[source]
            table, states, actions, task_ids = reader._load_episode(segment["episode_index"])
            language = {reader.tasks[int(t)] for t in np.unique(task_ids)}
            images = (reader, table)
        elif storage == "lerobot_video":
            import pyarrow.parquet as pq
            info = read_json(root / "meta/info.json")
            episode = segment["episode_index"]
            params = dict(episode_chunk=episode//info["chunks_size"], episode_index=episode)
            table = pq.read_table(inside(root, info["data_path"].format(**params)))
            states = np.asarray(table["observation.state"].to_pylist(), np.float32)
            actions = np.asarray(table["action"].to_pylist(), np.float32)
            tasks = {t["task_index"]: t["task"] for t in
                     [__import__("json").loads(line) for line in (root/"meta/tasks.jsonl").read_text().splitlines() if line.strip()]}
            language = {tasks[int(t)] for t in np.unique(table["task_index"].to_numpy())}
            if not np.array_equal(table["frame_index"].to_numpy(), np.arange(len(actions))):
                raise ValueError("Recovery frame_index is not contiguous")
            if not np.array_equal(table["episode_index"].to_numpy(), np.full(len(actions), episode)):
                raise ValueError("Recovery episode_index differs from manifest")
            images = []
            if self.load_images:
                for view in ("observation.images.image", "observation.images.wrist_image"):
                    path = inside(root, info["video_path"].format(**params, video_key=view))
                    images.append(_VideoHandle(path, len(actions), prefer_pyav=True))
        elif storage == "rollout_npz":
            episode_dir = inside(root, segment["episode_path"])
            with np.load(episode_dir/"trajectory.npz", allow_pickle=False) as payload:
                if set(payload.files) != {"actions", "states"}:
                    raise ValueError("Published failure archive must contain only actions and states")
                actions, states = payload["actions"].copy(), payload["states"].copy()
            metadata = read_json(episode_dir/"episode.json")
            if (metadata["success"] is not False or metadata["length"] != len(actions)
                    or metadata["terminal_reason"] != segment["boundary"]["reason"]
                    or segment["boundary"]["end"] != len(actions)):
                raise ValueError("Failure payload and terminal boundary disagree")
            language = {metadata["source_lang"]}
            images = [_VideoHandle(episode_dir/name, len(actions))
                      for name in ("agentview.mp4", "wrist.mp4")] if self.load_images else []
        else:
            raise ValueError("Unsupported storage")
        length = len(actions)
        if states.shape != (length, 8) or actions.shape != (length, 7):
            raise ValueError("State/action shapes must be [N,8]/[N,7]")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Nonfinite states/actions")
        if segment["boundary"]["end"] > length or language != {segment["source_lang"]}:
            raise ValueError("Source boundary or instruction differs from frozen manifest")
        item = (states, actions, images)
        self._cache[key] = item
        while len(self._cache) > self.cache_episodes:
            _, (_, _, old_images) = self._cache.popitem(last=False)
            if isinstance(old_images, list):
                for handle in old_images:
                    handle.release()
        return item

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        position = bisect_right(self.offsets, index)-1
        segment = self.segments[position]
        local = index-self.offsets[position]
        start, end = segment["boundary"]["start"], segment["boundary"]["end"]
        frame, valid = start+local, min(50, end-start-local)
        states, actions, views = self._load(segment)
        raw_actions = np.zeros((50, 7), np.float32)
        raw_actions[:valid] = convert_actions(actions[frame:frame+valid], segment["source_gripper_convention"])
        if not self.load_images:
            images = [None, None]
        elif isinstance(views, tuple):
            reader, table = views
            images = [reader._decode_image(table[key][frame].as_py(), key) for key in ("image", "wrist_image")]
        else:
            images = [Image.fromarray(v.read(frame)[:, :, ::-1].copy()) for v in views]
        pad, ss, sm, fs, fm, am, ab, sb, fb = signal_targets(segment["segment_type"], end-start, local)
        example = dict(image=images, lang=segment["canonical_prompt"],
                       state=normalize_quantile(states[frame:frame+1], self.norm_stats["state"]),
                       action=normalize_quantile(raw_actions, self.norm_stats["actions"]),
                       segment_type=segment["segment_type"], episode_index=position)
        if self.method == "baseline":
            example["baseline_action_bound"] = ab
        else:
            example.update(echo_ss=ss, echo_fs=fs, echo_action_mask=am,
                           echo_ss_mask=sm, echo_fs_mask=fm,
                           action_is_pad=pad, echo_action_bound=ab,
                           echo_ss_bound=sb, echo_fs_bound=fb)
        return example
