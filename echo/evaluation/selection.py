"""Freeze recovery trials before evaluation, without selecting by outcomes."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from echo.io import atomic_json, inside, read_json, sha256

SUITES = ("spatial", "object", "goal", "10", "90")
SCENE_FIELDS = ("path", "bddl_sha256", "state_sha256", "suite")


def scene_index(root):
    """Index released BDDL/snapshot pairs in lexicographic path order."""
    root = Path(root)
    rows = []
    for suite in SUITES:
        paths = sorted((root/("selected_bddl_scenes_"+suite)).rglob("*.bddl"))
        if not paths:
            raise ValueError(f"No released scenes for suite {suite}")
        for path in paths:
            snapshot = path.with_name(path.stem+"_sim_state.npy")
            state = np.load(snapshot, allow_pickle=False)
            if state.ndim != 1 or state.dtype.kind not in "fiu" or not np.isfinite(state).all():
                raise ValueError("Invalid released snapshot")
            rows.append(dict(path=path.relative_to(root).as_posix(), suite=suite,
                             bddl_sha256=sha256(path), state_sha256=sha256(snapshot)))
    return rows


def select_scenes(rows, seed, per_suite):
    if type(seed) is not int or not 0 <= seed < 2**32-len(SUITES):
        raise ValueError("Selection seed must fit RandomState's seed range")
    if type(per_suite) is not int or per_suite <= 0:
        raise ValueError("scenes_per_suite must be positive")
    selected = []
    for index, suite in enumerate(SUITES):
        candidates = [row for row in rows if row["suite"] == suite]
        if len(candidates) < per_suite:
            raise ValueError(f"Not enough scenes in suite {suite}")
        order = np.random.RandomState(seed+index).permutation(len(candidates))
        selected.extend(candidates[int(i)] for i in sorted(order[:per_suite]))
    return selected


def make_jobs(scenes, seed, trials):
    if type(seed) is not int or seed < 0 or type(trials) is not int or trials <= 0:
        raise ValueError("Need a nonnegative episode seed and positive trial count")
    jobs = []
    seen = set()
    for row in scenes:
        if row["path"] in seen:
            raise ValueError("Duplicate scene")
        seen.add(row["path"])
        scene = {key: row[key] for key in SCENE_FIELDS}
        for gripper in ("close", "open"):
            for trial in range(trials):
                key = f'{seed}|{row["path"]}|{gripper}|{trial}'
                digest = hashlib.sha256(key.encode()).hexdigest()
                jobs.append(dict(**scene, gripper=gripper, trial=trial,
                                 seed=int(digest[:8], 16), id=digest[:20]))
    return jobs


def freeze(config_path):
    config = read_json(config_path)
    if config["mode"] == "reseed":
        original = read_json(config["reference_manifest"])
        scenes, seen = [], set()
        for job in original["jobs"]:
            if job["path"] not in seen:
                scenes.append({key: job[key] for key in SCENE_FIELDS})
                seen.add(job["path"])
        selection_seed = original["selection_seed"]
    elif config["mode"] == "new_scenes":
        scenes = select_scenes(scene_index(config["scene_root"]), config["selection_seed"], config["scenes_per_suite"])
        selection_seed = config["selection_seed"]
    else:
        raise ValueError("mode must be reseed or new_scenes")
    for scene in scenes:
        path = inside(config["scene_root"], scene["path"])
        if sha256(path) != scene["bddl_sha256"] or sha256(path.with_name(path.stem+"_sim_state.npy")) != scene["state_sha256"]:
            raise ValueError("Selected scene/snapshot differs from the frozen reference")
    manifest = dict(schema_version=1, episode_seed=config["episode_seed"], selection_seed=selection_seed,
                    jobs=make_jobs(scenes, config["episode_seed"], config["trials_per_gripper"]))
    target = Path(config["output"])
    if target.exists() and read_json(target) != manifest:
        raise FileExistsError("Refuse replacing a different frozen evaluation manifest")
    atomic_json(target, manifest)
    print(f'{len(scenes)} scenes | {len(manifest["jobs"])} trials | {sha256(target)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection.json")
    freeze(parser.parse_args().config)


if __name__ == "__main__":
    main()
