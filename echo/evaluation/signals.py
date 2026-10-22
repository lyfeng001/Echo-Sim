"""Frame-level task status signal accuracy from whole-chunk SS/FS means."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from echo.io import atomic_json, read_json


@dataclass(frozen=True)
class SignalMetricConfig:
    ss_threshold: float = 0.5
    fs_threshold: float = 0.5
    success_window_seconds: float = 3.0
    control_hz: int = 20

    def __post_init__(self):
        if (not math.isfinite(self.ss_threshold) or not math.isfinite(self.fs_threshold)
                or not math.isfinite(self.success_window_seconds) or self.success_window_seconds < 0
                or type(self.control_hz) is not int or self.control_hz <= 0):
            raise ValueError("Invalid signal thresholds, window or control frequency")


def load_metric_config(path):
    values = read_json(path)
    return SignalMetricConfig(**{key: values[key] for key in
                                 ("ss_threshold", "fs_threshold", "success_window_seconds", "control_hz")})


def success_identity(events):
    events = np.asarray(events)
    if events.ndim != 1 or events.dtype != np.bool_ or not len(events):
        raise ValueError("Need a nonempty boolean success_after_action vector")
    hits = np.flatnonzero(events)
    first = int(hits[0]+1) if len(hits) else None
    return dict(success=bool(len(hits)), first_success_step=first, terminal_success=bool(events[-1]))


def whole_chunk(signals_path, steps):
    """Return actual query steps and H50 means, with no clipping or smoothing."""
    with np.load(signals_path, allow_pickle=False) as payload:
        queries = payload["query_controller_step"].copy()
        ss, fs = payload["ss"].copy(), payload["fs"].copy()
    if (queries.ndim != 1 or queries.dtype.kind not in "iu" or not len(queries)
            or queries[0] != 0 or np.any(np.diff(queries) <= 0) or queries[-1] >= steps):
        raise ValueError("Invalid signal query timestamps")
    if (ss.shape != (len(queries), 50) or fs.shape != ss.shape
            or not np.isfinite(ss).all() or not np.isfinite(fs).all()):
        raise ValueError("Signals must be finite [queries,H50]")
    return queries, ss.mean(axis=1, dtype=np.float64), fs.mean(axis=1, dtype=np.float64)


def signal_accuracy(events, queries, ss, fs, config=SignalMetricConfig()):
    """Score observation frames, holding each query until the next query.

    Query t observes the state after t actions. success_after_action[t-1]
    describes action t, so first_success_step is a one-based action count.
    The successful window includes both endpoints. Only available frames are
    scored; no observation is synthesized after the final action.
    """
    identity = success_identity(events)
    steps = len(events)
    queries = np.asarray(queries)
    ss, fs = np.asarray(ss), np.asarray(fs)
    if (queries.ndim != 1 or queries.dtype.kind not in "iu" or not len(queries)
            or queries[0] != 0 or queries[-1] >= steps or np.any(np.diff(queries) <= 0)
            or ss.shape != queries.shape or fs.shape != queries.shape
            or not np.isfinite(ss).all() or not np.isfinite(fs).all()):
        raise ValueError("Invalid signal summaries/timestamps")
    ongoing = ss < config.ss_threshold
    success = (ss > config.ss_threshold) & (fs < config.fs_threshold)
    first = identity["first_success_step"]
    window_steps = config.success_window_seconds * config.control_hz
    # Score all available pre-action observation frames, including held values.
    frame_steps = np.arange(steps)
    index = np.searchsorted(queries, frame_steps, side="right")-1
    if first is None:
        included = np.ones(steps, bool)
        frame_correct = ongoing[index]
        before = np.ones(steps, bool)
        after = np.zeros(steps, bool)
    else:
        before = frame_steps < first
        after = (frame_steps >= first) & (frame_steps <= first+window_steps)
        included = before | after
        frame_correct = np.where(before, ongoing[index], success[index])
    return dict(**identity,
                missing_post_success_query=bool(first is not None and not np.any(
                    (queries >= first) & (queries <= first+window_steps))),
                window_truncated=bool(first is not None and first+window_steps > steps-1),
                frame_correct=int((frame_correct & included).sum()), frame_total=int(included.sum()),
                ongoing_frame_correct=int((frame_correct & before).sum()), ongoing_frame_total=int(before.sum()),
                success_frame_correct=int((frame_correct & after).sum()), success_frame_total=int(after.sum()),
                threshold_equal_queries=int((ss == config.ss_threshold).sum()))


def summarize_signals(run_dir, manifest, config):
    from echo.evaluation.results import inspect_results
    identity_path = Path(run_dir)/"identity.json"
    if identity_path.exists() and read_json(identity_path)["protocol"]["control_hz"] != config.control_hz:
        raise ValueError("Signal control_hz differs from the evaluation's physical control frequency")
    valid, status = inspect_results(run_dir, manifest)
    rows, invalid = [], {}
    for episode_id, result in valid.items():
        directory = Path(run_dir)/"episodes"/episode_id
        try:
            with np.load(directory/"trajectory.npz", allow_pickle=False) as payload:
                events = payload["success_after_action"].copy()
            queries, ss, fs = whole_chunk(directory/"signals.npz", len(events))
            rows.append(dict(id=episode_id, **signal_accuracy(events, queries, ss, fs, config)))
        except (ValueError, KeyError, OSError) as error:
            invalid[episode_id] = str(error)
    total = len(rows)
    frame_total = sum(r["frame_total"] for r in rows)
    frame_correct = sum(r["frame_correct"] for r in rows)
    report = dict(config=config.__dict__, evaluation=status, invalid_signals=invalid,
                  frame=dict(correct=frame_correct, total=frame_total,
                             accuracy=frame_correct/frame_total if frame_total else None),
                  missing_post_success_query=sum(r["missing_post_success_query"] for r in rows),
                  truncated_success_windows=sum(r["window_truncated"] for r in rows), episodes=rows)
    report["complete"] = status["complete"] and total == status["expected"] and not invalid
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--manifest", default="manifests/evaluation_seed7.json")
    parser.add_argument("--config", default="configs/signals.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = summarize_signals(args.run, args.manifest, load_metric_config(args.config))
    atomic_json(args.output or Path(args.run)/"signal_accuracy.json", report)
    row = report["frame"]
    rate = f'{100*row["accuracy"]:.2f}%' if row["accuracy"] is not None else "-"
    print(f'frame | {row["correct"]} / {row["total"]} | {rate} | complete={report["complete"]}')


if __name__ == "__main__":
    main()
