"""Validate full rollouts and summarize success at its first occurrence."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from echo.io import atomic_json, read_json, sha256
from echo.evaluation.signals import success_identity


def inspect_episode(directory, job, run_identity=None):
    row = read_json(Path(directory)/"result.json")
    if any(row.get(key) != job[key] or type(row.get(key)) is not type(job[key])
           for key in ("id", "path", "seed", "bddl_sha256", "state_sha256", "suite", "gripper", "trial")):
        raise ValueError("Episode identity differs from the evaluation manifest")
    if row.get("terminal_reason") != "max_steps" or type(row.get("steps")) is not int or row["steps"] <= 0:
        raise ValueError("Rollout did not reach its configured maximum steps")
    with np.load(Path(directory)/"trajectory.npz", allow_pickle=False) as payload:
        events = payload["success_after_action"]
        if events.shape != (row["steps"],) or payload["actions"].shape != (row["steps"], 7):
            raise ValueError("Incomplete action/event record")
        if (payload["states"].shape != (row["steps"], 8) or not np.isfinite(payload["actions"]).all()
                or not np.isfinite(payload["states"]).all()):
            raise ValueError("Invalid state/action record")
        expected = success_identity(events)
    if any(type(row.get(k)) is not type(v) or row.get(k) != v for k, v in expected.items()):
        raise ValueError("Success latch or first success differs from observed environment events")
    if run_identity is not None:
        if row.get("run_identity") != run_identity:
            raise ValueError("Episode checkpoint/protocol differs from the run identity")
        if row["steps"] != run_identity["protocol"]["max_steps"]:
            raise ValueError("Rollout length differs from the configured maximum")
    return row


def inspect_results(run_dir, manifest):
    run_dir = Path(run_dir)
    jobs = read_json(manifest)["jobs"]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate manifest episode IDs")
    identity = read_json(run_dir/"identity.json") if (run_dir/"identity.json").exists() else None
    if identity and identity["manifest_sha256"] != sha256(manifest):
        raise ValueError("Run manifest checksum differs")
    valid, invalid, missing = {}, {}, []
    for job in jobs:
        directory = run_dir/"episodes"/job["id"]
        if not (directory/"result.json").is_file():
            missing.append(job["id"])
            continue
        try:
            valid[job["id"]] = inspect_episode(directory, job, identity)
        except (ValueError, KeyError, OSError) as error:
            invalid[job["id"]] = str(error)
    checkpoint_hashes = {r.get("checkpoint_sha256") for r in valid.values()}
    if len(checkpoint_hashes) > 1:
        raise ValueError("Results mix different model checkpoints")
    successes = sum(r["success"] for r in valid.values())
    status = dict(expected=len(jobs), completed=len(valid), successes=successes,
                  success_rate=successes/len(valid) if valid else None,
                  complete=len(valid) == len(jobs) and not invalid, missing=missing, invalid=invalid,
                  terminal_successes=sum(r["terminal_success"] for r in valid.values()))
    return valid, status


def summary(run_dir, manifest, baseline=None):
    rows, report = inspect_results(run_dir, manifest)
    groups = {}
    for suite in sorted({r["suite"] for r in rows.values()}):
        group = [r for r in rows.values() if r["suite"] == suite]
        hits = sum(r["success"] for r in group)
        groups[suite] = dict(successes=hits, completed=len(group), success_rate=hits/len(group))
    report["suites"] = groups
    if baseline:
        base, _ = inspect_results(baseline, manifest)
        shared = sorted(set(rows) & set(base))
        hits = sum(base[i]["success"] for i in shared)
        report["matched_baseline"] = dict(completed=len(shared), successes=hits,
                                           success_rate=hits/len(shared) if shared else None,
                                           evaluated_model_successes=sum(rows[i]["success"] for i in shared))
    return report


def display(name, report):
    rate = f'{report["success_rate"]*100:.2f}%' if report["success_rate"] is not None else "-"
    print(f'{name} | {report["completed"]} / {report["expected"]} | {report["successes"]} | {rate} | '
          f'{len(report["invalid"])} invalid | {"final" if report["complete"] else "partial"}', flush=True)
    if "matched_baseline" in report:
        matched = report["matched_baseline"]
        rate = f'{matched["success_rate"]*100:.2f}%' if matched["success_rate"] is not None else "-"
        print(f'baseline on shared IDs | {matched["completed"]} | {matched["successes"]} | {rate}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["summary", "monitor"])
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--manifest", default="manifests/evaluation_seed7.json")
    parser.add_argument("--baseline")
    parser.add_argument("--watch", type=float, default=0, help="Seconds between checks; 0 checks once")
    args = parser.parse_args()
    if args.watch < 0 or not np.isfinite(args.watch):
        parser.error("--watch must be nonnegative and finite")
    try:
        while True:
            print("Run | Completed / Expected | Successes | Success rate | Invalid | Status", flush=True)
            for run in args.run:
                report = summary(run, args.manifest, args.baseline)
                display(Path(run).name, report)
                if args.command == "summary":
                    atomic_json(Path(run)/"summary.json", report)
            if not args.watch:
                return
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
