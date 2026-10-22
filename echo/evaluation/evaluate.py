"""Local LIBERO-Recovery evaluation with authentic continuation after success."""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np

from echo.io import atomic_json, inside, read_json, sha256, source_hashes
from echo.evaluation.signals import success_identity


def configure_libero():
    """Create the upstream local paths before LIBERO's interactive import."""
    import yaml
    root = Path(__file__).resolve().parents[2]/"LIBERO/libero/libero"
    config_root = Path(os.environ.get("LIBERO_CONFIG_PATH", "data/libero_config"))
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root.resolve())
    # This is local runtime configuration, excluded from the release Git tree.
    config = dict(benchmark_root=str(root), bddl_files=str(root/"bddl_files"),
                  init_states=str(root/"init_files"), assets=str(root/"assets"), datasets=str(Path("data").resolve()))
    target = config_root/"config.yaml"
    if not target.exists():
        target.write_text(yaml.safe_dump(config))


def state_vector(observation):
    from echo.models.pi05_utils import quat2axisangle
    return np.concatenate([observation["robot0_eef_pos"], quat2axisangle(observation["robot0_eef_quat"]),
                           observation["robot0_gripper_qpos"]]).astype(np.float32)


def prepare_environment(scene, job, protocol):
    from echo.evaluation.environment import create_snapshot_env, restore_snapshot
    from robosuite.utils.errors import RandomizationError

    snapshot_path = scene.with_name(scene.stem+"_sim_state.npy")
    if sha256(scene) != job["bddl_sha256"] or sha256(snapshot_path) != job["state_sha256"]:
        raise ValueError("Scene/snapshot differ from the frozen evaluation manifest")
    snapshot = np.load(snapshot_path, allow_pickle=False)
    env = None
    for retry in range(12):
        try:
            env, _ = create_snapshot_env(scene, snapshot, seed=(job["seed"]+retry) % 2**32,
                                         camera_heights=256, camera_widths=256, control_freq=protocol["control_hz"],
                                         render_gpu_device_id=protocol["render_gpu_id"])
            break
        except RandomizationError:
            if retry == 11:
                raise
    try:
        restore_snapshot(env, snapshot.copy())
        sim = env.env.sim
        rng = np.random.RandomState(job["seed"])
        original = sim.data.qpos.copy()
        intended = np.zeros_like(original)
        for body_id in env.env.obj_body_id.values():
            joint = int(sim.model.body_jntadr[body_id])
            if joint < 0 or int(sim.model.jnt_type[joint]) != 0:
                continue
            address = int(sim.model.jnt_qposadr[joint])
            delta = rng.uniform(-protocol["noise_xy"], protocol["noise_xy"], size=2)
            sim.data.qpos[address:address+2] += delta
            intended[address:address+2] += delta
        if not np.allclose(sim.data.qpos-original, intended, rtol=0, atol=1e-12):
            raise ValueError("Perturbation changed a non-XY coordinate")
        observation = env.set_init_state(sim.get_state().flatten().copy())
        if not np.allclose(sim.data.qpos, original+intended, rtol=0, atol=1e-12):
            raise ValueError("Snapshot restore changed intended perturbation")
        for robot in env.robots:
            robot.controller.update(force=True)
            robot.controller.update_initial_joints(robot.controller.joint_pos)
            robot.controller.reset_goal()
        settle_action = [0.0]*6 + [1.0 if job["gripper"] == "close" else -1.0]
        for _ in range(protocol["settle_steps"]):
            observation, _, _, _ = env.step(settle_action)
        if env.env.horizon <= env.env.timestep+protocol["max_steps"]:
            raise ValueError("Environment internal horizon cannot cover this rollout")
        return env, observation, bool(env.check_success())
    except BaseException:
        env.close()
        raise


def rollout(env, observation, policy, instruction, protocol):
    pending = deque()
    states, actions, events, images, wrists = [], [], [], [], []
    queries, ss, fs = [], [], []
    for step in range(protocol["max_steps"]):
        images.append(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]))
        wrists.append(np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1]))
        states.append(state_vector(observation))
        if not pending:
            chunk = policy.infer(observation, instruction)
            if np.shape(chunk) != (50, 7) or not np.isfinite(chunk).all():
                raise ValueError("Invalid action chunk")
            queries.append(step)
            if policy.last_signals is not None:
                ss.append(policy.last_signals["ss"].copy())
                fs.append(policy.last_signals["fs"].copy())
            pending.extend(chunk[:protocol["execute_horizon"]])
        action = pending.popleft()
        actions.append(action.copy())
        observation, _, done, _ = env.step(action.tolist())
        # LIBERO returns task success, not a time-limit flag. Keep executing
        # the same policy/actions after success, with no reset or substitute.
        events.append(bool(done))
    events = np.asarray(events, dtype=bool)
    return dict(terminal_image=np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]),
                states=np.asarray(states), actions=np.asarray(actions), success_after_action=events,
                images=images, wrists=wrists, query_controller_step=np.asarray(queries, np.int64),
                ss=np.asarray(ss, np.float32), fs=np.asarray(fs, np.float32), **success_identity(events))


def canonical_instruction(scene, registry):
    match = re.search(r"\(:language\s+(.+?)\s*\)", scene.read_text())
    if not match:
        raise ValueError("Scene has no language instruction")
    source = match.group(1).strip()
    matches = {row["canonical_prompt"] for row in registry["tasks"] if source in
               (row["source_bddl_language"], row["canonical_prompt"])}
    if len(matches) != 1:
        raise ValueError("Scene language must resolve to exactly one canonical instruction")
    return matches.pop()


def initialize_run(run_dir, identity, num_shards):
    """Freeze both scientific inputs and the disjoint writer layout."""
    with (Path(run_dir)/"identity.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        layout = {"num_shards": num_shards}
        for name, expected in (("identity.json", identity), ("sharding.json", layout)):
            path = Path(run_dir)/name
            if path.exists() and read_json(path) != expected:
                raise ValueError(f"Existing evaluation has different {name}; choose a new run_dir")
        atomic_json(Path(run_dir)/"identity.json", identity)
        atomic_json(Path(run_dir)/"sharding.json", layout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.json")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Stop after this many assigned trials, for smoke checks")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards or args.limit is not None and args.limit <= 0:
        parser.error("Invalid shard or limit")
    config = read_json(args.config)
    for key in ("max_steps", "execute_horizon", "settle_steps", "control_hz", "video_fps", "num_inference_steps"):
        if type(config[key]) is not int or config[key] < (0 if key == "settle_steps" else 1):
            raise ValueError(f"Invalid {key}")
    if config["execute_horizon"] > 50 or not np.isfinite(config["noise_xy"]) or config["noise_xy"] < 0:
        raise ValueError("Invalid execute horizon or perturbation")
    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_libero()
    from echo.models.policy import Policy
    from echo.evaluation.results import inspect_episode, summary
    import imageio.v2 as imageio
    import torch

    protocol = {key: config[key] for key in ("max_steps", "execute_horizon", "settle_steps", "noise_xy", "control_hz")}
    protocol["render_gpu_id"] = config["render_gpu_id"]
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir/f"shard_{args.shard_index:03d}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        policy = Policy(config["checkpoint"], config["tokenizer"], config["device"], config["num_inference_steps"])
        # The renderer's device is operational, not part of the scientific
        # protocol. Different shards can use different visible GPU indices.
        identity = dict(model=policy.metadata, manifest_sha256=sha256(config["manifest"]),
                        prompt_sha256=sha256(config["prompt_registry"]),
                        protocol={k: v for k, v in protocol.items() if k != "render_gpu_id"},
                        video_fps=config["video_fps"], code=source_hashes())
        initialize_run(run_dir, identity, args.num_shards)
        jobs = read_json(config["manifest"])["jobs"]
        if len({j["id"] for j in jobs}) != len(jobs):
            raise ValueError("Duplicate evaluation IDs")
        assigned = jobs[args.shard_index::args.num_shards]
        if args.limit is not None:
            assigned = assigned[:args.limit]
        registry = read_json(config["prompt_registry"])
        for job in assigned:
            target = inside(run_dir/"episodes", job["id"])
            if (target/"result.json").is_file():
                inspect_episode(target, job, identity)
                continue
            if target.exists():
                raise FileExistsError("Incomplete committed episode requires review")
            env = None
            began = time.monotonic()
            try:
                scene = inside(config["scene_root"], job["path"])
                instruction = canonical_instruction(scene, registry)
                env, observation, initial_success = prepare_environment(scene, job, protocol)
                torch.manual_seed(job["seed"])
                torch.cuda.manual_seed_all(job["seed"])
                data = rollout(env, observation, policy, instruction, protocol)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=".pending-", dir=target.parent) as temporary:
                    stage = Path(temporary)
                    np.savez_compressed(stage/"trajectory.npz", actions=data["actions"], states=data["states"],
                                        success_after_action=data["success_after_action"])
                    if policy.method == "echo":
                        np.savez_compressed(stage/"signals.npz", query_controller_step=data["query_controller_step"],
                                            ss=data["ss"], fs=data["fs"])
                    for name, frames in (("agentview", data["images"]), ("wrist", data["wrists"])):
                        imageio.mimwrite(stage/(name+".mp4"), frames, fps=config["video_fps"], codec="libx264",
                                        quality=7, macro_block_size=1)
                    imageio.imwrite(stage/"terminal.png", data["terminal_image"])
                    record = dict(**job, success=data["success"], first_success_step=data["first_success_step"],
                                  terminal_success=data["terminal_success"], initial_success_after_settle=initial_success,
                                  terminal_reason="max_steps", steps=protocol["max_steps"], instruction=instruction,
                                  wall_seconds=time.monotonic()-began, run_identity=identity,
                                  checkpoint_sha256=policy.metadata["checkpoint_sha256"])
                    atomic_json(stage/"result.json", record)
                    inspect_episode(stage, job, identity)
                    os.rename(stage, target)
                print(f'{job["id"]} | success={record["success"]} | first={record["first_success_step"]} | {record["steps"]} steps', flush=True)
                atomic_json(run_dir/f"progress_shard_{args.shard_index:03d}.json", dict(last_episode=job["id"], stage="evaluating"))
            except Exception as error:
                atomic_json(run_dir/"errors"/(job["id"]+".json"), dict(id=job["id"], error_type=type(error).__name__, error=str(error)))
                raise
            finally:
                if env is not None:
                    env.close()
        atomic_json(run_dir/"summary.json", summary(run_dir, config["manifest"]))
        atomic_json(run_dir/f"progress_shard_{args.shard_index:03d}.json", dict(stage="completed_assigned_trials"))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
