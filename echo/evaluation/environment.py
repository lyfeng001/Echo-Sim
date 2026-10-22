"""Snapshot-backed LIBERO construction with movable-only placement relaxation.

Normal robot, renderer, model, fixture, and observable initialization is intact.
Only movable-object boundary/collision rejection is disabled for the temporary
placement that the recorded snapshot overwrites. Static fixture placement stays
unchanged because flat snapshots omit model.body_pos and model.body_quat.
No BDDL/XML/source/snapshot files are changed. Construction must be serial in a
process during the brief class patch; separate worker processes are supported.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import types
from typing import Iterator

import numpy as np


def _relax_movable_samplers(inner) -> list[dict]:
    changes = []
    seen = set()

    def visit(sampler):
        if sampler is None or id(sampler) in seen:
            return
        seen.add(id(sampler))
        children = getattr(sampler, "samplers", None)
        if children is not None:
            for child in children.values():
                visit(child)
            return
        objects = getattr(sampler, "mujoco_objects", [])
        if not isinstance(objects, (tuple, list)):
            objects = [objects]
        names = [obj.name for obj in objects]
        if not names:
            return
        fixture_names = [name for name in names if name in inner.fixtures_dict]
        if fixture_names:
            if len(fixture_names) != len(names):
                raise RuntimeError(f"Mixed fixture/movable sampler cannot be relaxed safely: {names}")
            changes.append({"sampler": sampler.name, "objects": names, "fixture": True,
                            "modified": False,
                            "ensure_object_boundary_in_range": bool(sampler.ensure_object_boundary_in_range),
                            "ensure_valid_placement": bool(sampler.ensure_valid_placement)})
            return
        if any(name not in inner.objects_dict for name in names):
            raise RuntimeError(f"Unknown object in placement sampler: {names}")
        before = {}
        for flag in ("ensure_object_boundary_in_range", "ensure_valid_placement"):
            if not hasattr(sampler, flag):
                raise RuntimeError(f"Unsupported sampler {type(sampler).__name__}: missing {flag}")
            before[flag] = bool(getattr(sampler, flag))
            setattr(sampler, flag, False)
        changes.append({"sampler": sampler.name, "objects": names, "fixture": False,
                        "modified": True, "before": before,
                        "after": {flag: False for flag in before}})

    for attribute in ("placement_initializer", "conditional_placement_initializer",
                      "conditional_placement_on_objects_initializer"):
        visit(getattr(inner, attribute, None))
    inner._recovery_sampler_audit = changes
    return changes


@contextmanager
def _constructor_sampler_adapter() -> Iterator[object]:
    from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
    original = BDDLBaseDomain._setup_placement_initializer

    def setup(self, arena):
        original(self, arena)
        _relax_movable_samplers(self)

    BDDLBaseDomain._setup_placement_initializer = setup
    try:
        yield setup
    finally:
        BDDLBaseDomain._setup_placement_initializer = original


def model_schema(env) -> dict:
    model = env.env.sim.model
    joints = []
    for index in range(int(model.njnt)):
        joints.append({"id": index, "name": model.joint_id2name(index),
                       "type": int(model.jnt_type[index]),
                       "body_id": int(model.jnt_bodyid[index]),
                       "qpos_adr": int(model.jnt_qposadr[index]),
                       "dof_adr": int(model.jnt_dofadr[index])})
    return {"nq": int(model.nq), "nv": int(model.nv), "na": int(model.na),
            "bodies": [model.body_id2name(i) for i in range(int(model.nbody))],
            "joints": joints}


def fixture_poses(env) -> dict:
    inner = env.env
    model = inner.sim.model
    return {
        name: {"body_id": int(model.body_name2id(obj.root_body)),
               "body_pos": model.body_pos[model.body_name2id(obj.root_body)].tolist(),
               "body_quat": model.body_quat[model.body_name2id(obj.root_body)].tolist()}
        for name, obj in inner.fixtures_dict.items()
    }


def validate_snapshot_shape(env, snapshot: np.ndarray) -> None:
    state = np.asarray(snapshot)
    model = env.env.sim.model
    expected = 1 + int(model.nq) + int(model.nv)
    if state.ndim != 1 or state.dtype.kind not in "fiu" or not np.isfinite(state).all():
        raise ValueError(f"Invalid flat snapshot: shape={state.shape}, dtype={state.dtype}")
    if int(model.na) != 0 or state.size != expected:
        raise ValueError(f"Snapshot/model mismatch: length={state.size}, expected=1+nq+nv={expected}, na={model.na}")


def restore_snapshot(env, snapshot: np.ndarray):
    """Return obs/audit after checking flat roundtrip and fixture invariance."""
    validate_snapshot_shape(env, snapshot)
    state = np.asarray(snapshot, dtype=np.float64).copy()
    before_fixtures = fixture_poses(env)
    obs = env.set_init_state(state)
    actual = np.asarray(env.get_sim_state(), dtype=np.float64)
    max_error = float(np.max(np.abs(actual - state))) if state.size else 0.0
    if not np.allclose(actual, state, rtol=0.0, atol=1e-10):
        raise ValueError(f"Snapshot roundtrip changed time/qpos/qvel: maximum difference {max_error}")
    after_fixtures = fixture_poses(env)
    if before_fixtures != after_fixtures:
        raise ValueError("set_init_state unexpectedly changed static fixture model poses")
    schema = model_schema(env)
    audit = {
        "adapter": "relax throwaway movable placement constraints; preserve all fixture samplers",
        "samplers": getattr(env.env, "_recovery_sampler_audit", []),
        "fixture_poses": after_fixtures,
        "model_schema": schema,
        "model_schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
        "snapshot_length": int(state.size),
        "snapshot_roundtrip_max_abs_error": max_error,
        "pairing_limit": "shape and roundtrip do not prove original joint-name ordering; asset package supplies no source schema",
    }
    env.recovery_audit = audit
    return obs, audit


def create_snapshot_env(
    bddl_file: str | Path,
    snapshot: np.ndarray,
    *,
    seed: int = 7,
    camera_heights: int = 256,
    camera_widths: int = 256,
    render_gpu_device_id: int = -1,
    **kwargs,
):
    """Return (env, obs), with snapshot installed and env.recovery_audit.

    Later env.env.reset() still runs normal superclass initialization and
    original fixture sampling. The sampler adapter persists per instance across
    hard resets; restore_snapshot should follow each reset for per-trial audit.
    """
    from libero.libero.envs import OffScreenRenderEnv
    bddl_file = Path(bddl_file).resolve(strict=True)
    np.random.seed(int(seed) % (2**32))
    with _constructor_sampler_adapter() as adapted_setup:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=int(camera_heights),
            camera_widths=int(camera_widths),
            render_gpu_device_id=int(render_gpu_device_id),
            **kwargs,
        )
    env.env._setup_placement_initializer = types.MethodType(adapted_setup, env.env)
    try:
        obs, _ = restore_snapshot(env, snapshot)
    except BaseException:
        env.close()
        raise
    return env, obs
