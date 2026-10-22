"""Complete local checkpoints with strict identity and per-rank resume state."""
from __future__ import annotations

from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch

from echo.io import atomic_json, read_json, sha256
from echo.training.distributed import all_rank_call, rank_zero_call


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def optimizer_partition(optimizer, model):
    names = {id(p): n for n, p in model.named_parameters()}
    local = optimizer.optim if hasattr(optimizer, "optim") else optimizer
    return [[names[id(p)] for p in group["params"]] for group in local.param_groups]


def verify(directory):
    directory = Path(directory)
    record = read_json(directory/"complete.json")
    if record.get("format_version") != 1:
        raise ValueError("Unknown checkpoint format")
    training = read_json(directory/"training.json")
    world_size = training.get("world_size")
    if (type(world_size) is not int or world_size <= 0 or
            type(record.get("step")) is not int or record["step"] <= 0 or
            training.get("step") != record["step"] or
            record.get("method") != training["identity"]["config"]["method"]):
        raise ValueError("Checkpoint step, method or world size differs")
    expected_files = {"fp32/model.safetensors", "config.yaml", "dataset_statistics.json", "training.json"}
    expected_files.update(f"rank_{rank:02d}.pt" for rank in range(world_size))
    if set(record["files"]) != expected_files:
        raise ValueError("Checkpoint inventory is missing weights or rank state")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != expected_files | {"complete.json"} or any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Unexpected checkpoint files or symlinks")
    cursor = training["cursor"]
    if set(cursor) != {"epoch", "batch_offset"} or any(type(v) is not int or v < 0 for v in cursor.values()):
        raise ValueError("Invalid checkpoint sampler cursor")
    for name, expected in record["files"].items():
        from echo.io import inside
        path = inside(directory, name)
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Incomplete or changed checkpoint object: {name}")
    return record


def save(run_dir, model, optimizer, step, cursor, identity, rank, world_size, loader_generator):
    from omegaconf import OmegaConf
    from safetensors.torch import save_file

    target = Path(run_dir)/"checkpoints"/f"step_{step:08d}"
    def prepare():
        if target.exists():
            raise FileExistsError("Refuse overwriting an existing checkpoint")
        stage = target.with_name(".pending-"+target.name+"-"+uuid.uuid4().hex)
        (stage/"fp32").mkdir(parents=True)
        return str(stage)
    stage = Path(rank_zero_call(prepare, world_size))
    def save_rank():
        local = optimizer.optim if hasattr(optimizer, "optim") else optimizer
        state = dict(step=step, cursor=dict(cursor), world_size=world_size, rank=rank,
                     identity=identity, optimizer=cpu_tree(local.state_dict()),
                     partition=optimizer_partition(optimizer, model), rng=capture_rng(),
                     loader_rng=loader_generator.get_state())
        torch.save(state, stage/f"rank_{rank:02d}.pt")
    all_rank_call(save_rank, world_size)
    def commit():
        weights = {name: tensor.detach().to("cpu", copy=True).contiguous() for name, tensor in model.state_dict().items()}
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in weights.values()):
            raise ValueError("Training checkpoint must preserve FP32 master parameters")
        save_file(weights, str(stage/"fp32/model.safetensors"))
        del weights
        config = OmegaConf.create(OmegaConf.to_container(model.config, resolve=True))
        # Portable artifact paths; inference supplies its own local tokenizer.
        OmegaConf.update(config, "framework.tokenizer.model_path", "data/paligemma_tokenizer.model", force_add=True)
        (stage/"config.yaml").write_text(OmegaConf.to_yaml(config))
        statistics = Path(identity["config"]["checkpoint"]).parents[1]/"dataset_statistics.json"
        shutil.copyfile(statistics, stage/"dataset_statistics.json")
        atomic_json(stage/"training.json", dict(step=step, cursor=cursor, world_size=world_size, identity=identity))
        files = ["fp32/model.safetensors", "config.yaml", "dataset_statistics.json", "training.json"]
        files += [f"rank_{r:02d}.pt" for r in range(world_size)]
        records = {name: dict(bytes=(stage/name).stat().st_size, sha256=sha256(stage/name)) for name in files}
        for name in files:
            import os
            with (stage/name).open("rb") as handle:
                os.fsync(handle.fileno())
        atomic_json(stage/"complete.json", dict(format_version=1, step=step, method=identity["config"]["method"], files=records))
        stage.rename(target)
        atomic_json(Path(run_dir)/"latest.json", dict(step=step, checkpoint=str(target.relative_to(run_dir)),
                                                    complete_sha256=sha256(target/"complete.json")))
    rank_zero_call(commit, world_size)
    return target


def resume(directory, model, optimizer, identity, rank, world_size):
    from safetensors.torch import load
    complete = verify(directory)
    state = read_json(Path(directory)/"training.json")
    if state["identity"] != identity or state["world_size"] != world_size:
        raise ValueError("Resume config, sources or world size differ")
    payload = (Path(directory)/"fp32/model.safetensors").read_bytes()
    weights = load(payload)
    del payload
    model.load_state_dict(weights, strict=True)
    del weights
    # Only deserialize a local, checksum-verified artifact produced by this code.
    local_state = torch.load(Path(directory)/f"rank_{rank:02d}.pt", map_location="cpu", weights_only=False)
    if (local_state["identity"] != identity or local_state["rank"] != rank
            or local_state["world_size"] != world_size or local_state["step"] != complete["step"]
            or local_state["cursor"] != state["cursor"]
            or local_state["partition"] != optimizer_partition(optimizer, model)):
        raise ValueError("Rank optimizer partition or cursor differs")
    local = optimizer.optim if hasattr(optimizer, "optim") else optimizer
    local.load_state_dict(local_state["optimizer"])
    return complete["step"], state["cursor"], local_state["rng"], local_state["loader_rng"]
