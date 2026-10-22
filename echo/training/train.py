"""Local PI05/Echo finetuning; torchrun supplies the distributed ranks."""
from __future__ import annotations

import argparse
from datetime import timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np

from echo.config import learning_rate, load_training
from echo.data.chunks import ChunkDataset
from echo.training.distributed import all_rank_call, rank_zero_call
from echo.io import atomic_json, read_json, sha256, source_hashes
from echo.data.sampling import EchoWeightedSampler
from echo.data.validation import verify_receipt


def collate(examples):
    return examples


def training_dataset(config):
    from echo.models.pi05_utils import load_dataset_statistics
    stats = load_dataset_statistics(Path(config["checkpoint"]).parents[1]/"dataset_statistics.json", "PI05")
    return ChunkDataset(config["manifest"], config["data_config"], stats, config["method"], config["cache_episodes"])


def preflight(config):
    checkpoint = Path(config["checkpoint"])
    data_config = read_json(config["data_config"])
    if sha256(config["tokenizer"]) != data_config["tokenizer"]["sha256"]:
        raise ValueError("Tokenizer differs from pinned public input")
    expected_model = data_config["weights"]["sha256"]
    if sha256(checkpoint) != expected_model:
        raise ValueError("Training initialization differs from the pinned public PI05; use --resume for continuation")
    statistics = checkpoint.parents[1]/"dataset_statistics.json"
    verify_receipt(config["validation_receipt"], config["manifest"], config["data_config"], statistics, config["method"])
    dataset = training_dataset(config)
    counts = {}
    selected = {}
    for i, segment in enumerate(dataset.segments):
        counts[segment["segment_type"]] = counts.get(segment["segment_type"], 0)+1
        selected.setdefault(segment["source"], i)
    # First/middle/tail exercise images, boundary padding and source instructions.
    for i in selected.values():
        start, stop = dataset.offsets[i:i+2]
        for index in sorted({start, (start+stop-1)//2, stop-1}):
            dataset[index]
    identity = dict(config=config, manifest_sha256=sha256(config["manifest"]),
                    data_config_sha256=sha256(config["data_config"]),
                    checkpoint_sha256=expected_model, tokenizer_sha256=sha256(config["tokenizer"]),
                    statistics_sha256=sha256(statistics), config_yaml_sha256=sha256(checkpoint.parents[1]/"config.yaml"),
                    data_validation_sha256=sha256(config["validation_receipt"]),
                    segment_counts=counts, frames=len(dataset),
                    sources=source_hashes())
    return dataset, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/echo_tau07.json")
    parser.add_argument("--resume", help="Complete local checkpoint directory")
    parser.add_argument("--stop-after", type=int, help="Absolute optimizer step; preserves full LR schedule")
    parser.add_argument("--preflight", action="store_true", help="CPU input/image checks only")
    args = parser.parse_args()
    config = load_training(args.config)
    rank, world_size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if args.preflight:
        dataset, identity = preflight(config)
        print(json.dumps(dict(event="preflight_passed", segment_counts=identity["segment_counts"], frames=len(dataset))))
        return
    if world_size != config["world_size"]:
        raise ValueError("torchrun world size differs from config")
    stop_after = args.stop_after if args.stop_after is not None else config["max_steps"]
    if not 0 < stop_after <= config["max_steps"]:
        raise ValueError("--stop-after must be in (0,max_steps]")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from echo.training import checkpoint as artifacts

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires CUDA with native BF16 support")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    # Large immutable input files are hashed once, rather than once per GPU.
    identity = rank_zero_call(lambda: preflight(config)[1], world_size)
    dataset = all_rank_call(lambda: training_dataset(config), world_size)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir/f"rank_{rank:02d}.lock").open("a")
    try:
        all_rank_call(lambda: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB), world_size)
        def initialize_run():
            if (run_dir/"identity.json").exists():
                if read_json(run_dir/"identity.json") != identity:
                    raise ValueError("Existing run has different code/config/input identity")
                if not args.resume:
                    raise ValueError("Existing run requires explicit --resume; choose a new run_dir for a new run")
            else:
                if args.resume:
                    raise ValueError("Resume requires an existing run identity")
                atomic_json(run_dir/"identity.json", identity)
            if args.resume:
                latest = read_json(run_dir/"latest.json")
                checkpoint = (run_dir/latest["checkpoint"]).resolve()
                if (Path(args.resume).resolve() != checkpoint or
                        checkpoint.parent != (run_dir/"checkpoints").resolve() or
                        sha256(checkpoint/"complete.json") != latest["complete_sha256"]):
                    raise ValueError("Resume must use this run's latest complete checkpoint")
        rank_zero_call(initialize_run, world_size)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(config["seed"]+rank)
        np.random.seed((config["seed"]+rank) % 2**32)
        torch.manual_seed(config["seed"]+rank)
        torch.cuda.manual_seed(config["seed"]+rank)
        device = f"cuda:{local_rank}"
        if config["method"] == "echo":
            from echo.models.model import build_echo_training_model
            model, _, _ = build_echo_training_model(
                config["checkpoint"], config["tokenizer"], device=device,
                gradient_checkpointing=config["gradient_checkpointing"],
                echo_config=dict(horizon=50, robot_action_dim=7, signal_token_mode="mode5",
                                 ss_t_clean=config["tau_clean"], fs_t_clean=config["tau_clean"],
                                 ss_loss_weight=config["ss_loss_weight"], fs_loss_weight=config["fs_loss_weight"],
                                 ss_no_padding_weight=config["ss_no_padding_weight"]))
        else:
            from echo.models.baseline import build_recovery_training_model
            model, _, _ = build_recovery_training_model(
                config["checkpoint"], config["tokenizer"], device=device,
                gradient_checkpointing=config["gradient_checkpointing"])
        parameters = [p for p in model.parameters() if p.requires_grad]
        kwargs = dict(lr=config["lr"], betas=tuple(config["betas"]), eps=config["eps"],
                      weight_decay=config["weight_decay"], foreach=False)
        if world_size > 1:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            optimizer = ZeroRedundancyOptimizer(parameters, optimizer_class=torch.optim.AdamW, **kwargs)
        else:
            optimizer = torch.optim.AdamW(parameters, **kwargs)
        ddp = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True,
                                      gradient_as_bucket_view=True) if world_size > 1 else model
        step, cursor, pending_rng, loader_rng = 0, dict(epoch=0, batch_offset=0), None, None
        if args.resume:
            step, cursor, pending_rng, loader_rng = all_rank_call(
                lambda: artifacts.resume(args.resume, model, optimizer, identity, rank, world_size), world_size)
        if step >= stop_after:
            raise ValueError("Resume step must precede --stop-after")
        sampler = EchoWeightedSampler(dataset, world_size, rank, config["seed"], config["samples_per_epoch"],
                                      config["sampling_weights"], config["sampling_block_size"],
                                      data_profile="three_type" if config["method"] == "echo" else "baseline")
        stop_requested = [False]
        def request_stop(signum, frame):
            stop_requested[0] = True
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        optimizer.zero_grad(set_to_none=True)
        while step < stop_after:
            sampler.set_epoch(cursor["epoch"], cursor["batch_offset"], config["batch_size"])
            generator = torch.Generator().manual_seed(config["seed"]+cursor["epoch"]*world_size+rank)
            iterator = iter(DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler,
                                       collate_fn=collate, num_workers=0, generator=generator))
            if loader_rng is not None:
                generator.set_state(loader_rng)
                loader_rng = None
            if pending_rng is not None:
                artifacts.restore_rng(pending_rng)
                pending_rng = None
            batches = math.ceil(sampler.full_num_samples/config["batch_size"])
            for examples in iterator:
                began = time.monotonic()
                lr = learning_rate(step, config)
                for owner in (optimizer, getattr(optimizer, "optim", optimizer)):
                    for group in owner.param_groups:
                        group["lr"] = lr
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = ddp(examples)
                    loss = output["loss"]
                finite = torch.isfinite(loss).to(torch.int32)
                if world_size > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("Nonfinite training loss on at least one rank")
                # Local masked mean followed by DDP rank averaging; do not
                # substitute a global count-normalized objective.
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                cursor["batch_offset"] += 1
                if cursor["batch_offset"] == batches:
                    cursor = dict(epoch=cursor["epoch"]+1, batch_offset=0)
                metrics = torch.tensor([float(output.get(k, 0)) for k in ("loss", "action_loss", "ss_loss", "fs_loss")], device=device)
                requested = torch.tensor(int(stop_requested[0]), device=device)
                if world_size > 1:
                    dist.all_reduce(metrics)
                    metrics /= world_size
                    dist.all_reduce(requested, op=dist.ReduceOp.MAX)
                stop = bool(requested.item()) or step >= stop_after
                def record_progress():
                    record = dict(step=step, max_steps=config["max_steps"], lr=lr, cursor=cursor,
                                  loss=metrics[0].item(), action_loss=metrics[1].item(), ss_loss=metrics[2].item(),
                                  fs_loss=metrics[3].item(), gradient_norm=float(gradient_norm),
                                  seconds=time.monotonic()-began, stage="training")
                    with (run_dir/"loss.jsonl").open("a") as handle:
                        handle.write(json.dumps(record)+"\n")
                    atomic_json(run_dir/"progress.json", record)
                    print(json.dumps(record), flush=True)
                rank_zero_call(record_progress, world_size)
                if step in config["save_steps"] or step % config["save_interval"] == 0 or stop:
                    artifacts.save(run_dir, model, optimizer, step, cursor, identity, rank, world_size, generator)
                if stop:
                    break
            if stop:
                break
        def finish():
            atomic_json(run_dir/"progress.json", dict(step=step, max_steps=config["max_steps"], cursor=cursor,
                                                      stage="completed" if step == config["max_steps"] else "stopped"))
        rank_zero_call(finish, world_size)
    finally:
        lock.close()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
