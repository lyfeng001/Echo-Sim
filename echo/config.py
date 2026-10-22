"""Explicit JSON configuration; run commands from the repository root."""
from __future__ import annotations

import math
from echo.io import read_json


def load_training(path):
    config = read_json(path)
    required = {
        "method", "tau_clean", "checkpoint", "tokenizer", "data_config", "manifest", "validation_receipt",
        "run_dir", "seed", "world_size", "batch_size", "max_steps", "warmup_steps",
        "lr", "min_lr", "lr_schedule", "weight_decay", "betas", "eps", "grad_clip",
        "samples_per_epoch", "sampling_block_size", "sampling_weights", "num_workers",
        "cache_episodes", "save_steps", "save_interval", "gradient_checkpointing",
        "ss_loss_weight", "fs_loss_weight", "ss_no_padding_weight",
    }
    if set(config) != required:
        raise ValueError(f"Training fields differ: missing={required-set(config)}, unknown={set(config)-required}")
    if config["method"] not in {"echo", "baseline"}:
        raise ValueError("method must be echo or baseline")
    for key in ("seed", "warmup_steps", "num_workers"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("world_size", "batch_size", "max_steps", "samples_per_epoch",
                "sampling_block_size", "cache_episodes", "save_interval"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    # A deterministic reader and exact continuation require no worker prefetch.
    if config["num_workers"] != 0:
        raise ValueError("Exact resume currently requires num_workers=0")
    if type(config["gradient_checkpointing"]) is not bool:
        raise ValueError("gradient_checkpointing must be boolean")
    for key in ("lr", "min_lr", "eps", "grad_clip"):
        if isinstance(config[key], bool) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("weight_decay", "ss_loss_weight", "fs_loss_weight", "ss_no_padding_weight"):
        if isinstance(config[key], bool) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= config["warmup_steps"] < config["max_steps"] or config["min_lr"] > config["lr"]:
        raise ValueError("Invalid warmup or minimum LR")
    if config["lr_schedule"] not in {"cosine", "constant"}:
        raise ValueError("lr_schedule must be cosine or constant")
    if config["lr_schedule"] == "constant" and config["min_lr"] != config["lr"]:
        raise ValueError("constant schedule requires min_lr=lr")
    tau = config["tau_clean"]
    if isinstance(tau, bool) or not math.isfinite(tau) or not 0 <= tau < 1:
        raise ValueError("tau_clean must be in [0,1)")
    if config["method"] == "baseline" and tau != 0:
        raise ValueError("Native PI05 baseline has no signal forcing; use tau_clean=0")
    kinds = {"success", "recovery"} | ({"failure"} if config["method"] == "echo" else set())
    if set(config["sampling_weights"]) != kinds:
        raise ValueError("Echo uses three types; baseline uses success and recovery only")
    if any(isinstance(w, bool) or not math.isfinite(w) or w <= 0 for w in config["sampling_weights"].values()):
        raise ValueError("Sampling weights must be finite and positive")
    if (len(config["betas"]) != 2 or
            any(isinstance(b, bool) or not math.isfinite(b) or not 0 <= b < 1 for b in config["betas"])):
        raise ValueError("Invalid AdamW betas")
    if any(type(s) is not int or not 0 < s <= config["max_steps"] for s in config["save_steps"]):
        raise ValueError("save_steps must contain optimizer steps in (0,max_steps]")
    return config


def learning_rate(completed, config):
    """LR for the next update; matches the historical one-based warmup."""
    warmup = config["warmup_steps"]
    if warmup and completed < warmup:
        return config["lr"] * (completed + 1) / warmup
    if config["lr_schedule"] == "constant":
        return config["lr"]
    progress = min(1.0, max(0.0, (completed-warmup)/(config["max_steps"]-warmup)))
    return config["min_lr"] + (config["lr"]-config["min_lr"]) * (1+math.cos(math.pi*progress))/2
