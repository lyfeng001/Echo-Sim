"""Strict FP32 PI05 initialization for the separate H50 finetuning run.

Use the same pinned StarVLA + OpenPI PYTHONPATH as pi_policy.py. This module
does not discover frameworks, change the inference adapter, or freeze modules.
The trainer owns autocast, distributed wrapping, and optimizer construction.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _phase(event, **details):
    print(json.dumps({"event": event, "at": datetime.now(timezone.utc).isoformat(),
                      "rank": int(os.environ.get("RANK", "0")), **details}, sort_keys=True), flush=True)


def _checkpoint_shared_layer(layer_forward, layer_idx, models, streams, cond,
                             attention_mask, position_ids):
    """One recomputation closure per pair; the layer index cannot late-bind.

    Both streams and the differentiable time condition are explicit checkpoint
    inputs. Nonreentrant checkpointing also works when image inputs have no
    requires_grad, and preserves the surrounding autocast context.
    """
    def run(prefix, suffix, condition, mask, positions):
        return tuple(layer_forward(layer_idx, models, [prefix, suffix],
                                   [None, condition], mask, positions))

    return activation_checkpoint(
        run, streams[0], streams[1], cond, attention_mask, position_ids,
        use_reentrant=False,
    )


def _install_shared_checkpointing(model, norm_fn):
    """Replace this instance's paired training path, preserving module names."""
    depth = int(model.vlm.config.text_config.num_hidden_layers)
    if len(model.vlm.language_model.layers) != depth or len(model.action_head.model.layers) != depth:
        raise ValueError("PI05 language and action experts must have matching depths")
    original = model.forward_with_action_expert

    def forward_with_action_expert(self, *, prefix_embs, suffix_embs,
                                   attention_mask, position_ids, adarms_cond):
        if not self.training or not torch.is_grad_enabled():
            return original(prefix_embs=prefix_embs, suffix_embs=suffix_embs,
                            attention_mask=attention_mask, position_ids=position_ids,
                            adarms_cond=adarms_cond)
        streams = [prefix_embs, suffix_embs]
        models = [self.vlm.language_model, self.action_head.model]
        for layer_idx in range(depth):
            streams = _checkpoint_shared_layer(
                self.forward_shared_gemma_layer, layer_idx, models, streams,
                adarms_cond, attention_mask, position_ids,
            )
        suffix_out, _ = norm_fn(models[1].norm, streams[1], adarms_cond)
        return suffix_out

    model.forward_with_action_expert = MethodType(forward_with_action_expert, model)
    return {"method": "paired_shared_gemma_layer", "layers": depth, "use_reentrant": False}


def _install_vision_checkpointing(model):
    """Checkpoint the complete image embedding, including its projector.

    The pinned SigLIP exposes a gradient-checkpoint flag, but vendored versions
    differ in whether the encoder consumes it. Wrapping embed_image guarantees
    recomputation without depending on that optional upstream implementation.
    """
    original = model.vlm.embed_image

    def embed_image(self, image):
        if self.training and torch.is_grad_enabled():
            return activation_checkpoint(original, image, use_reentrant=False)
        return original(image)

    model.vlm.embed_image = MethodType(embed_image, model.vlm)
    return {"method": "whole_image_embedding_and_projector", "use_reentrant": False}


def _require_complete_action_chunks(model):
    original = model._prepare_examples

    def prepare_examples(self, examples, include_actions):
        if include_actions:
            import numpy as np

            for index, example in enumerate(examples):
                actions = np.asarray(example["action"])
                if actions.ndim != 2 or actions.shape[0] != self.action_horizon:
                    raise ValueError(
                        f"Example {index} actions must already have H={self.action_horizon}; "
                        f"received {actions.shape}. Dataset owns temporal padding."
                    )
                if actions.shape[1] not in (7, self.action_dim) or not np.isfinite(actions).all():
                    raise ValueError(f"Invalid finite native/model actions in example {index}: {actions.shape}")
        return original(examples, include_actions=include_actions)

    model._prepare_examples = MethodType(prepare_examples, model)


def build_training_model(checkpoint, tokenizer, horizon=50,
                         gradient_checkpointing=True, device="cuda"):
    """Return ``(model, serializable_metadata, checkpoint_norm_stats)``.

    Safetensors load on CPU with strict key/shape validation before any device
    move. H10→H50 changes token count only; the pinned model has no learnable
    action-horizon position table. Floating parameters remain FP32.
    """
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from safetensors.torch import load

    from echo.models.pi05_utils import canonicalize_model_name, configure_torch, resolve_norm_stats_source
    from echo.models.precision import apply_pi05_adarms_input_cast
    from starVLA.model.framework.share_tools import read_mode_config
    from starVLA.model.framework.VLM4A.PI0 import _norm
    from starVLA.model.framework.VLM4A.PI05 import PI05

    if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) < 1:
        raise ValueError("horizon must be a positive integer")
    horizon = int(horizon)
    checkpoint_path = Path(checkpoint).resolve(strict=True)
    tokenizer_path = Path(tokenizer).resolve(strict=True)
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError("Training initialization requires a safetensors checkpoint")
    run_dir = checkpoint_path.parents[1]
    config_path = run_dir / "config.yaml"
    statistics_path = run_dir / "dataset_statistics.json"
    raw_config, _ = read_mode_config(checkpoint_path)
    cfg = OmegaConf.create(raw_config)
    if canonicalize_model_name(str(cfg.framework.name)) != "PI05":
        raise ValueError("Training requires a PI05 checkpoint")
    if OmegaConf.select(cfg, "framework.discrete_state_input") is not False:
        raise ValueError("This run requires checkpoint discrete_state_input=false")
    original_horizon = int(cfg.framework.action_horizon)
    for key, value in {
        "framework.action_horizon": horizon,
        "framework.action_model.action_horizon": horizon,
        "framework.action_model.future_action_window_size": horizon - 1,
        "framework.precision": "float32",
        "framework.tokenizer.model_path": str(tokenizer_path),
        "trainer.pretrained_checkpoint": None,
    }.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    configure_torch()
    norm_stats = resolve_norm_stats_source("PI05", checkpoint=checkpoint_path)
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        stored_dtypes = dict(Counter(str(handle.get_slice(key).get_dtype()) for key in handle.keys()))

    _phase("model_construct_start", horizon=horizon, device=str(device))
    phase_started = time.monotonic()
    model = PI05(cfg)
    _phase("model_construct_complete", seconds=time.monotonic() - phase_started)
    # Resolve the lazy tokenizer before expensive weight loading/device moves.
    model.tokenizer._load()
    # Sequential reads avoid tens of thousands of mmap page faults on VEPFS.
    # The bytes API preserves the stored dtype and strict state-dict semantics.
    _phase("weight_read_start", checkpoint=str(checkpoint_path))
    phase_started = time.monotonic()
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint_size = len(checkpoint_bytes)
    _phase("weight_read_complete", seconds=time.monotonic() - phase_started,
           bytes=checkpoint_size, checkpoint_sha256=checkpoint_sha256)
    weights = load(checkpoint_bytes)
    del checkpoint_bytes
    _phase("weight_strict_load_start")
    model.load_state_dict(weights, strict=True)
    del weights
    _phase("weight_strict_load_complete")
    if model.action_horizon != horizon or model.action_head.action_horizon != horizon:
        raise RuntimeError("Constructed PI05 does not honor the requested horizon")
    if model.discrete_state_input:
        raise RuntimeError("PI05 unexpectedly enabled discrete state input")
    if any(param.is_floating_point() and param.dtype != torch.float32 for param in model.parameters()):
        raise RuntimeError("Training parameters must stay FP32 before mixed-precision autocast")
    _require_complete_action_chunks(model)
    checkpointing = {"enabled": bool(gradient_checkpointing)}
    if gradient_checkpointing:
        checkpointing["shared_gemma"] = _install_shared_checkpointing(model, _norm)
        checkpointing["vision"] = _install_vision_checkpointing(model)
    precision_patch = apply_pi05_adarms_input_cast(model)
    _phase("device_load_start", device=str(device))
    phase_started = time.monotonic()
    model.to(torch.device(device)).train()
    _phase("device_loaded", device=str(device), seconds=time.monotonic() - phase_started)
    params = list(model.named_parameters())
    frozen = [name for name, param in params if not param.requires_grad]
    # The upstream adaRMS implementation disables its unused base norm weight.
    # Preserve and expose that architectural choice, but reject frozen backbones.
    if any(not (name.startswith("action_head.gemma_expert.model.") and
                (name.endswith("layernorm.weight") or name.endswith("model.norm.weight")))
           for name in frozen):
        raise RuntimeError(f"Unexpected frozen model parameters: {frozen}")
    metadata = {
        "model": "PI05", "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_size, "checkpoint_sha256": checkpoint_sha256,
        "config_path": str(config_path), "config_sha256": _sha256(config_path),
        "dataset_statistics_path": str(statistics_path),
        "dataset_statistics_sha256": _sha256(statistics_path),
        "tokenizer": str(tokenizer_path), "tokenizer_sha256": _sha256(tokenizer_path),
        "strict_load": True, "strict_weight_loading": True,
        "weight_io": "sequential-bytes-to-CPU",
        "initial_action_horizon": original_horizon, "action_horizon": horizon,
        "discrete_state_input": False, "normalization": "quantile",
        "parameter_precision": "float32", "autocast_owned_by_trainer": True,
        "parameter_dtypes": dict(Counter(str(param.dtype) for _, param in params)),
        "checkpoint_stored_tensor_dtypes": stored_dtypes,
        "parameter_count": sum(param.numel() for _, param in params),
        "trainable_parameter_count": sum(param.numel() for _, param in params if param.requires_grad),
        "architecturally_frozen_parameter_names": frozen,
        "known_unused_parameter_names": ["action_head.gemma_expert.lm_head.weight"],
        "checkpointing": checkpointing, "precision_patch": precision_patch,
        "device": str(torch.device(device)), "training": model.training,
        "effective_framework_config": OmegaConf.to_container(model.config.framework, resolve=True),
        "adapter_sha256": _sha256(Path(__file__)),
    }
    return model, metadata, norm_stats
