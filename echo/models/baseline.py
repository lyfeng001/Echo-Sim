"""Keep native PI05 modules and flow; match Echo's recovery-tail error mask.

This patches only the instance's training forward. State-dict keys, 32D flow
noise, action attention, and prediction/inference remain native PI05.
"""
from __future__ import annotations

from types import MethodType

import numpy as np
import torch


def reduce_action_error(error, action_bound):
    """Zero out-of-episode errors and retain all B*H*32 denominator entries."""
    if error.ndim != 3 or action_bound.shape != error.shape[:2]:
        raise ValueError("Expected error [B,H,D] and action_bound [B,H]")
    if action_bound.dtype != torch.bool:
        raise ValueError("action_bound must be boolean")
    numerator = (error * action_bound[..., None]).sum()
    count = error.numel()
    loss = numerator / count
    return {
        "loss": loss,
        "action_loss": loss,
        "loss_numerator": numerator,
        "loss_count": torch.as_tensor(count, device=error.device),
    }


def install_recovery_loss(model, loss_padding_mode="match_echo_recovery"):
    if loss_padding_mode not in {"match_echo_recovery", "native_unmasked"}:
        raise ValueError("Unknown loss_padding_mode")
    original = model.forward

    def forward(self, examples=None, **kwargs):
        if not examples:
            raise ValueError("Baseline batch cannot be empty")
        for example in examples:
            if example.get("segment_type") not in {"success", "recovery"}:
                raise ValueError("Baseline accepts success and recovery only")
            if any(key.startswith("echo_") for key in example):
                raise ValueError("Baseline examples cannot contain Echo signal fields")
        return_debug = bool(kwargs.pop("return_debug", False))
        output = original(examples=examples, return_debug=True, **kwargs)
        error = (output["v_t"] - output["u_t"]).square()
        if error.shape != (len(examples), self.action_horizon, self.action_dim):
            raise ValueError("Native PI05 flow error shape changed")
        values = np.stack([
            np.asarray(example["baseline_action_bound"])
            for example in examples
        ])
        if values.dtype != np.bool_ or values.shape != error.shape[:2]:
            raise ValueError("Invalid baseline_action_bound")
        bound = torch.as_tensor(values, dtype=torch.bool, device=error.device)
        if loss_padding_mode == "native_unmasked":
            bound = torch.ones_like(bound)
        result = reduce_action_error(error, bound)
        if return_debug:
            # Native debug fields must not replace the masked scalar losses.
            result.update({
                key: value for key, value in output.items() if key not in result
            })
        return result

    model.forward = MethodType(forward, model)
    return model


def build_recovery_training_model(*args, loss_padding_mode="match_echo_recovery", **kwargs):
    from echo.models.initialization import build_training_model

    model, metadata, stats = build_training_model(*args, **kwargs)
    if model.action_dim != 32 or model.action_horizon != 50:
        raise ValueError("Recovery baseline requires native PI05 H50 D32")
    install_recovery_loss(model, loss_padding_mode)
    metadata.update(
        variant="success_recovery_finetuned_H50_v2",
        loss_padding_mode=loss_padding_mode,
        signal_supervision=False,
        native_flow_dimensions=32,
        loss_denominator="B*50*32, including zeroed recovery tail errors",
    )
    return model, metadata, stats
