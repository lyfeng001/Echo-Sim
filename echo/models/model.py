"""Echo mode5 on the pinned LIBERO PI05, without changing baseline sources.

The reference's ``*_use_separate_proj`` means independent input/output linear
projections for SS and FS. Transformer FFNs and the time MLP remain shared.
Signals have H tokens each, giving suffix order [A(H), SS(H), FS(H)].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EchoModelConfig:
    horizon: int = 50
    robot_action_dim: int = 7
    signal_token_mode: str = "mode5"
    ss_t_clean: float = 0.5
    fs_t_clean: float = 0.5
    ss_loss_weight: float = 1.0
    fs_loss_weight: float = 1.0
    ss_no_padding_weight: float = 0.1

    def __post_init__(self):
        if self.horizon != 50 or self.robot_action_dim != 7:
            raise ValueError("Echo LIBERO requires H=50 and native 7D actions")
        if self.signal_token_mode != "mode5":
            raise ValueError("This implementation supports the requested mode5 only")
        for name in ("ss_t_clean", "fs_t_clean"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("ss_loss_weight", "fs_loss_weight", "ss_no_padding_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def signal_time(action_time: torch.Tensor, clean_time: float) -> torch.Tensor:
    """Reference forcing: a signal is clean when action time reaches t_clean."""
    if not 0 <= clean_time < 1:
        raise ValueError("clean_time must be in [0, 1)")
    return ((action_time - clean_time) / (1.0 - clean_time)).clamp(0.0, 1.0)


def make_echo_attention_mask(prefix_mask: torch.Tensor, horizon: int) -> torch.Tensor:
    """Boolean mask with query rows, key columns, and physical prefix padding."""
    if prefix_mask.ndim != 2 or prefix_mask.dtype != torch.bool or horizon < 1:
        raise ValueError("Expected bool [B, prefix_length] and a positive horizon")
    batch, prefix = prefix_mask.shape
    length = prefix + 3 * horizon
    mask = torch.zeros(length, length, device=prefix_mask.device, dtype=torch.bool)
    p = slice(0, prefix)
    a = slice(prefix, prefix + horizon)
    s = slice(prefix + horizon, prefix + 2 * horizon)
    f = slice(prefix + 2 * horizon, length)
    mask[p, p] = True
    mask[a, :] = True
    for query, own in ((s, s), (f, f)):
        mask[query, p] = True
        mask[query, a] = True
        mask[query, own] = True
    suffix_mask = torch.ones(batch, 3 * horizon, device=prefix_mask.device, dtype=torch.bool)
    valid = torch.cat((prefix_mask, suffix_mask), dim=1)
    return mask.unsqueeze(0) & valid[:, :, None] & valid[:, None, :]


def _attention_bias(mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask[:, None], 0.0, -2.3819763e38)


def _time_embedding(times: torch.Tensor, width: int) -> torch.Tensor:
    if width % 2:
        raise ValueError("Time embedding width must be even")
    # The reference implementation computes frequencies in float64,
    # then casts the embedding back to the timestep dtype before the time MLP.
    dtype = torch.float32 if times.device.type == "mps" else torch.float64
    fractions = torch.linspace(
        0.0, 1.0, width // 2, dtype=dtype, device=times.device,
    )
    periods = 4e-3 * (4.0 / 4e-3) ** fractions
    scaling = 1.0 / periods * 2 * math.pi
    angles = scaling * times[..., None]
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1).to(times.dtype)


def normalize_signal(signal: torch.Tensor) -> torch.Tensor:
    """Reference min/max normalization for binary labels with min=0, max=1.

    The reference adds epsilon to the range before dividing, then maps to
    ``[-1, 1]``.  This is deliberately not the mathematically exact affine
    map, since the epsilon is part of the trained policy's preprocessing.
    """
    return (signal / (1.0 + 1e-6)) * 2.0 - 1.0


def unnormalize_signal(signal: torch.Tensor) -> torch.Tensor:
    """Reference Unnormalize omits epsilon; do not invent an exact inverse."""
    return (signal + 1.0) / 2.0


def _per_token_norm_forward(self, x, cond=None):
    """Preserve vendored RMS math while allowing [B, sequence, width] cond."""
    output = self._norm(x)
    if cond is None or self.dense is None:
        return (output * (1.0 + self.weight.float())).type_as(x), None
    if cond.shape[-1] != self.cond_dim:
        raise ValueError("Unexpected adaRMS condition width")
    if cond.ndim == 3 and cond.shape[:2] != x.shape[:2]:
        raise ValueError("Per-token adaRMS conditions must match the suffix")
    modulation = self.dense(cond.to(self.dense.weight.dtype))
    if x.ndim == 3 and modulation.ndim == 2:
        modulation = modulation.unsqueeze(1)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    output = output * (1.0 + scale.float()) + shift.float()
    return output.type_as(x), gate.type_as(x)


def install_per_token_adarms(base_model) -> int:
    """Patch this Echo instance only; retain names, weights, and checkpointing."""
    count = 0
    for module in base_model.action_head.model.modules():
        if getattr(module, "dense", None) is None or not hasattr(module, "_norm"):
            continue
        if not hasattr(module, "cond_dim"):
            continue
        module.forward = MethodType(_per_token_norm_forward, module)
        count += 1
    if count == 0:
        raise ValueError("PI05 action expert exposes no adaptive RMS norms")
    return count


class EchoSignalProjections(nn.Module):
    """The two independent SS/FS projection pairs used by the reference."""

    def __init__(self, padded_action_dim: int, width: int, robot_action_dim: int = 7):
        super().__init__()
        if padded_action_dim < robot_action_dim + 2:
            raise ValueError("Signal dimensions do not fit the model action width")
        self.padded_action_dim = padded_action_dim
        self.ss_dim = robot_action_dim
        self.fs_dim = robot_action_dim + 1
        self.ss_in_proj = nn.Linear(padded_action_dim, width)
        self.ss_out_proj = nn.Linear(width, padded_action_dim)
        self.fs_in_proj = nn.Linear(padded_action_dim, width)
        self.fs_out_proj = nn.Linear(width, padded_action_dim)

    def embed(self, ss, fs):
        ss_input = F.pad(ss, (self.ss_dim, self.padded_action_dim - self.ss_dim - 1))
        fs_input = F.pad(fs, (self.fs_dim, self.padded_action_dim - self.fs_dim - 1))
        ss_emb = self.ss_in_proj(ss_input.to(self.ss_in_proj.weight.dtype))
        fs_emb = self.fs_in_proj(fs_input.to(self.fs_in_proj.weight.dtype))
        return ss_emb, fs_emb

    def decode(self, ss_hidden, fs_hidden):
        ss = self.ss_out_proj(ss_hidden.to(self.ss_out_proj.weight.dtype))
        fs = self.fs_out_proj(fs_hidden.to(self.fs_out_proj.weight.dtype))
        return ss[..., self.ss_dim:self.ss_dim + 1].float(), fs[..., self.fs_dim:self.fs_dim + 1].float()


def masked_echo_loss(action_error, ss_error, fs_error, action_mask, ss_mask,
                     fs_mask, action_is_pad, config: EchoModelConfig,
                     action_bound=None, ss_bound=None, fs_bound=None):
    """Preserve the historical masked reduction with dataset-supplied bounds.

    ``*_mask`` determines the valid-loss denominator.  ``*_bound`` is the
    policy-side ``in_episode_bound`` multiplier applied before reduction.  In
    particular, recovery keeps padded actions in the denominator while its
    action bound multiplies those errors by zero. Recovery SS now has a true
    bound over the entire horizon, matching ordinary success supervision.
    """
    if action_bound is None:
        action_bound = torch.ones(action_error.shape[:2], dtype=torch.bool,
                                  device=action_error.device)
    if ss_bound is None:
        ss_bound = torch.ones(ss_error.shape[:2], dtype=torch.bool,
                              device=ss_error.device)
    if fs_bound is None:
        fs_bound = torch.ones(fs_error.shape[:2], dtype=torch.bool,
                              device=fs_error.device)
    ss_scale = torch.where(
        action_is_pad.any(dim=1),
        torch.ones(action_error.shape[0], device=action_error.device),
        torch.full((action_error.shape[0],), config.ss_no_padding_weight,
                   device=action_error.device),
    )[:, None, None]
    errors = (
        action_error * action_bound[..., None],
        ss_error * ss_scale * config.ss_loss_weight * ss_bound[..., None],
        fs_error * config.fs_loss_weight * fs_bound[..., None],
    )
    masks = (action_mask, ss_mask, fs_mask)
    numerators = [(error * mask).sum() for error, mask in zip(errors, masks)]
    counts = [mask.sum() for mask in masks]
    numerator = sum(numerators)
    count = sum(counts)
    loss = numerator / count.clamp_min(1)
    return {
        "loss": loss,
        "action_loss": numerators[0] / counts[0].clamp_min(1),
        "ss_loss": numerators[1] / counts[1].clamp_min(1),
        "fs_loss": numerators[2] / counts[2].clamp_min(1),
        "action_numerator": numerators[0],
        "ss_numerator": numerators[1],
        "fs_numerator": numerators[2],
        "action_count": counts[0],
        "ss_count": counts[1],
        "fs_count": counts[2],
        "loss_numerator": numerator,
        "loss_count": count,
    }


class EchoPI05Adapter(nn.Module):
    """Full training and cached inference around one existing PI05 instance.

    Dataset actions retain baseline quantile normalization. Dataset signals are
    [0,1] labels and are mapped here to [-1,1]. Masks supplied by the dataset
    already include the three data types and recovery boundary rules.
    """

    model_name = "PI05"

    def __init__(self, base_model, config: EchoModelConfig | None = None):
        super().__init__()
        self.base = base_model
        self.echo_config = config or EchoModelConfig()
        if base_model.action_horizon != self.echo_config.horizon:
            raise ValueError("Base PI05 and Echo horizons differ")
        if not getattr(base_model.action_head, "use_adarms", False):
            raise ValueError("Echo requires the PI05 adaptive action expert")
        self.signals = EchoSignalProjections(
            base_model.action_dim, base_model.action_head.action_in_proj.out_features,
            self.echo_config.robot_action_dim,
        )
        self.signals.to(device=base_model.action_head.action_in_proj.weight.device,
                        dtype=base_model.action_head.action_in_proj.weight.dtype)
        self.adarms_norm_count = install_per_token_adarms(base_model)

    @property
    def config(self):
        return self.base.config

    @property
    def action_horizon(self):
        return self.echo_config.horizon

    @property
    def action_dim(self):
        return self.base.action_dim

    @property
    def action_head(self):
        return self.base.action_head

    def to_precision(self, precision):
        self.base.to_precision(precision)
        self.signals.to(dtype=self.action_head.action_in_proj.weight.dtype)
        return self

    def _embed_suffix(self, actions, ss, fs, action_time):
        batch, horizon = actions.shape[:2]
        if horizon != self.action_horizon or actions.shape[-1] != 7:
            raise ValueError("Expected native [B,50,7] robot actions")
        if action_time.shape != (batch,):
            raise ValueError("Expected one diffusion time per example")
        action_times = action_time[:, None].expand(batch, horizon)
        token_times = torch.cat((
            action_times,
            signal_time(action_times, self.echo_config.ss_t_clean),
            signal_time(action_times, self.echo_config.fs_t_clean),
        ), dim=1)
        head = self.action_head
        action_input = F.pad(actions, (0, self.action_dim - 7))
        action_emb = head.action_in_proj(action_input.to(head.action_in_proj.weight.dtype))
        ss_emb, fs_emb = self.signals.embed(ss, fs)
        suffix = torch.cat((action_emb, ss_emb, fs_emb), dim=1)
        time_emb = _time_embedding(token_times, head.action_in_proj.out_features)
        time_emb = time_emb.to(head.time_mlp_in.weight.dtype)
        conditions = F.silu(head.time_mlp_out(F.silu(head.time_mlp_in(time_emb))))
        return suffix, conditions

    def _decode(self, suffix_hidden):
        horizon = self.action_horizon
        if suffix_hidden.shape[1] != 3 * horizon:
            raise ValueError("Action expert did not return A/SS/FS token blocks")
        action_hidden, ss_hidden, fs_hidden = suffix_hidden.split(horizon, dim=1)
        head = self.action_head
        actions = head.action_out_proj(action_hidden.to(head.action_out_proj.weight.dtype))
        ss, fs = self.signals.decode(ss_hidden, fs_hidden)
        return actions[..., :7].float(), ss, fs

    def _prefix(self, observation):
        images, image_masks, tokens, token_masks, _ = self.base._preprocess_observation(observation)
        return self.base.embed_prefix(images, image_masks, tokens, token_masks)

    def forward_from_observation(self, observation, actions, ss, fs, *, action_mask,
                                 ss_mask, fs_mask, action_is_pad, noise=None,
                                 time=None, return_debug=False,
                                 action_bound=None, ss_bound=None, fs_bound=None):
        batch, horizon = actions.shape[:2]
        if actions.shape != (batch, self.action_horizon, 7):
            raise ValueError("Expected normalized [B,50,7] action targets")
        if ss.shape != (batch, horizon, 1) or fs.shape != ss.shape:
            raise ValueError("Signals must have [B,50,1] shape")
        targets = torch.cat((actions, normalize_signal(ss), normalize_signal(fs)), dim=-1)
        if noise is None:
            noise = self.action_head.sample_action_noise(targets.shape, targets.device)
        else:
            noise = torch.as_tensor(noise, device=targets.device, dtype=torch.float32)
        if noise.shape != targets.shape:
            raise ValueError("Echo noise must have [B,50,9] shape")
        if time is None:
            time = self.action_head.sample_flow_time(batch, targets.device)
        else:
            time = torch.as_tensor(time, device=targets.device, dtype=torch.float32)
        if time.shape != (batch,) or not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
            raise ValueError("Diffusion time must be finite [B] in [0,1]")
        channels = torch.cat((
            time[:, None].expand(batch, 7),
            signal_time(time, self.echo_config.ss_t_clean)[:, None],
            signal_time(time, self.echo_config.fs_t_clean)[:, None],
        ), dim=-1)[:, None]
        noisy = channels * noise + (1.0 - channels) * targets
        suffix, conditions = self._embed_suffix(noisy[..., :7], noisy[..., 7:8],
                                                 noisy[..., 8:9], time)
        prefix, prefix_mask, _ = self._prefix(observation)
        if self.base.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix = prefix.bfloat16()
            suffix = suffix.bfloat16()
        suffix_mask = torch.ones(batch, 3 * horizon, device=prefix_mask.device, dtype=torch.bool)
        all_valid = torch.cat((prefix_mask, suffix_mask), dim=1)
        outputs = self.base.forward_with_action_expert(
            prefix_embs=prefix,
            suffix_embs=suffix,
            attention_mask=_attention_bias(make_echo_attention_mask(prefix_mask, horizon)),
            position_ids=all_valid.cumsum(dim=1) - 1,
            adarms_cond=conditions,
        )
        predicted = self._decode(outputs)
        velocity = noise - targets
        errors = (
            (predicted[0] - velocity[..., :7]).square(),
            (predicted[1] - velocity[..., 7:8]).square(),
            (predicted[2] - velocity[..., 8:9]).square(),
        )
        result = masked_echo_loss(
            *errors,
            action_mask,
            ss_mask,
            fs_mask,
            action_is_pad,
            self.echo_config,
            action_bound=action_bound,
            ss_bound=ss_bound,
            fs_bound=fs_bound,
        )
        if return_debug:
            result.update(noisy=noisy, targets=targets, velocity=velocity, time=time,
                          predicted_velocity=torch.cat(predicted, dim=-1))
        return result

    def forward(self, examples, *, noise=None, time=None, return_debug=False):
        if not examples:
            raise ValueError("Echo batch cannot be empty")
        batch = self.base._prepare_examples(examples, include_actions=True)
        observation = self.base._build_observation_from_batch(batch)
        device = self.action_head.action_in_proj.weight.device

        def stack(name, shape, *, boolean=False):
            try:
                values = np.stack([np.asarray(example[name]) for example in examples])
            except KeyError as exc:
                raise ValueError(f"Echo batch is missing {exc.args[0]}") from exc
            if values.shape != (len(examples), *shape):
                raise ValueError(f"Invalid {name} shape: {values.shape}")
            if boolean:
                if values.dtype != np.bool_:
                    raise ValueError(f"{name} must contain boolean masks")
            elif not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"{name} labels must be finite and in [0,1]")
            return torch.as_tensor(values, device=device,
                                   dtype=torch.bool if boolean else torch.float32)

        horizon = self.action_horizon
        return self.forward_from_observation(
            observation, batch["action"].to(device=device, dtype=torch.float32)[..., :7],
            stack("echo_ss", (horizon,))[..., None],
            stack("echo_fs", (horizon,))[..., None],
            action_mask=stack("echo_action_mask", (horizon, 7), boolean=True),
            ss_mask=stack("echo_ss_mask", (horizon,), boolean=True)[..., None],
            fs_mask=stack("echo_fs_mask", (horizon,), boolean=True)[..., None],
            action_is_pad=stack("action_is_pad", (horizon,), boolean=True),
            action_bound=stack("echo_action_bound", (horizon,), boolean=True),
            ss_bound=stack("echo_ss_bound", (horizon,), boolean=True),
            fs_bound=stack("echo_fs_bound", (horizon,), boolean=True),
            noise=noise, time=time, return_debug=return_debug,
        )

    def _cached_velocity(self, prefix_mask, cache, noisy, action_time):
        suffix, conditions = self._embed_suffix(noisy[..., :7], noisy[..., 7:8],
                                                 noisy[..., 8:9], action_time)
        prefix_length = prefix_mask.shape[1]
        mask = make_echo_attention_mask(prefix_mask, self.action_horizon)[:, prefix_length:]
        positions = prefix_mask.sum(dim=1)[:, None] + torch.arange(
            3 * self.action_horizon, device=prefix_mask.device)[None]
        if self.base.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix = suffix.bfloat16()
        set_eager = getattr(self.action_head, "set_gemma_expert_eager_attention", None)
        if set_eager is not None:
            set_eager()
        output = self.action_head.forward_gemma_expert(
            inputs_embeds=suffix,
            attention_mask=_attention_bias(mask),
            position_ids=positions,
            past_key_values=cache,
            use_cache=False,
            adarms_cond=conditions,
        )
        return torch.cat(self._decode(output.last_hidden_state), dim=-1)

    @torch.inference_mode()
    def sample_actions(self, observation, noise=None, num_steps=10):
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        prefix, prefix_mask, _ = self._prefix(observation)
        self.base.vlm.language_model.config._attn_implementation = "eager"
        output = self.base.vlm.forward_prefix(
            inputs_embeds=prefix,
            attention_mask=_attention_bias(prefix_mask[:, :, None] & prefix_mask[:, None, :]),
            position_ids=prefix_mask.cumsum(dim=1) - 1,
            past_key_values=None,
            use_cache=True,
        )
        batch = prefix.shape[0]
        shape = (batch, self.action_horizon, 9)
        if noise is None:
            x = self.action_head.sample_action_noise(shape, prefix.device)
        else:
            x = torch.as_tensor(noise, device=prefix.device, dtype=torch.float32).clone()
        if x.shape != shape or not torch.isfinite(x).all():
            raise ValueError("Echo inference noise must be finite [B,50,9]")
        for step in range(num_steps):
            # Integer-indexed times avoid an extra step from floating point drift.
            time = x.new_full((batch,), 1.0 - step / num_steps)
            next_time = x.new_full((batch,), 1.0 - (step + 1) / num_steps)
            velocity = self._cached_velocity(prefix_mask, output.past_key_values, x, time)
            robot_delta = (next_time - time)[:, None].expand(batch, 7)
            ss_delta = signal_time(next_time, self.echo_config.ss_t_clean) - signal_time(time, self.echo_config.ss_t_clean)
            fs_delta = signal_time(next_time, self.echo_config.fs_t_clean) - signal_time(time, self.echo_config.fs_t_clean)
            delta = torch.cat((robot_delta, ss_delta[:, None], fs_delta[:, None]), dim=-1)
            x = x + delta[:, None] * velocity
        return {"normalized_actions": F.pad(x[..., :7], (0, self.action_dim - 7)),
                "ss": unnormalize_signal(x[..., 7]),
                "fs": unnormalize_signal(x[..., 8])}

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        if isinstance(examples, dict):
            examples = [examples]
        batch = self.base._prepare_examples(examples, include_actions=False)
        observation = self.base._build_observation_from_batch(batch)
        output = self.sample_actions(
            observation, noise=kwargs.get("noise"),
            num_steps=int(kwargs.get("num_inference_steps", kwargs.get("num_steps", 10))),
        )
        return {name: value.cpu().numpy() for name, value in output.items()}


def build_echo_training_model(checkpoint, tokenizer, horizon=50,
                              gradient_checkpointing=True, device="cuda",
                              echo_config=None):
    """Strictly load an unchanged baseline first, then initialize Echo heads."""
    from omegaconf import OmegaConf
    from echo.models.initialization import build_training_model

    config = EchoModelConfig(**echo_config) if isinstance(echo_config, dict) else echo_config
    config = config or EchoModelConfig(horizon=horizon)
    if config.horizon != horizon:
        raise ValueError("Echo config and requested horizon differ")
    base, metadata, stats = build_training_model(
        checkpoint, tokenizer, horizon=horizon,
        gradient_checkpointing=gradient_checkpointing, device=device,
    )
    model = EchoPI05Adapter(base, config)
    OmegaConf.update(base.config, "echo", asdict(config), force_add=True)
    # Only the suffix output contributes to the policy loss. The final prefix
    # layer supplies K/V before its output projection and FFN, and the final
    # prefix norm is never consumed. Record these known unused branches by
    # parameter identity instead of assuming a backend-specific name/depth.
    prefix = base.vlm.language_model
    last_layer = prefix.layers[-1]
    unused_modules = (
        last_layer.self_attn.o_proj,
        last_layer.post_attention_layernorm,
        last_layer.mlp,
        prefix.norm,
    )
    unused_ids = {
        id(parameter)
        for module in unused_modules
        for parameter in module.parameters()
    }
    known_unused = set(metadata["known_unused_parameter_names"])
    known_unused.update(
        name for name, parameter in base.named_parameters()
        if id(parameter) in unused_ids
    )
    metadata = dict(metadata)
    metadata.update({
        "variant": "echo_mode5",
        "echo_config": asdict(config),
        "echo_projection_initialization": "fresh torch Linear; baseline strict-loaded before extension",
        "echo_signal_normalization": "raw_signal / (1 + 1e-6) * 2 - 1; MIN_MAX min=0,max=1",
        "echo_signal_unnormalization": "(signal + 1) / 2; reference omits epsilon",
        "echo_signal_tokens": "A(H), SS(H), FS(H); shared transformer and time MLP",
        "echo_separate_projections": "SS and FS each have independent in_proj/out_proj",
        "echo_robot_flow_dimensions": 7,
        "echo_robot_flow_padding": "25 unused model-width dimensions are zero; no noise or loss",
        "baseline_flow_difference": (
            "Baseline adds diffusion noise/loss over all 32 model dimensions. "
            "Echo uses independent signal tokens and diffuses "
            "only 7 physical robot dimensions plus the two signal channels. "
            "Native LIBERO action semantics and action normalization are unchanged."
        ),
        "echo_forcing": "piecewise signal time; clean at action time t_clean; no value override",
        "echo_adarms_norm_count": model.adarms_norm_count,
        "echo_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()
                                           if parameter.requires_grad),
        "known_unused_parameter_names": ["base." + name for name in sorted(known_unused)],
        "architecturally_frozen_parameter_names": ["base." + name for name in metadata["architecturally_frozen_parameter_names"]],
    })
    return model, metadata, stats
