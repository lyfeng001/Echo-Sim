"""Strict PI05 or Echo inference, selected from the saved model config."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from echo.io import inside, read_json, sha256


class Policy:
    def __init__(self, checkpoint, tokenizer, device="cuda:0", num_inference_steps=10):
        import torch
        from omegaconf import OmegaConf
        from safetensors.torch import load
        from starVLA.model.framework.VLM4A.PI05 import PI05
        from echo.models.pi05_utils import (
            configure_torch, load_dataset_statistics, canonicalize_model_name)
        from echo.models.model import EchoModelConfig, EchoPI05Adapter
        from echo.models.precision import apply_pi05_adarms_input_cast

        checkpoint = Path(checkpoint)
        receipt = read_json(checkpoint/"complete.json")
        for name in ("config.yaml", "dataset_statistics.json"):
            path = inside(checkpoint, name)
            expected = receipt["files"][name]
            if sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError("Checkpoint config/statistics checksum differs")
        config = OmegaConf.load(checkpoint/"config.yaml")
        if canonicalize_model_name(str(config.framework.name)) != "PI05" or int(config.framework.action_horizon) != 50:
            raise ValueError("Expected saved PI05 H50 configuration")
        if OmegaConf.select(config, "framework.discrete_state_input") is not False:
            raise ValueError("Expected continuous PI05 state input")
        OmegaConf.update(config, "framework.precision", "float32", force_add=True)
        OmegaConf.update(config, "framework.tokenizer.model_path", str(Path(tokenizer).resolve()), force_add=True)
        OmegaConf.update(config, "trainer.pretrained_checkpoint", None, force_add=True)
        configure_torch()
        self.norm_stats = load_dataset_statistics(checkpoint/"dataset_statistics.json", "PI05")
        base = PI05(config)
        base.tokenizer._load()
        echo = OmegaConf.select(config, "echo")
        self.method = "echo" if echo is not None else "baseline"
        if self.method == "echo":
            self.model = EchoPI05Adapter(base, EchoModelConfig(**OmegaConf.to_container(echo, resolve=True)))
        else:
            self.model = base
        expected = receipt["files"]["fp32/model.safetensors"]
        payload = (checkpoint/"fp32/model.safetensors").read_bytes()
        model_sha = hashlib.sha256(payload).hexdigest()
        if model_sha != expected["sha256"] or len(payload) != expected["bytes"]:
            raise ValueError("Model checksum differs")
        weights = load(payload)
        del payload
        if any(v.is_floating_point() and v.dtype != torch.float32 for v in weights.values()):
            raise ValueError("Saved model must contain FP32 floating tensors")
        self.model.load_state_dict(weights, strict=True)
        del weights
        if self.method == "baseline":
            apply_pi05_adarms_input_cast(base)
        self.model.to_precision("bfloat16")
        self.model.to(torch.device(device)).eval()
        self.resize_size = int(base.image_resolution[0])
        self.num_inference_steps = num_inference_steps
        if type(num_inference_steps) is not int or num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self.last_signals = None
        self.metadata = dict(method=self.method, checkpoint_sha256=model_sha, training_step=receipt["step"],
                             config_sha256=receipt["files"]["config.yaml"]["sha256"],
                             statistics_sha256=receipt["files"]["dataset_statistics.json"]["sha256"],
                             tokenizer_sha256=sha256(tokenizer), inference_precision="bfloat16", strict_load=True,
                             num_inference_steps=num_inference_steps)

    def infer(self, observation, instruction):
        import torch
        from echo.models.pi05_utils import (
            OPENPI_MODEL_SOURCE, normalize_openpi_value, postprocess_openpi_actions, preprocess_env_obs)

        example, _ = preprocess_env_obs(observation, instruction, self.resize_size)
        raw_state = example["raw_state"]
        model_input = dict(image=example["image"], lang=example["lang"],
                           state=normalize_openpi_value(raw_state, self.norm_stats["state"], True).astype(np.float32)[None])
        with torch.inference_mode():
            output = self.model.predict_action([model_input], num_inference_steps=self.num_inference_steps)
        normalized = np.asarray(output["normalized_actions"][0], dtype=np.float32)
        actions = postprocess_openpi_actions(normalized, raw_state, self.norm_stats, "PI05", OPENPI_MODEL_SOURCE)
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid native H50 robot actions")
        if self.method == "echo":
            signals = {name: np.asarray(output[name][0], dtype=np.float32) for name in ("ss", "fs")}
            if any(v.shape != (50,) or not np.isfinite(v).all() for v in signals.values()):
                raise ValueError("Invalid H50 status signals")
            self.last_signals = signals
        return np.ascontiguousarray(actions)
