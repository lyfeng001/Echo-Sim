"""Instance-local PI05 adaRMS condition-dtype bridge for BF16 inference.

The pinned PI implementation keeps adaRMS dense parameters in FP32 but emits
the conditioning vector from a BF16 time MLP. Vendored GemmaRMSNorm.forward
calls dense(cond) without matching dtypes. This hook performs exactly the
missing cond.to(dense.weight.dtype); no parameters, RMS arithmetic, output
casts, nonadaptive norms, global classes, or other model instances are changed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def _cast_linear_input(module, args):
    if len(args) != 1:
        raise ValueError("Expected the single positional condition tensor from GemmaRMSNorm")
    return (args[0].to(dtype=module.weight.dtype),)


def apply_pi05_adarms_input_cast(model) -> dict:
    """Install once on a loaded PI05 instance; return serializable audit info."""
    from starVLA.model.modules.vlm.openpi_transformers.gemma.modeling_gemma import GemmaRMSNorm

    if str(getattr(model, "model_name", "")).upper() != "PI05":
        raise ValueError("This narrowly scoped patch accepts only a PI05 instance")
    installed = getattr(model, "_pi05_adarms_dtype_patch", None)
    if installed is not None:
        return installed["metadata"]
    records = []
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, GemmaRMSNorm) or module.dense is None:
            continue
        dense = module.dense
        handle = dense.register_forward_pre_hook(_cast_linear_input)
        handles.append(handle)
        records.append({"name": name, "weight_dtype": str(dense.weight.dtype),
                        "bias_dtype": str(dense.bias.dtype) if dense.bias is not None else None,
                        "cond_dim": int(module.cond_dim)})
    if not records:
        raise RuntimeError("No adaptive GemmaRMSNorm found; refusing a silently ineffective patch")
    metadata = {
        "name": "pi05_adarms_condition_matches_dense_parameter_dtype_v1",
        "equivalent_source_fix": "modulation = self.dense(cond.to(self.dense.weight.dtype))",
        "scope": "only adaptive norm dense modules on this PI05 instance",
        "changes_parameters": False,
        "changes_rms_arithmetic": False,
        "patched_module_count": len(records),
        "modules": records,
        "patch_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    # Retain handles for the model lifetime and make repeated calls idempotent.
    model._pi05_adarms_dtype_patch = {"handles": handles, "metadata": metadata}
    return metadata


def self_test() -> None:
    """CPU-only regression: BF16 cond/activation + FP32 adaptive dense."""
    import torch
    from starVLA.model.modules.vlm.openpi_transformers.gemma.modeling_gemma import GemmaRMSNorm

    class TinyPI05(torch.nn.Module):
        model_name = "PI05"

        def __init__(self):
            super().__init__()
            self.norm = GemmaRMSNorm(8, cond_dim=4).float()

    torch.manual_seed(7)
    model = TinyPI05()
    with torch.no_grad():
        model.norm.dense.weight.normal_()
        model.norm.dense.bias.normal_()
    x = torch.randn(2, 3, 8).bfloat16()
    cond = torch.randn(2, 4).bfloat16()
    before = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    # Manual source-level reference makes the required cast explicit.
    with torch.no_grad():
        rms = model.norm._norm(x)
        scale, shift, expected_gate = model.norm.dense(cond.float()).unsqueeze(1).chunk(3, dim=-1)
        expected_output = (rms * (1.0 + scale.float()) + shift.float()).type_as(x)
        expected_gate = expected_gate.type_as(x)
        try:
            model.norm(x, cond)
        except RuntimeError as exc:
            if "dtype" not in str(exc):
                raise
        else:
            raise AssertionError("The unpatched regression case unexpectedly succeeded")
        report = apply_pi05_adarms_input_cast(model)
        output, gate = model.norm(x, cond)
        assert torch.equal(output, expected_output)
        assert torch.equal(gate, expected_gate)
        assert output.dtype == gate.dtype == torch.bfloat16
        assert model.norm.dense.weight.dtype == torch.float32
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
        assert apply_pi05_adarms_input_cast(model) == report
    print("CPU PASS: original dtype regression reproduced; patched result equals explicit FP32 reference; weights unchanged")


if __name__ == "__main__":
    self_test()
