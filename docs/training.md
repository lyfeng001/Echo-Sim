# Training

Run commands from the repository root in the environment described in the README.

## Configuration

Copy `configs/echo_tau07.json`, `echo_tau0.json` or `baseline.json` and edit the
JSON fields. Each experiment needs a distinct `run_dir`.

| Field | Meaning | Default |
| --- | --- | --- |
| `method` | Echo or native action-only π0.5 | `echo` / `baseline` |
| `tau_clean` | Signal denoising time offset; `[0,1)` | `0.7` / `0` |
| `max_steps` | Optimizer updates | 2500 |
| `world_size`, `batch_size` | GPU count and batch per GPU | 4, 32 |
| `lr_schedule` | `cosine` or `constant` after warmup | `cosine` |
| `lr`, `min_lr`, `warmup_steps` | Learning-rate schedule | 1e-5, 1e-6, 125 |
| `sampling_weights` | Relative type probabilities | success:failure:recovery = 40:1:20 |
| `ss_loss_weight`, `fs_loss_weight` | Signal loss coefficients | 1, 1 |
| `save_steps`, `save_interval` | Explicit saves and periodic interval | [10], 500 |
| `seed` | Training seed | 42 |

The baseline uses success:recovery = 40:20 and has no signal tokens. Echo with
`tau_clean=0` retains the signals and uses synchronous denoising.
For warmup + constant learning rate, set `lr_schedule="constant"` and `min_lr=lr`.
The action horizon is 50. Exact continuation uses `num_workers=0` and no gradient
accumulation. Changing GPU count or batch size changes the training setup.

## Launch

```bash
python -m echo.training.train --config configs/echo_tau07.json --preflight
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m echo.training.train --config configs/echo_tau07.json
```

Use `configs/baseline.json` or `configs/echo_tau0.json` for the other models.
Preflight checks pinned inputs, validation receipts and first/middle/tail
chunks from each source before allocating the training model.

## Checkpoints and resume

To first run 10 updates while preserving the full 2,500-update LR schedule:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m echo.training.train --config configs/echo_tau07.json --stop-after 10
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m echo.training.train --config configs/echo_tau07.json \
  --resume runs/echo_tau07/checkpoints/step_00000010
```

This is an alternative to the full launch above. Resume from the run's latest
complete checkpoint, listed in `latest.json`. A checkpoint contains FP32
weights, model configuration, normalization statistics, and every rank's
optimizer, RNG and sampler cursor. `complete.json` is committed after all files
are saved; incomplete checkpoints are never used for resume.

Resume requires the same code, config, inputs and world size. Use a new run
for a new experiment or a code change. Checkpoints are saved locally under
`runs/<run>/checkpoints/`.

## Progress and losses

`runs/<run>/loss.jsonl` records total, action, SS and FS loss for every optimizer
update. `progress.json` records the latest training state and cursor.

```bash
tail -f runs/echo_tau07/loss.jsonl
cat runs/echo_tau07/progress.json
```

Each rank computes a masked mean; DDP averages rank gradients. See
[the protocol](experiments.md#training-protocol) for target and reduction details.
