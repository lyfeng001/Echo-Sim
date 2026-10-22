# Simulation experiments

## Reference results

The README table contains completed evaluations of the original experiment
checkpoints. The refactored implementation has undergone software integration
checks; the table is not a new 2,500-update benchmark rerun.

| Model | Seed 7 | Seed 17 | Seed 27 | Total |
| --- | ---: | ---: | ---: | ---: |
| π0.5, success + recovery | 493/1500 | 466/1500 | 511/1500 | 1470/4500 |
| Echo, τclean=0.7 | 517/1500 | 520/1500 | 527/1500 | 1564/4500 |

Echo executes all 520 actions, including after success. The historical baseline
reference stops at first success; both use the same first-success metric within
a 520-action horizon. The current evaluator executes all 520 actions for both
models. The simulator's default renderer does not guarantee pixel-identical
images across runs with fixed seeds; see
[rendering reproducibility](evaluation.md#rendering-reproducibility).

Counts and checkpoint identities are available in
[reference_results.json](../assets/reference_results.json). The three seeds are
evaluation seeds of one trained checkpoint per model, not independent training
replicates. Mean success is the mean of the three equally sized seed results.

## Signal accuracy results

Frame-level signal accuracy uses the same Echo checkpoint on all 4,500 full
520-action rollouts, with the [metric definition](evaluation.md#signal-accuracy).
SS and FS are whole-H50 means; both thresholds are 0.5 and the post-success
window is 3 seconds at 20 Hz. Counts below are correct / total.

| Metric | Seed 7 | Seed 17 | Seed 27 | Overall |
| --- | ---: | ---: | ---: | ---: |
| Frame accuracy | 572436/637721 | 579898/640418 | 573376/640042 | 1725710/1918181 |

Overall accuracy is **89.97%**, the pooled correct frame count divided by the
pooled included frame count. Of the 1,564 successful trials, 49 have success
windows truncated by the 520-action limit. Only recorded observation frames
are scored, with predictions held between queries.

[signal_accuracy.json](../assets/signal_accuracy.json) contains the frame counts
(also broken down by ground-truth class), configuration, checkpoint identity
and source checksums. Recompute from an evaluation run with
`python -m echo.evaluation.signals` as shown in the
[evaluation guide](evaluation.md#signal-accuracy).

## Training protocol

This repository implements the three-type simulation experiment. Its numerical
choices are listed here to specify the training objective precisely.

| Component | Three-type simulation |
| --- | --- |
| SS target, all three types | Temporal padding mask; padded positions are 1 |
| Recovery FS transition | Python `round(H*u)`; one-frame segments are all zero |
| Failure FS supervision | Terminal-reaching chunks only |
| Failure action supervision | Masked out |
| Recovery action padding | Zero loss numerator; mask count retained |

Compared with the paper's four-type supervision description, this simulation
variant supervises recovery SS padding and uses `round`, rather than `floor`,
for the recovery FS transition. Signal accuracy uses strict `SS > threshold`
for Success predictions.

Echo's native action width is seven; it adds SS and FS as separate H50 token
streams. The native PI05 baseline retains its 32 model action dimensions and
does not construct signal tokens. Its training pool contains success and
recovery only. Therefore Echo with `tau_clean=0` is still Echo, not the baseline.

Signals use `signal_time = max(0, (action_time - tau_clean)/(1-tau_clean))`.
Action and signal tokens attend to one another; SS and FS cannot directly
attend to each other. Binary labels are normalized as
`2*s/(1+1e-6)-1`, preserving the historical epsilon. The action-noise draw is
restricted to seven Echo dimensions; the baseline diffuses all 32 dimensions.

Each rank computes its masked local mean. DDP averages rank gradients; it does
not divide by a globally pooled count. A bound zeros the loss numerator but
does not remove the corresponding mask count. This matters for padded
recovery actions. Component action/SS/FS losses are logged separately.

The deterministic sampler chooses a type per block of 32 frames, then a
length-weighted segment of that type, then uniformly sampled frames with
replacement. It generates a canonical global sequence with
`default_rng(SeedSequence([seed, epoch, 9127]))`, and gives each rank a contiguous
slice. Default weights are success:failure:recovery = 40:1:20 and baseline
success:recovery = 40:20. These are probabilities, not exact quotas. Frozen
segment order and recovery episode numbering must not be changed.

Training uses FP32 master parameters, BF16 autocast, activation checkpointing,
AdamW and four ranks with batch 32 each (global batch 128, no accumulation).
The LR for the next update is `lr*(completed+1)/warmup` during warmup, then a
cosine using `(completed-warmup)/(max_steps-warmup)`. A paused run keeps this
schedule, optimizer partitions, RNG and data cursor. A complete checkpoint
contains the model plus every rank's optimizer/RNG state. Evaluation loads
the stored FP32 weights strictly, then converts the inference model to BF16.

The three bundled manifests use the same 150 recovery scenes (30 per suite),
two initial gripper settings and five perturbation trials per setting: 1,500
trials per seed. Seeds 17 and 27 change trial randomness while retaining the
seed-7 scenes. Each per-trial seed and ID is derived from
`sha256(f'{seed}|{scene_path}|{gripper}|{trial}')`.
