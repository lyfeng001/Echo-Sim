# Evaluation and signal analysis

## Scenes and seeds

The supplied manifests use the same 150 recovery scenes: 30 from each of five
LIBERO suites, two initial gripper settings and five perturbations per setting.
Each seed therefore has 1,500 trials. Seeds 7, 17 and 27 share scene and initial
state identities and vary trial randomness. Use the same manifest across models.

To create another evaluation seed, edit `episode_seed` in `configs/selection.json`:

```bash
python -m echo.evaluation.selection --config configs/selection.json
```

`mode="same_scenes"` retains the scenes. `mode="new_scenes"` draws new scenes
using `selection_seed` and `scenes_per_suite`. Sampling is independent of outcomes.
The manifest stores scene/state checksums and all per-trial seeds.

## Run evaluation

Edit `checkpoint`, `manifest`, `run_dir` and `device` in `configs/evaluation.json`.
The default loads `runs/echo_tau07/checkpoints/step_00002500`. The saved checkpoint
configuration determines whether the policy is Echo or the native π0.5 baseline.

```bash
# Optional single-trial check, then continue the same run.
CUDA_VISIBLE_DEVICES=0 python -m echo.evaluation.evaluate --config configs/evaluation.json --limit 1
CUDA_VISIBLE_DEVICES=0 python -m echo.evaluation.evaluate --config configs/evaluation.json
```

For the baseline or another seed, copy the evaluation config and change its
checkpoint, manifest and output directory. For example:

```bash
python - <<'PY'
import json
from pathlib import Path
base = json.loads(Path("configs/evaluation.json").read_text())
out = Path("local_config")
out.mkdir(exist_ok=True)
for model in ("echo_tau07", "echo_tau0", "baseline"):
    for seed in (7, 17, 27):
        config = dict(base, checkpoint=f"runs/{model}/checkpoints/step_00002500",
                      manifest=f"manifests/evaluation_seed{seed}.json",
                      run_dir=f"runs/eval_{model}_seed{seed}")
        (out/f"eval_{model}_seed{seed}.json").write_text(json.dumps(config, indent=2)+"\n")
PY
CUDA_VISIBLE_DEVICES=0 python -m echo.evaluation.evaluate --config local_config/eval_baseline_seed7.json
```

All trials execute 520 policy actions at 20 Hz, including after success. A query
predicts H50 and executes the first eight actions. Success is latched at its
first environment event during actions 1–520; losing success later does not
change the outcome. Initial success after settling is recorded separately.
Signals never terminate the rollout.

Each completed trial saves its result, actions/states, environment events and
two camera videos. Echo also saves the complete H50 SS/FS arrays. Completed
trials are verified and skipped on restart. Errors remain errors, not timeouts.

For two evaluation shards, run these in separate terminals:

```bash
CUDA_VISIBLE_DEVICES=0 python -m echo.evaluation.evaluate --config configs/evaluation.json --num-shards 2 --shard-index 0
CUDA_VISIBLE_DEVICES=1 python -m echo.evaluation.evaluate --config configs/evaluation.json --num-shards 2 --shard-index 1
```

The config's `device=cuda:0` and `render_gpu_id=0` refer to each process's visible
GPU. Keep the shard count fixed when resuming the same run.

## Rendering reproducibility

The default MuJoCo/EGL renderer does not guarantee pixel-identical images,
even with fixed trial seeds. This is a simulator rendering issue: repeated
renders of the same static scene can differ slightly under the default
multisample antialiasing. These pixel differences can propagate through policy
actions and change subsequent trajectories and outcomes. Fixed seeds control
trial sampling and policy noise, but do not make the rendering pipeline
deterministic. Use the same evaluation settings across models and report
results over multiple seeds. The default protocol retains the simulator's
rendering settings and executes all 520 actions.

## Success rates and monitoring

```bash
python -m echo.evaluation.results summary --run runs/eval_tau07_seed7 --manifest manifests/evaluation_seed7.json
python -m echo.evaluation.results monitor --run runs/eval_tau07_seed7 \
  --baseline runs/eval_baseline_seed7 --manifest manifests/evaluation_seed7.json --watch 60
```

Success rate is the number of valid completed trials with any environment
success divided by the number of valid completed trials. Reports include suite
breakdowns, completed/expected counts, errors and missing trials. Incomplete
runs are marked `partial`. A baseline comparison uses only completed trial IDs
shared with the target model. Repeat `--run` to monitor multiple models.

## Visualization

```bash
python -m echo.evaluation.visualize --run runs/eval_tau07_seed7 \
  --manifest manifests/evaluation_seed7.json --config configs/signals.json
python -m http.server 8000 --bind 127.0.0.1 --directory runs/eval_tau07_seed7/visualizations
```

Open `http://127.0.0.1:8000/`. The gallery contains synchronized camera videos
and PNG/PDF signal plots. Sampling uses `visualization_sample_seed` and
`visualization_samples`; it draws uniformly from completed trials, then saves
the selected IDs and source hashes for repeatable rendering.

SS and FS are the means over all 50 predicted values, without clipping or
smoothing. Between queries the previous value is held. Time uses `control_hz`,
not playback speed. The bundled README demos use sampling seed 20261013, two
trials per evaluation seed, from the then-completed pools (1367/1387/1348 trials).
All six samples are included; the README previews three of them. Their videos
include a one-second terminal hold after the 26-second rollout, with no new
policy actions or predictions.

## Signal accuracy

We report **frame-level signal accuracy**. Published results and exact counts are in
[Signal accuracy results](experiments.md#signal-accuracy-results).

```bash
python -m echo.evaluation.signals --run runs/eval_tau07_seed7 \
  --manifest manifests/evaluation_seed7.json --config configs/signals.json
```

SS and FS are averaged over the entire 50-step predicted chunk, without
clipping or smoothing. Each query's prediction is held until the next query.
The thresholds and success window are configured in `configs/signals.json`:

- **Success prediction:** SS > `ss_threshold` and FS < `fs_threshold`.
- **Ongoing prediction:** SS < `ss_threshold`.
- **Window:** `success_window_seconds` after the first environment success
  (default 3 seconds). Equality to the SS threshold satisfies neither class.

The ground-truth label and evaluation interval are:

| Rollout | Observation frames | Ground truth |
| --- | --- | --- |
| Timeout | All available frames | Ongoing |
| Successful | Before the first environment success | Ongoing |
| Successful | First success through the end of the window, inclusive | Success |
| Successful | After the window | Excluded |

```text
Frame accuracy = correctly classified included frames / all included frames
```

Counts are pooled across all trials and evaluation seeds, rather than averaging
per-trial percentages. A frame is correct when its prediction satisfies the
threshold condition for its ground-truth class.

Frame `t` is the observation after `t` actions; `success_after_action[t-1]` is
the corresponding event for `t >= 1`. A 520-action rollout has pre-action video
frames 0–519. Only available frames are scored; a window extending beyond the
rollout is truncated. A success at action 520 has no subsequent observation,
so its recorded frames are all scored as Ongoing. Reports include frame counts,
per-trial details, truncated windows and missing signals in `signal_accuracy.json`.
