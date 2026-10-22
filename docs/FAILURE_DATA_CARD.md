---
pretty_name: Echo Failure LIBERO
task_categories:
  - robotics
language:
  - en
---

# Echo Failure LIBERO

Terminal-failure trajectories for **Echo: Coupled Action and Task Status Signal
Denoising for Vision Language Action Models**.

[Project](https://lyfeng001.github.io/Echo/) ·
[Paper](https://lyfeng001.github.io/Echo/assets/paper/Echo.pdf)

The dataset contains **9,521 trajectories and 4,950,920 frames** collected by a
π0.5 policy from LIBERO-Recovery scenes. Each trajectory reaches the 520-action
time limit without task success. It provides the terminal-failure component of
Echo's three-type simulation training data.

## Download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="echo-iclr/echo-failure-libero",
    repo_type="dataset",
    revision="71e79ed6bcf4c9a6468f6919194a780f84e7dae9",
    local_dir="data/failure",
)
```

All 96 data shards are available. `archives.json` lists shard sizes and SHA-256
checksums; `inventory.json` lists the 47,605 episode files. The Echo codebase's
`python -m echo.data.fetch --source failure` command downloads, verifies and
extracts them automatically.

## Format

Each tar shard contains files with the following layout:

```text
episodes/failure_000000/
  trajectory.npz   # actions: float32 [520,7]; states: float32 [520,8]
  agentview.mp4     # third-person RGB observations
  wrist.mp4        # wrist-camera RGB observations
  episode.json     # length, instruction and outcome
  files.json       # per-file sizes and SHA-256 checksums
```

Videos contain the observation before each policy action, at 20 Hz. The gripper
uses LIBERO's signed action convention. Training windows stay within each
trajectory; failure actions are masked out of the imitation loss.

## Other training data

| Type | Dataset |
| --- | --- |
| Success | [physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero), [yzembodied/libero_90_image](https://huggingface.co/datasets/yzembodied/libero_90_image) |
| Recovery | [ataier/LIBERO_Recovery_Expert](https://www.modelscope.cn/datasets/ataier/LIBERO_Recovery_Expert) |

Evaluation scenes are provided by
[ataier/LIBERO_Recovery_Assets](https://www.modelscope.cn/datasets/ataier/LIBERO_Recovery_Assets).
We thank [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and
[LIBERO-Recovery](https://github.com/liulin815/LIBERO-Recovery) for the simulation
environments and recovery data. Upstream data and assets retain their original terms.
