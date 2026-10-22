# Development

```bash
python -m unittest discover -s tests/echo -v
```

The test modules cover data preparation and behavior that can change scientific results:

| Module | Coverage |
| --- | --- |
| `test_training.py` | Signal targets, forcing time, attention, masked losses, gripper conversion, deterministic sampling and LR |
| `test_checkpoint.py` | Distributed checkpoint completeness and optimizer/RNG/cursor resume |
| `test_data.py` | Source boundaries, full validation and downloaded archive integrity |
| `test_fetch.py` | Scene archive layout, complete pairs, safe paths and repeated downloads |
| `test_evaluation.py` | First-success latching, continued rollout, sharding and frame-level signal accuracy |

CUDA integration checks additionally exercise a real optimizer update, save and
resume for Echo, baseline training, and full 520-action simulation. The unit
suite runs on CPU; a full training/evaluation experiment requires the inputs and
GPU environment documented in the README.

`echo/` owns the experiment workflow. `starVLA/` retains the PI05 implementation
and its shared dependencies; `LIBERO/` retains the simulator and scene assets.
Upstream robot deployments, alternative models, training frameworks and website
files are omitted. `UPSTREAM_SNAPSHOT.json` records the upstream repository,
source revision and archive checksum. No numerical model behavior is changed by
this directory cleanup.
