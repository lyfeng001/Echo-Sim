# Data preparation

All paths are relative to the repository root. Change `local_dir` in
`configs/data.json` to reuse an existing local dataset. Revisions and checksums
identify the inputs used by the supplied configurations.

| Source | Episodes | Frames |
| --- | ---: | ---: |
| physical-intelligence/libero | 1,693 | 273,465 |
| yzembodied/libero_90_image | 4,500 | 669,043 |
| ataier/LIBERO_Recovery_Expert | 3,184 | 625,731 |
| echo-iclr/echo-failure-libero | 9,521 | 4,950,920 |

Echo uses all 18,898 segments. The action-only π0.5 baseline uses success and
recovery. The training manifest fixes segment order, language, boundaries and
gripper conventions. Recovery episode IDs retain upstream numbering, including
the missing IDs 1697 and 2968.

## Download and validate

```bash
python -m echo.data.fetch --source all
python -m echo.data.validation
```

The fetch command downloads pinned source revisions, evaluation snapshots,
π0.5 initialization and tokenizer. Failure shards and extracted files are checked
against their SHA-256 inventories. Individual sources can be selected with
`--source success`, `success90`, `recovery`, `failure`, `assets` or `weights`.
For `--source assets`, the pinned ModelScope archive is also safely unpacked so
`data/scenes/selected_bddl_scenes_{spatial,object,goal,10,90}` is ready for the
evaluation selector.

Validation reads all selected numeric arrays, instructions and boundaries,
decodes the images/videos, and hashes each input file. It writes
`data/training_validation.json`, used by training preflight. For a baseline-only
installation, fetch success, success90, recovery and weights, then run:

```bash
python -m echo.data.validation --method baseline
```

A full three-type validation can also be used by the baseline. Training checks
its receipt and input hashes once on rank 0. Rerun validation after changing
source data or the data reader.

## Data conventions

Actions are native 7D end-effector deltas. Dataset gripper values in `[0,1]`
are converted to LIBERO's signed convention before padding. Each H50 training
window stays within its segment. Failure trajectories train the signals and
have zero action supervision. Exact signal targets and masks are specified in
[the experiment protocol](experiments.md#training-protocol).

See [the failure dataset card](FAILURE_DATA_CARD.md) for the archive schema.
