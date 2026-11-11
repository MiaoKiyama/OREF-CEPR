# Release validation — 2026-11-10

## Release-scope cleanup

The current training CLI permits only OOF120/replay120 graphQS fitting. Mixed-pool
training, comparison arms, historical scoring/controller entry points and duplicate
historical source directories were removed. The cleanup does not change the
provenance of existing published weights or add a new accuracy measurement.

| Check | Result |
| --- | --- |
| Retained computational cores | 29 blocks matched the prior implementation; only the fixed audit-scope flag was normalized for comparison |
| Short OREF training parity | Both pools: four synthetic optimizer steps spanning CE/soft-IoU; losses, draws, normalization, final model tensors and next RNG draws matched exactly before/after cleanup |
| Real input parity | Both actual 120-scene banks opened; first-scene sampled raw features and graph geometry/relations/masks matched exactly |
| Scope rejection | Removed sources/arms, target labels in fit/calibration/selection, and recipe mutations rejected |
| Truth read guards | Target-array reads through NumPy and pathlib rejected; guards restore normal IO afterward |
| New experiment preparation | Python/C++ dependencies copied and sealed; source tampering and existing-run overwrite rejected |
| Native dependencies | Original OREF routing and object-graph kernels compiled; routing output matched reference |
| Unchanged inference regression | Basic model/Q/objective checks and all five CEPR kernels passed |

```bash
python tests/test_release_scope.py -v
python tests/oref_training_trace.py --output oref_trace.json
```

These are bounded checks, not a full retraining or a new official30 benchmark.

## Earlier original-training restoration

The previous portable-loop smoke results are not used as evidence for the
restored original trainers. The following checks were performed on the restored
source projects in the Linux research environment:

| Check | Result |
| --- | --- |
| Original source integrity | SHA256 checked against the collected original source files; scientific implementations unchanged |
| DINO training contracts | 22 original tests passed using the published DINO source |
| MOS training contracts | 9 original tests passed using the published MOS source |
| DINO full120 | Both archived configs validated; each original sampler produced 38400 indices; real 4x512x512 samples loaded |
| MOS full120 | Original boundary-inclusive sampler produced 38400 samples; real optical/local/global/SAR inputs loaded; LR factors checked |
| DINO five folds | All five folds at epochs 0/4/16 matched the complete saved historical sampling audits exactly; real crop loaded |
| MOS five folds | All five folds at epochs 0/2/8 passed the saved historical plan-hash checks |
| Relocated configurations | MOS and fold0 DINO passed the original configuration/source/pretraining/data binding validators |
| OREF original protocol suite | Passed: original logits/loss, group draws, graph normalization, continuous prefetch, stage transitions, routing, input scopes and tamper rejection |
| Training CLIs | Original DINO/MOS/five-fold/OREF entry imports and help passed |
| Repository regression | Python compilation, basic model/objective/Q checks and diagnostic C++ graph construction checked |

The retained training and release-scope checks can be rerun from the repository root:

```bash
PYTHONPATH=training_original/dino python -m pytest -q training_original/dino/tests/test_dino_web_training.py
PYTHONPATH=training_original/mos python -m pytest -q training_original/mos/tests/test_convnext_upgrade_training.py
python tests/test_release_scope.py -v
python tests/smoke.py -v
python tests/graph_pipeline_smoke.py
```

No complete expert retraining, all fifteen full fold fits, 126000-update OREF fit,
or new official30 accuracy run was performed for this update. The original
training cache/store requirements are documented in [TRAINING.md](TRAINING.md).
The restored training processes have not been run through a full new GPU fit.

## Unchanged inference components

The initial release separately checked strict loading of the three published
expert checkpoints, pretrained model initialization, native-size synthetic GPU
forward/backward, a real Qwen observation, all five CEPR native kernels, and the
final bundle's 3778 embedded-file manifest. Those checks concern the unchanged
model/inference components, not a claim of newly completed original training.

The bundle's archived full raw-image acceptance record contains 30 scenes,
83.09642201424109% mIoU and 94.99315835192752% OA with final pixels equal to its
frozen reference. These remain archived values, not new measurements from this
update. The homepage scores, weights and manuscript were not edited.
