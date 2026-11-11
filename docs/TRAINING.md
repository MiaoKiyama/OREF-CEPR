# Original FBP training setup

The four training entries use separate original source projects under
`training_original/`. The original training loops and samplers are retained. OREF exposes only
OOF120/replay120 graphQS fitting; comparison experiments and target-scoring
entry points have been removed.

## 1. Inputs

Use Linux, the segmentation environment, the original optical images/labels,
registered SAR and official generic pretrained weights. Copy or verify the small
original split files and point indices from `configs/fbp/sampling/` in the FBP
root. Their bytes are preserved, including the index hashes enforced by DINO.

DINO full120 reads the original TIFFs directly. MOS and five-fold training use
prepared optical arrays, global views, class reservoirs, boundary reservoirs and
fold-specific manifests. The original full120 cache builders are exposed by
`02_mos_training/prepare.py`. They can require hundreds of GB of cache space.

Five-fold configurations require their original `fold_data.json`,
`plans_manifest.json`, raw/views/boundary manifests and SAR asset bindings.
OREF additionally requires the original authenticated OOF and replay stores,
observable/Q sidecars, group targets, graph features and configuration/index
records. These large research stores are external inputs; the repository and
final official30 inference bundle do not include them. This update restores the
original training workflow; it does not claim a clean-checkout reconstruction of
all historical intermediate stores from the public inference bundle.

## 2. Configuration and host paths

Training templates are in `configs/original/`. Source-hash inventories include
only the shipped project files; data and pretrained-weight bindings are retained.
`bind_config.py` creates a new
configuration and output directory; it never overwrites a sealed experiment.
It changes path prefixes and source/output bindings, verifies retained input
hashes and binds the verified published source closure. It does not rewrite
sampling proportions, seeds, augmentation, update counts, scheduler horizons,
losses, optimizer groups, stages, EMA or split membership.

```bash
python training_original/bind_config.py --project fivefold \
  --template configs/original/fivefold/fold0/d2.json \
  --output runs/fold0_d2.json --run-dir runs/fold0/d2 \
  --old-project /archive/hot5/yeqi/projects/native24_fivefold_gpu03456_20261015_v1
```

This example reuses authenticated inputs at their archived locations. For another
machine, add `--map OLD=NEW` for each input root. JSON cache manifests themselves
must already refer to the prepared assets at their new locations; mapping the
outer config does not rewrite or silently re-sign nested data manifests.
Missing or changed bound inputs stop preparation instead of substituting a new
sampler. MOS can explicitly bind new original-builder manifests through
`--prepared-data /data/fbp_cache/prepared_data.json`.

Set physical GPU IDs/UUIDs and CPU affinity for your host. Keep two ranks for
full120 MOS, one rank for OOF MOS, and the recorded microbatch/accumulation.
The launcher adapts only host device authorization. DINO binds the visible GPU
through `CUDA_VISIBLE_DEVICES`. Host/resource changes require a fresh run binding;
resume is allowed only within the same bound run.

## 3. Entrypoints

- [DINO](../01_dino_training/README.md): original full120 trainer and sweep.
- [MOS](../02_mos_training/README.md): original two-rank limited retrain.
- [Five folds](../03_fivefold_training/README.md): original 15 expert jobs.
- [OREF](../05_oref_training/README.md): original CE then soft-IoU workflow.

The original CPU-only supplemental MOS EMA exporter is included as
`training_original/mos/tools/native24_extra_candidates.py`. It explains how
additional epochs 4/6/8/9 were captured from recovery files without modifying the
trainer or its schedule; its original manifest and host bindings must be supplied
for use. The main trainer's recorded native save list remains 10/11/12.

Use each original trainer's smoke/resume options as documented. Diagnostic
prediction and example-bank utilities retained from the initial release are not
substitutes for the authenticated historical OOF/replay stores.

## 4. Verification

See [validation](VALIDATION.md). Source integrity, original contract tests, real
training sample loading, saved sampling-plan parity and relocated input binding
were checked. A full new expert/OREF training run has not been performed in this
update; no new training mIoU is claimed.

## 5. Native dependencies

The release includes the original routing and object-graph C++ sources. Routing
kernels compile automatically into a source-keyed temporary cache on first use;
`OREF_BUILD_DIR` can select the cache root. Graph kernels use the experiment's
`graphs/_build` cache. Linux `g++` and OpenMP are required. The two bank read guards
continue to reject target labels during training, calibration and selection.
