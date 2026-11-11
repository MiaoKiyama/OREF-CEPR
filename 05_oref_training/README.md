# 5. QS-OREF training

The training entry retains the original `native24_object_graph_worker.py` training
loop, restricted to `oof120` and `replay120` with `graphQS`. Removed experiment
sources, ablation arms and target-scoring modes are rejected.
It preserves the exact group draws, continuous prefetch RNG, sampled feature and
graph normalization, valid-mass weights, SAR auxiliary objective, optimizer
resets and checkpoint/calibration sequence.

The full original schedule is 120000 CE updates at LR 0.001, followed by 6000
soft-IoU updates at LR 0.00008. Batch size is 4096, weight decay 0.0001 and gradient
clipping 5. The soft stage starts from the original best calibrated CE checkpoint
without resetting the sampler. The released model uses the selected CE_8000
checkpoint; that selection does not replace the original schedule by 8000 steps.

Prepare a new sealed experiment from the original authenticated OOF and replay
stores (including their configuration/index records, Q sidecars and graph inputs):

```bash
python 05_oref_training/prepare.py --root /data/new_oref \
  --source /data/original_oof_experiment --replay /data/original_replay_experiment
python 05_oref_training/train.py --mode train --config /data/new_oref/config.json \
  --source oof120 --arm graphQS --gpu 0
python 05_oref_training/train.py --mode train --config /data/new_oref/config.json \
  --source replay120 --arm graphQS --gpu 0
```

Preparation copies and binds the released Python and C++ dependencies into the
new experiment automatically.
Configure input roots and host resources in the source/replay configs before
sealing. The original graph feature entry is
`training_original/oref/native24_object_graph_features.py --config ... --domain
{oof,replay} --scene ...`; generate the required scene features before
training. Existing sealed experiments are never overwritten.

The original stores and large training caches are external inputs, not part of
the final official30 inference bundle. `prepare_example_bank.py` is retained only
as a small diagnostic feature example; its output is not accepted as a substitute
for the original training stores. See [scope](../docs/TRAINING.md).
