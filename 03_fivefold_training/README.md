# 3. Five-fold training

Each fold uses the original fixed 96/24 split. The entry dispatches directly to
`train_native24_oof_dino.py` or `train_native24_oof_pyramid.py` in the isolated
historical five-fold source project.

| Expert | Epochs | Updates/epoch | Microbatch | LR horizon | Warmup updates |
| --- | --- | --- | --- | --- | --- |
| DINOv2-L | 17 | 480 | 4 | 33600 updates | 2000 |
| DINOv3-L | 17 | 480 | 4 | 33600 updates | 2000 |
| MOS-Net | 9 | 480 | 4 | 30 epochs | 480 |

All use effective batch 64. DINO keeps four distinct 16-crop scene blocks;
MOS keeps the original boundary-inclusive sampler. The continuous update
sequence, optimizer stages, EMA and resume behavior are unchanged.

The 15 archived configurations are under `configs/original/fivefold/`.
Bind each to a new output and the prepared original fold manifests with
`training_original/bind_config.py --project fivefold`; see
[required artifacts and configuration](../docs/TRAINING.md).

```bash
CUDA_VISIBLE_DEVICES=0 python 03_fivefold_training/train.py --expert dino \
  --config runs/fold0_d2.json --max-updates 1
CUDA_VISIBLE_DEVICES=0 python 03_fivefold_training/train.py --expert dino \
  --config runs/fold0_d2.json --resume
CUDA_VISIBLE_DEVICES=GPU-UUID torchrun --standalone --nproc_per_node=1 \
  03_fivefold_training/train.py --expert mos --config runs/fold0_mos.json
```

`--max-updates 1` uses the original resumable update limit without changing the
LR horizon. Repeat with each fold's d2/d3/pyramid configuration. GPU UUID and CPU
affinity values must describe the current host.

`predict.py` remains a separate diagnostic single-patch prediction utility from
the initial release. It is **not** the historical cache-generation pipeline and
must not be used to claim identical OREF training-bank reconstruction. Original
OREF training consumes the authenticated OOF/replay stores described in part 5.
