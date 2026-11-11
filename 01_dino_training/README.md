# 1. DINOv2 / DINOv3 training

The entry calls the original `tools/train_dino_web.py` hooks and original
`train.py`. It retains the original label-side point reservoirs, four distinct
16-crop scene blocks per global64, augmentation, BF16, optimizer groups, staged
unfreezing, EMA, successful-update accounting and recovery.

Copy the authenticated files from `configs/fbp/sampling/` into the corresponding
`metadata/` and `splits/` directories of the FBP dataset (verify existing files
rather than overwriting different data). See [configuration](../docs/TRAINING.md).

```bash
python training_original/bind_config.py --project dino \
  --template configs/original/v2l_e17.json --output runs/d2_config.json \
  --run-dir runs/d2 \
  --map /archive/hot5/yeqi/datasets/fbponlymsdata=/data/FBP \
  --map /archive/hot5/yeqi/pretrained/dino_web_20260914=/data/pretrained
CUDA_VISIBLE_DEVICES=0 python 01_dino_training/train.py --config runs/d2_config.json
```

For DINOv3 use `configs/original/v3l_e26.json` and separate config/output paths.
Inspect the archived pretrained path in each template and map it to your actual
file. The file SHA256 remains mandatory.

| Setting | DINOv2-L | DINOv3-L |
| --- | --- | --- |
| Updates per epoch | 600 | 600 |
| Microbatch / accumulation | 8 / 8 | 8 / 8 |
| LR horizon / warmup updates | 42000 / 2500 | 42000 / 2500 |
| Historical sweep terminal epoch | 22 | 30 |
| Released checkpoint epoch | 17 | 26 |

The selected checkpoint epoch does not shorten the cosine schedule or the
original sweep. The release adapter replaces the old server-specific source
provenance binding with a verified published-source binding; it does not replace
any sampler, model, objective or optimizer function.
