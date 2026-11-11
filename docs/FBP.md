# FBP workflow

Run commands from the repository root. This first release covers FBP only.
The homepage retains the other datasets' results and download links.

## Environment

Linux x86-64, Python 3.10, a CUDA GPU, and `g++` with OpenMP are required for
the complete pipeline. Use separate segmentation and Qwen environments:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/segmentation.txt
python tests/smoke.py -v
python tests/graph_pipeline_smoke.py
```

For Qwen, create another environment and install `requirements/qwen.txt`.
The reference GPU was an RTX 3090 with 24 GiB VRAM. Full-scene processing also
needs substantial host RAM and disk space; the frozen final pipeline recommends
48–64 GiB available RAM and at least 20 GiB output space, plus the checkpoint
and its approximately 5.2 GB extracted runtime.

## Data and weights

```text
data/FBP/
  images/<scene_id>.tif       # uint8 NIR,R,G,B
  labels/<scene_id>.png       # 0 ignored; semantic classes 1..24 (TIFF also accepted)
  auxiliary/availability.csv
  auxiliary/...              # registered SAR, validity, geometry and metadata
pretrained/
  dinov2_vitl14_pretrain.pth
  dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth
  CROMA_base_0238d814.pt
```

The scene IDs and original five-fold membership are in
[`configs/fbp/splits.json`](../configs/fbp/splits.json). Preserve the registered
SAR package's internal paths. Missing SAR scenes are handled through availability
metadata and zero validity masks. Do not replace the metadata by treating all
scenes as SAR-available.

Use the upstream DINOv2, DINOv3 and CROMA releases for **generic pretrained**
initialization (see [third-party sources](../README.md#third-party-models)). The four FBP
checkpoints on Hugging Face are already trained and are separate from those
generic pretrained files.

```bash
python 07_final_inference/download.py --model all --output weights
```

## Seven parts

1. [DINO expert training](../01_dino_training/README.md)
2. [MOS-Net training](../02_mos_training/README.md)
3. [Five-fold training and prediction](../03_fivefold_training/README.md)
4. [Qwen observations](../04_qwen_observations/README.md)
5. [QS-OREF group construction and training](../05_oref_training/README.md)
6. [Frozen CEPR](../06_cepr/README.md)
7. [Final checkpoint inference and evaluation](../07_final_inference/README.md)

For new training, run parts 1–2 on official120 and part 3 on the five training
folds. Generate image-only Q for the same training scenes with part 4. Use the
OOF predictions and the full-training replay predictions to build two separate
group banks, then train one QS-OREF per bank with part 5.

For the existing published result, go directly to part 7. It contains all three
experts, the two trained OREF networks, frozen CEPR policies and cached Q; the
three standalone expert downloads are unnecessary for this route.

## Training and frozen reproduction are different routes

Training entries use isolated DINO, MOS, five-fold and OREF source projects.
OREF exposes only the two released train120 pools and the graphQS model. Their sampling, augmentation, update budgets, LR schedules, stages,
EMA and recovery logic are retained. See [original training setup](TRAINING.md)
for archived configurations, path binding and the required prepared stores.
The previous portable training loops and replacement crop sampler were removed.

CEPR in part 6 is the **frozen policy implementation** used by the published
bundle, not an automatic policy-fitting tool for newly trained experts. A newly
trained OREF is not a drop-in replacement for the calibrated final package.

The final FBP bundle is bound to its recorded **official30** images and cached Q
observations. Its cohort-level C/E rules require all 30 scenes. It does not run
Qwen or support arbitrary new images. Part 4 can generate Q for new images, but
does not turn the frozen final bundle into a general-image service.

See [validation](VALIDATION.md) for what has actually been executed in this release.
