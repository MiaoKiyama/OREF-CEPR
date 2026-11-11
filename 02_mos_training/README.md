# 2. MOS-Net training

This entry runs the original native24 limited-retrain wrapper over the two-rank
full120 trainer. It keeps global batch 64, microbatch 4 per rank, the original
50/30/10/5/5 scene/rare/hard/pond/boundary sampler, spectral augmentation,
two warm-up epochs, 600 updates per epoch, 30-epoch LR horizon and EMA 0.999.
The recorded run stops at epoch 12; the published epoch 9 checkpoint was retained
separately in the historical `extra_candidates` directory. The recorded candidate
save list is preserved; training does not relabel an epoch 12 model as epoch 9.

Prepare optical caches and boundary reservoirs with the original builders:

```bash
python 02_mos_training/prepare.py --data-root /data/FBP --output /data/fbp_cache
```

Use `training_original/bind_config.py --project mos` with
`configs/original/mos_full120.json`, `--prepared-data /data/fbp_cache/prepared_data.json`,
and explicit path mappings for pretrained weights and registered SAR. Set
`--old-project /archive/hot5/yeqi/projects/native24_retrain_20261015_v1/pyramid`.
The new `--run-dir` must end in `native24_retrain_20261015_v1/pyramid`, as required
by the original isolated-output contract. See [configuration](../docs/TRAINING.md).

Set `physical_gpus`, `gpu_uuids` and two disjoint CPU affinity lists for your host.
Keep all scientific settings unchanged. Bind the devices by UUID:

```bash
CUDA_VISIBLE_DEVICES=GPU-UUID0,GPU-UUID1 torchrun --standalone --nproc_per_node=2 \
  02_mos_training/train.py --config runs/mos_config.json --smoke
CUDA_VISIBLE_DEVICES=GPU-UUID0,GPU-UUID1 torchrun --standalone --nproc_per_node=2 \
  02_mos_training/train.py --config runs/mos_config.json
```

The smoke and formal paths are the original trainer's paths. Smoke is diagnostic;
it is not a shortened formal learning-rate schedule.

The complete published inference checkpoint still loads independently:

```bash
python 02_mos_training/load_fbp_mos.py --model weights/FBP_MOSNet.pth --device cpu
```

SAR-ON and SAR-OFF share the same state.
