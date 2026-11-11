# 4. Qwen observation generation

Activate the separate Qwen environment. The fixed 4x8 grid produces a 512-pixel
RGB context view and a 256-pixel target view per cell, using the archived prompt:

```bash
python 04_qwen_observations/prepare.py --images data/FBP/images --output runs/q_views
python 04_qwen_observations/generate.py --requests runs/q_views/requests.json --output runs/q_records --model /path/to/Qwen3.5-9B --check-inputs
python 04_qwen_observations/generate.py --requests runs/q_views/requests.json --output runs/q_records --model /path/to/Qwen3.5-9B
python 04_qwen_observations/pack.py --records runs/q_records --output runs/q
```

`prepare.py --scene-list train.txt` restricts preparation to one scene ID per
line. None of these scripts reads segmentation labels or model predictions.
Generation uses BF16, SDPA, greedy decoding, 384 maximum new tokens and disabled
thinking. Invalid/truncated responses become unknown observations.

The first nine visible attributes and their knownness flags form `q` with shape
33x18. Row 32 stays zero; the remaining nine semantic/function attributes are
forced unknown. Each sidecar records the source-image SHA256. New generation
may differ from the archived observations; it does not replace Q inside the
frozen final checkpoint.

Model: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).
