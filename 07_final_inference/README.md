# 7. Final FBP model inference

```bash
python 07_final_inference/download.py --model final --output weights
python 07_final_inference/infer_fbp.py --model weights/FBP_QS_OREF_CEPR.pth --verify-only
python 07_final_inference/infer_fbp.py --model weights/FBP_QS_OREF_CEPR.pth --data-root data/FBP --output runs/fbp_final --gpu 0
```

The final checkpoint contains all three experts, both OREF networks, frozen CEPR
rules, runtime sources and cached Q. Separate expert checkpoint downloads are
optional. It verifies the checkpoint and materializes a runtime cache on first
use. The inference entry is unchanged from the released local FBP package.

Input: all 30 recorded original FBP TIFFs and matching registered SAR/geometry
assets. The bundle checks image identities; resized or re-encoded inputs are not
accepted. It does not read labels or generate Q. This is the frozen official30
route, not an arbitrary-image inference release.

Evaluate after inference, separately:

```bash
python 07_final_inference/infer_fbp.py --model weights/FBP_QS_OREF_CEPR.pth --output runs/fbp_final --evaluate --labels-root data/FBP/labels
```

Final predictions: `runs/fbp_final/fusion/predictions/two_oref/<scene>.npz`,
member `prediction`, uint8 class IDs 1..24. Evaluation ignores label 0 and pools
the full 24-class confusion matrix over all scenes. The archived reference is
83.09642201424109% mIoU; see [validation](../docs/VALIDATION.md) for the present
release's checks and the [protocol scope](../docs/FBP.md).
