# 6. CEPR

`runtime/code/` exposes the frozen C/E/P/R implementation:

| Stage | Main source |
| --- | --- |
| C and E | `frozen_groups_fast.py`, `evidence.py`, `group_cases.py` |
| P and R | `reference_spatial.py`, `fast_spatial.py`, `vendor/` |
| Numerical kernels | `gains.cpp`, `components.cpp`, `context_refine.cpp`, `context_refresh.cpp`, `neighborhood.cpp`, `raw_graph.cpp` |

To run the backend on a manifest produced by the **frozen final pipeline**:

```bash
python 06_cepr/run.py --model weights/FBP_QS_OREF_CEPR.pth --runtime-dir runs/fbp_runtime --inputs runs/fbp_final/observables.json --output runs/cepr --gpu 0
```

`--stage posteriors`, `--stage groups`, and `--stage spatial` expose the stages.
Run them in this order in the same output directory. The wrapper verifies and
extracts the final bundle, including its policies, expert definitions and native
libraries. The checked-in source snapshot is for inspection; use `run.py` rather
than executing snapshot files without their runtime assets.

For a complete raw-image run, use part 7. This frozen policy is specific to the
recorded official30 cohort and checkpoints; it is not a generic calibration tool
for newly trained models or arbitrary images.
