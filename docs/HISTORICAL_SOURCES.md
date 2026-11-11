# Released training source scope

DINO, MOS and five-fold training retain their original implementation bytes and
sampling/scheduling configurations. OREF keeps the original computational cores
and narrows its public workflow to OOF120/replay120 graphQS fitting. Shared data,
routing and runtime functions have been extracted from the removed experiment
modules; the mixed-pool training banks, comparison trainers, ablation launchers,
experiment controller and target-scoring modes are no longer shipped.

`original_training_hashes.json` binds the current release files. Modified files
retain an original digest where available; it is not a claim that their entire
files are unchanged. `training_core_hashes.json` separately verifies retained
computational blocks against the previous implementation, normalizing only the
fixed training-scope audit flag. Numerical parity is also checked on both pools.

Configuration source inventories cover shipped files only. Unshipped experiment
script records are omitted; data bindings and split-isolation checks remain.
Native routing and graph C++ dependencies are included; compilation does not require the old server's
prebuilt shared libraries. See [training setup](TRAINING.md) and
[validation](VALIDATION.md).
