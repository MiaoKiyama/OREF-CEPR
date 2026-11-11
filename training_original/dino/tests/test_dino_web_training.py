from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from tcmmsline3 import dino_web_data as data_module
from tcmmsline3.dino_web_protocol import (
    CANDIDATE_EPOCHS,
    FULL120_INDEX,
    FULL120_INDEX_SHA256,
    HELD24_SPLIT,
    RUNTIME_BINDING_KEY,
    SELECTION_RULE,
    TRAIN96_INDEX_SHA256,
    TRAIN96_SPLIT,
    build_dino_web_source_manifest,
    freeze_contract,
    load_dino_web_config,
    resolve_dino_web_runtime_binding,
    sha256_file,
    validate_dino_web_checkpoint_runtime_binding,
    validate_dino_web_config,
    validate_selection_provenance,
)
from tcmmsline3.losses import grouped_segmentation_loss, segmentation_loss
from tools import train_dino_web as runner

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "v3b": "dino_web_v3b_train96_gpu5_20260914.yaml",
    "v3l": "dino_web_v3l_train96_gpu6_20260914.yaml",
    "v2l": "dino_web_v2l_train96_gpu7_20260914.yaml",
}


def _config(variant="v3b"):
    return load_dino_web_config(
        ROOT / "configs" / CONFIGS[variant], require_pretrained_file=False
    )


def _full(config, epoch=18):
    result = copy.deepcopy(config)
    result["experiment"]["name"] += "_full120"
    result["data"].update(
        protocol_stage="dino_web_full120",
        train_split="train",
        val_split="train",
        test_split="train",
        official_120_30_protocol=True,
        samples_per_epoch=38400,
    )
    result["data"]["dynamic_sampling"].update(
        class_index=FULL120_INDEX, class_index_sha256=FULL120_INDEX_SHA256
    )
    result["train"].update(
        epochs=epoch + 1,
        max_successful_optimizer_updates=epoch * 600,
        successful_update_milestones=[epoch * 600],
        warmup_steps=2500,
        scheduler_total_successful_optimizer_updates=42000,
        primary_successful_optimizer_update=None,
    )
    result["selection_provenance"] = {
        "source_summary_path": "selected.json",
        "source_summary_sha256": "ab" * 32,
        "selected_epoch": epoch,
        "selected_train96_successful_updates": epoch * 480,
        "selected_full120_successful_updates": epoch * 600,
        "selected_candidate_sha256": "bc" * 32,
        "selection_split": HELD24_SPLIT,
        "selection_rule": SELECTION_RULE,
        "train96_checkpoint_loaded": False,
    }
    return result


class DinoWebProtocolTests(unittest.TestCase):
    def test_actual_three_formal_configs(self):
        for variant in CONFIGS:
            config = _config(variant)
            self.assertEqual(
                config["train"]["successful_update_milestones"],
                [epoch * 480 for epoch in CANDIDATE_EPOCHS],
            )
            self.assertEqual(config["train"]["max_successful_optimizer_updates"], 33600)
            self.assertEqual(
                config["train"]["freeze_modules"], freeze_contract(variant)[0]
            )
            self.assertEqual(
                config["train"]["staged_unfreeze_modules"], freeze_contract(variant)[1]
            )
            validate_dino_web_config(_full(config), require_pretrained_file=False)

    def test_every_physical_batch_preserves_effective64(self):
        for variant in CONFIGS:
            for batch in (1, 2, 4, 8, 16):
                config = _config(variant)
                config["train"].update(
                    batch_size=batch, gradient_accumulation_steps=64 // batch
                )
                validate_dino_web_config(config, require_pretrained_file=False)
                config["train"]["gradient_accumulation_steps"] += 1
                with self.assertRaisesRegex(ValueError, "effective crop batch"):
                    validate_dino_web_config(config, require_pretrained_file=False)

    def test_short_budget_requires_explicit_nonformal_identity(self):
        config = _config()
        config["train"].update(
            epochs=1,
            max_successful_optimizer_updates=10,
            successful_update_milestones=[],
        )
        with self.assertRaises(ValueError):
            validate_dino_web_config(config, require_pretrained_file=False)
        config["experiment"].update(mode="smoke", name="dino_web_v3b_train96_smoke")
        config["train"]["staged_unfreeze_epoch"] = 0
        validate_dino_web_config(config, require_pretrained_file=False)
        self.assertEqual(
            config["train"]["scheduler_total_successful_optimizer_updates"], 33600
        )
        config["train"]["scheduler_total_successful_optimizer_updates"] = 10
        with self.assertRaisesRegex(ValueError, "scheduler_total"):
            validate_dino_web_config(config, require_pretrained_file=False)

    def test_rejects_split_teacher_precision_budget_and_freeze_changes(self):
        mutations = [
            lambda c: c["data"].update(train_split="train"),
            lambda c: c["data"].update(load_pan=True),
            lambda c: c["data"]["dynamic_sampling"].update(
                confusion_index="teacher.npz"
            ),
            lambda c: c["train"].update(amp_dtype="float16"),
            lambda c: c["loss"].update(force_main_losses_fp32=False),
            lambda c: c["train"].update(max_successful_optimizer_updates=9600),
            lambda c: c["train"].update(freeze_modules=[]),
            lambda c: c["train"].update(initialization_checkpoint="train96/last.pt"),
        ]
        for mutation in mutations:
            config = _config()
            mutation(config)
            with self.assertRaises(ValueError):
                validate_dino_web_config(config, require_pretrained_file=False)

    def test_pretrained_actual_sha_is_verified_before_construction(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrained.pth"
            path.write_bytes(b"test pinned official initialization")
            config["model"].update(
                pretrained_file=str(path), pretrained_file_sha256=sha256_file(path)
            )
            with self.assertRaisesRegex(
                ValueError, "official variant pretrained SHA256"
            ):
                validate_dino_web_config(config, require_pretrained_file=False)
            with patch(
                "tcmmsline3.dino_web_protocol.DINO_WEB_VARIANTS",
                {
                    "v3b": SimpleNamespace(
                        sha256=config["model"]["pretrained_file_sha256"]
                    )
                },
            ):
                validate_dino_web_config(config, require_pretrained_file=True)
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "pretrained file SHA256"):
                    validate_dino_web_config(config, require_pretrained_file=True)

    def test_full120_selection_changes_only_duration_not_initialization(self):
        config = _full(_config(), epoch=18)
        validate_dino_web_config(config, require_pretrained_file=False)
        self.assertEqual(config["train"]["max_successful_optimizer_updates"], 10800)
        self.assertEqual(
            config["train"]["scheduler_total_successful_optimizer_updates"], 42000
        )
        for mutate in (
            lambda c: c["train"].update(initialization_checkpoint="train96/ema.pt"),
            lambda c: c["selection_provenance"].update(train96_checkpoint_loaded=True),
            lambda c: c["selection_provenance"].update(selected_epoch=19),
            lambda c: c["selection_provenance"].update(
                selected_full120_successful_updates=8640
            ),
        ):
            changed = copy.deepcopy(config)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_dino_web_config(changed, require_pretrained_file=False)

    def test_resume_foreign_output_is_rejected(self):
        config = _full(_config())
        config["train"]["resume_checkpoint"] = "/another/train96/checkpoints/last.pt"
        with self.assertRaisesRegex(ValueError, "resume must be local"):
            validate_dino_web_config(config, require_pretrained_file=False)

    def test_selection_summary_hash_variant_and_source_are_bound(self):
        config = _full(_config())
        train96_binding = {
            "role": "dino_web_checkpoint_runtime_binding",
            "stage": "train96",
            "mode": "formal",
            "variant": "v3b",
            "pretrained_file_sha256": config["model"]["pretrained_file_sha256"],
            "config_fingerprint": "cd" * 32,
            "source_manifest_sha256": "de" * 32,
        }
        summary = {
            "role": "dino_web_held24_selection",
            "status": "completed",
            "variant": "v3b",
            "pretrained_file_sha256": config["model"]["pretrained_file_sha256"],
            "candidate_epochs": list(CANDIDATE_EPOCHS),
            "selected_epoch": 18,
            "selected_candidate_sha256": "bc" * 32,
            "selection_split": HELD24_SPLIT,
            "selection_rule": SELECTION_RULE,
            "official30_used": False,
            "train96_runtime_binding": train96_binding,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            config["selection_provenance"].update(
                source_summary_path=str(path), source_summary_sha256=sha256_file(path)
            )
            validate_selection_provenance(config, verify_file=True)
            with patch(
                "tcmmsline3.dino_web_protocol.build_dino_web_source_manifest",
                return_value={"source_manifest_sha256": "ef" * 32, "files": []},
            ):
                with self.assertRaisesRegex(
                    ValueError, "selected train96 source manifest"
                ):
                    resolve_dino_web_runtime_binding(
                        config,
                        project_root=ROOT,
                        data_identity={"train_scene_count": 120},
                    )
            summary["variant"] = "v3l"
            path.write_text(json.dumps(summary), encoding="utf-8")
            config["selection_provenance"]["source_summary_sha256"] = sha256_file(path)
            with self.assertRaisesRegex(ValueError, "variant"):
                validate_selection_provenance(config, verify_file=True)

    def test_source_manifest_has_new_family_and_changes_with_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tools").mkdir()
            (root / "tcmmsline3").mkdir()
            for relative in (
                "train.py",
                "tools/train_dino_web.py",
                "tcmmsline3/dino_web_model.py",
            ):
                (root / relative).write_text("# first\n", encoding="utf-8")
            first = build_dino_web_source_manifest(root)
            self.assertEqual(first["role"], "dino_web_runtime_source_manifest")
            (root / "tcmmsline3/dino_web_model.py").write_text(
                "# changed\n", encoding="utf-8"
            )
            self.assertNotEqual(
                first["source_manifest_sha256"],
                build_dino_web_source_manifest(root)["source_manifest_sha256"],
            )

    def test_every_runtime_dimension_is_checked_before_state_load(self):
        binding = {
            "source_manifest_sha256": "ab" * 32,
            "config_fingerprint": "bc" * 32,
            "variant": "v3b",
            "stage": "train96",
            "mode": "formal",
            "pretrained_file_sha256": "cd" * 32,
            "data_identity": {"train_scene_count": 96},
        }
        checkpoint = {"data_provenance": {RUNTIME_BINDING_KEY: binding}}
        validate_dino_web_checkpoint_runtime_binding(checkpoint, binding)
        for key in binding:
            changed = copy.deepcopy(binding)
            changed[key] = "foreign"
            with self.assertRaisesRegex(RuntimeError, "before state load"):
                validate_dino_web_checkpoint_runtime_binding(checkpoint, changed)


class DinoWebDataTests(unittest.TestCase):
    def test_real_coordinate_validation_reads_only_active_training_labels(self):
        for full in (False, True):
            config = _full(_config()) if full else _config()
            count = 120 if full else 96
            train_ids = [f"scene_{i:03d}" for i in range(count)]
            held_ids = [] if full else [f"held_{i:03d}" for i in range(24)]
            official = train_ids if full else train_ids + held_ids
            class_index = SimpleNamespace(
                source_split="official_train" if full else "official_train_oof_fold",
                scene_ids=train_ids,
                seed=1337,
                stride=16,
                scenes_by_class={i: train_ids for i in range(24)},
                by_scene={
                    scene: {i: np.array([[i, 0]], dtype=np.int64) for i in range(24)}
                    for scene in train_ids
                },
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "labels").mkdir()
                config["data"]["root"] = str(root)
                for scene in train_ids:
                    Image.fromarray(
                        np.arange(1, 25, dtype=np.uint8).reshape(24, 1)
                    ).save(root / "labels" / f"{scene}.png")
                splits = {
                    "train": official,
                    TRAIN96_SPLIT: train_ids,
                    HELD24_SPLIT: held_ids,
                }
                index_hash = FULL120_INDEX_SHA256 if full else TRAIN96_INDEX_SHA256
                with (
                    patch.object(
                        data_module,
                        "validate_fbp_splits",
                        return_value={"train": official, "test": ["official_test"]},
                    ),
                    patch.object(
                        data_module,
                        "read_fbp_split",
                        side_effect=lambda _root, split: splits[split],
                    ),
                    patch.object(data_module, "sha256_file", return_value=index_hash),
                    patch.object(
                        data_module, "FBPSamplingIndex", return_value=class_index
                    ),
                ):
                    receipt = data_module.audit_dino_web_data_identity(config)
                    self.assertEqual(receipt["train_scene_count"], count)
                    self.assertEqual(receipt["class_anchor_point_counts"], [count] * 24)
                    self.assertFalse(receipt["official30_rasters_or_labels_read"])
                    self.assertTrue(receipt["sampling_coordinate_labels_verified"])
                    class_index.by_scene[train_ids[0]][0] = np.array([[-1, 0]])
                    with self.assertRaisesRegex(ValueError, "coordinate disagrees"):
                        data_module.audit_dino_web_data_identity(config)

    def test_shared_sampler_exact_ratios_and_physical_batch_invariance(self):
        for scenes, samples, stage in ((96, 30720, "train96"), (120, 38400, "full120")):
            schedule = data_module.exact_sampling_arm_schedule(
                samples // 16,
                scene_fraction=0.5,
                rare_fraction=0.3,
                hard_fraction=0.2,
                seed=1337,
            )
            blocks = [
                SimpleNamespace(scene_id=f"scene_{i % scenes}", kind=kind, key=i % 24)
                for i, kind in enumerate(schedule)
            ]
            order = data_module.four_scene_block_order([b.scene_id for b in blocks])
            digests = set()
            for batch in (1, 2, 4, 8, 16):
                audit = data_module.sampling_order_audit(
                    blocks, order, epoch=0, physical_batch=batch, stage=stage
                )
                self.assertEqual(
                    audit["arm_crop_counts"],
                    {
                        "scene": samples // 2,
                        "rare": samples * 3 // 10,
                        "hard": samples // 5,
                    },
                )
                self.assertEqual(audit["optimizer_update_count"], samples // 64)
                self.assertTrue(audit["all_updates_four_distinct_scenes"])
                digests.add(audit["ordered_plan_sha256"])
            self.assertEqual(len(digests), 1)

    def test_fitting_loader_cannot_open_held24_or_official30(self):
        with self.assertRaisesRegex(ValueError, "non-train"):
            data_module.build_dino_web_dataloader(_config(), "val")


class DinoWebTrainingTests(unittest.TestCase):
    def test_import_and_scoped_hooks_leave_mature_trainer_unchanged(self):
        original = runner.legacy.build_model
        with runner.installed_training_hooks():
            self.assertIs(runner.legacy.build_model, runner.build_dino_web_model)
        self.assertIs(runner.legacy.build_model, original)

    def test_bfloat_logits_produce_fp32_losses_and_gradients(self):
        config = _config()
        logits = torch.randn(1, 24, 4, 5, dtype=torch.bfloat16, requires_grad=True)
        outputs = {
            "logits": logits,
            "ms_logits": logits,
            "flat_logits": logits,
            "ms_shared": logits[:, :1],
            "sar_gate": logits.new_zeros((1, 1, 1, 1)),
            "sar_scale": logits.new_zeros(1),
        }
        loss, components = segmentation_loss(
            outputs,
            torch.randint(0, 24, (1, 4, 5)),
            torch.zeros(1, 1, 1, dtype=torch.bool),
            config,
        )
        for value in (
            loss,
            components["final_ce"],
            components["dice"],
            components["lovasz"],
        ):
            self.assertEqual(value.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_per_image_objective_is_invariant_across_all_probe_batches(self):
        torch.manual_seed(39)
        config = _config()
        initial = torch.randn(64, 24, 2, 3)
        target = torch.randint(0, 24, (64, 2, 3))
        for index in range(64):
            target[index].view(-1)[1 + index % 6 :] = -1

        def run(batch):
            logits = initial.clone().requires_grad_()
            total = 0.0
            for start in range(0, 64, batch):
                chunk = logits[start : start + batch]
                outputs = {
                    "logits": chunk,
                    "ms_logits": chunk,
                    "flat_logits": chunk,
                    "ms_shared": chunk[:, :1],
                    "sar_gate": chunk.new_zeros((batch, 1, 1, 1)),
                    "sar_scale": chunk.new_zeros(1),
                }
                loss, _ = grouped_segmentation_loss(
                    outputs,
                    target[start : start + batch],
                    torch.zeros(batch, 1, 1, dtype=torch.bool),
                    config,
                    group_size=1,
                )
                (loss / (64 // batch)).backward()
                total += float(loss.detach()) / (64 // batch)
            return total, logits.grad

        expected_loss, expected_grad = run(16)
        for batch in (1, 2, 4, 8):
            loss, gradient = run(batch)
            self.assertAlmostEqual(loss, expected_loss, places=5)
            torch.testing.assert_close(gradient, expected_grad, rtol=2e-6, atol=2e-8)

    def test_real_optimizer_groups_include_frozen_staged_parameters(self):
        config = _config()
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2) for _ in range(12)]
        )
        model.backbone.norm = torch.nn.LayerNorm(2)
        model.decoder = torch.nn.Linear(2, 24)
        staged = [
            p for block in list(model.backbone.blocks)[6:] for p in block.parameters()
        ]
        staged += list(model.backbone.norm.parameters())
        for parameter in staged:
            parameter.requires_grad_(False)
        optimizer, scheduler, steps, horizon = (
            runner.build_dino_web_optimizer_and_scheduler(
                model, config, staged_parameters=staged
            )
        )
        grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertTrue({id(p) for p in staged}.issubset(grouped))
        self.assertFalse(
            {
                id(p)
                for block in list(model.backbone.blocks)[:6]
                for p in block.parameters()
            }
            & grouped
        )
        self.assertEqual((steps, horizon), (480, 33600))
        self.assertEqual(scheduler.last_epoch, 0)

    def test_failed_optimizer_attempt_does_not_advance_scheduler(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        scheduler = runner.legacy._scheduler(optimizer, 33600, 2000)

        class Scaler:
            def __init__(self, skip):
                self.skip, self.scale = skip, 16.0

            def is_enabled(self):
                return True

            def get_scale(self):
                return self.scale

            def step(self, opt):
                if not self.skip:
                    opt.step()

            def update(self):
                if self.skip:
                    self.scale /= 2

        successes = 0
        for skip in (False, True, False):
            optimizer.zero_grad()
            parameter.square().backward()
            if runner.legacy._scaled_optimizer_step(Scaler(skip), optimizer):
                scheduler.step()
                successes += 1
        self.assertEqual((successes, scheduler.last_epoch), (2, 2))

    def test_disk_guard_preserves_existing_recovery_on_insufficient_space(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            config["experiment"]["output_dir"] = directory
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "last.pt").write_bytes(b"previous durable recovery")
            with (
                patch.object(
                    runner.shutil, "disk_usage", return_value=SimpleNamespace(free=1)
                ),
                patch.dict(
                    runner._ORIGINALS,
                    {
                        "_atomic_save_rolling_recovery": lambda *args,
                        **kwargs: self.fail("must not overwrite")
                    },
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "insufficient free disk"):
                    runner.atomic_save_rolling_recovery(
                        {}, checkpoint_dir=checkpoint_dir, config=config
                    )
            self.assertEqual(
                (checkpoint_dir / "last.pt").read_bytes(), b"previous durable recovery"
            )

    def test_serialized_recovery_preserves_raw_optimizer_scheduler_ema_and_rng_trajectory(
        self,
    ):
        def objects():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 3)
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=2e-4, weight_decay=0.05
            )
            scheduler = runner.legacy._scheduler(optimizer, 33600, 2000)
            ema = AveragedModel(
                model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True
            )
            return model, optimizer, scheduler, ema

        def update(state):
            model, optimizer, scheduler, ema = state
            optimizer.zero_grad()
            model(torch.randn(2, 4)).square().mean().backward()
            optimizer.step()
            scheduler.step()
            ema.update_parameters(model)

        def checkpoint(state, count):
            model, optimizer, scheduler, ema = state
            # torch 2.1 lacks uint32 tensor conversion; the production runtime
            # supports it. Preserve the exact NumPy RNG words as int64 in this
            # test-only compatibility shim, without changing train.py.
            numpy_rng = np.random.get_state()
            compatible_numpy_rng = (
                numpy_rng[0],
                numpy_rng[1].astype(np.int64),
                *numpy_rng[2:],
            )
            with patch.object(
                runner.legacy.np.random, "get_state", return_value=compatible_numpy_rng
            ):
                rng_state = runner.legacy._capture_rng_state()
            return {
                "selection_metric": "last_epoch",
                "model": ema.module.state_dict(),
                "training_model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": {},
                "global_optimizer_step": count,
                "ema_n_averaged": int(ema.n_averaged),
                "rng_state": rng_state,
            }

        def nested_equal(left, right):
            if torch.is_tensor(left):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            elif isinstance(left, dict):
                self.assertEqual(set(left), set(right))
                for key in left:
                    nested_equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    nested_equal(a, b)
            else:
                self.assertEqual(left, right)

        torch.manual_seed(443)
        uninterrupted = objects()
        for _ in range(6):
            update(uninterrupted)
        expected = checkpoint(uninterrupted, 6)
        torch.manual_seed(443)
        interrupted = objects()
        for _ in range(3):
            update(interrupted)
        with tempfile.TemporaryDirectory() as directory:
            config = _config()
            config["experiment"]["output_dir"] = directory
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            with (
                patch.object(runner, "_CONFIG", config),
                patch.object(runner, "_LAST_SAVE", None),
            ):
                state = checkpoint(interrupted, 3)
                runner.atomic_torch_save_with_identical_best_link(
                    state, checkpoint_dir / "last.pt"
                )
                runner.atomic_torch_save_with_identical_best_link(
                    dict(state), checkpoint_dir / "best.pt"
                )
                recovered = torch.load(checkpoint_dir / "best.pt", weights_only=False)
        resumed = objects()  # Deliberately consume RNG before restoration.
        resumed[0].load_state_dict(recovered["training_model"])
        resumed[1].load_state_dict(recovered["optimizer"])
        resumed[2].load_state_dict(recovered["scheduler"])
        resumed[3].module.load_state_dict(recovered["model"])
        resumed[3].n_averaged.fill_(recovered["ema_n_averaged"])
        runner.legacy._restore_rng_state(recovered["rng_state"])
        for _ in range(3):
            update(resumed)
        actual = checkpoint(resumed, 6)
        for key in (
            "model",
            "training_model",
            "optimizer",
            "scheduler",
            "scaler",
            "ema_n_averaged",
        ):
            nested_equal(expected[key], actual[key])
        torch.testing.assert_close(
            expected["rng_state"]["torch_cpu"],
            actual["rng_state"]["torch_cpu"],
            rtol=0,
            atol=0,
        )


class DinoWebRecoveryLinkTests(unittest.TestCase):
    def _state(self):
        return {
            "selection_metric": "last_epoch",
            "config_fingerprint": "ab" * 32,
            "global_optimizer_step": 5,
            "model": {"w": torch.tensor([1.0])},
            "training_model": {"w": torch.tensor([2.0])},
            "optimizer": {"state": {0: {"exp_avg": torch.tensor([3.0])}}},
            "rng_state": {"cpu": torch.tensor([4], dtype=torch.uint8)},
        }

    def test_unchanged_last_best_link_and_rolling_inode_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config()
            config["experiment"]["output_dir"] = directory
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            with (
                patch.object(runner, "_CONFIG", config),
                patch.object(runner, "_LAST_SAVE", None),
            ):
                state = self._state()
                runner.atomic_torch_save_with_identical_best_link(
                    dict(state), checkpoint_dir / "last.pt"
                )
                runner.atomic_torch_save_with_identical_best_link(
                    dict(state), checkpoint_dir / "best.pt"
                )
                self.assertEqual(
                    (checkpoint_dir / "last.pt").stat().st_ino,
                    (checkpoint_dir / "best.pt").stat().st_ino,
                )
                state["global_optimizer_step"] = 6
                state["training_model"]["w"].add_(10)
                runner.atomic_torch_save_with_identical_best_link(
                    state, checkpoint_dir / "last.pt"
                )
                self.assertNotEqual(
                    (checkpoint_dir / "last.pt").stat().st_ino,
                    (checkpoint_dir / "best.pt").stat().st_ino,
                )
                self.assertEqual(
                    torch.load(checkpoint_dir / "best.pt", weights_only=False)[
                        "global_optimizer_step"
                    ],
                    5,
                )
                self.assertEqual(
                    torch.load(checkpoint_dir / "last.pt", weights_only=False)[
                        "global_optimizer_step"
                    ],
                    6,
                )

    def test_any_payload_change_including_optimizer_data_bypassing_version_rejects_dedup(
        self,
    ):
        mutations = [
            lambda state: state.update(global_optimizer_step=6),
            lambda state: state["model"]["w"].data.add_(1),
            lambda state: state["training_model"]["w"].data.add_(1),
            lambda state: state["optimizer"]["state"][0]["exp_avg"].data.add_(1),
            lambda state: state["rng_state"]["cpu"].data.add_(1),
        ]
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as directory:
                config = _config()
                config["experiment"]["output_dir"] = directory
                checkpoint_dir = Path(directory) / "checkpoints"
                checkpoint_dir.mkdir()
                with (
                    patch.object(runner, "_CONFIG", config),
                    patch.object(runner, "_LAST_SAVE", None),
                ):
                    state = self._state()
                    runner.atomic_torch_save_with_identical_best_link(
                        state, checkpoint_dir / "last.pt"
                    )
                    mutate(state)
                    runner.atomic_torch_save_with_identical_best_link(
                        state, checkpoint_dir / "best.pt"
                    )
                    self.assertNotEqual(
                        (checkpoint_dir / "last.pt").stat().st_ino,
                        (checkpoint_dir / "best.pt").stat().st_ino,
                    )


if __name__ == "__main__":
    unittest.main()
