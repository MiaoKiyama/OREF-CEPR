"""Reject removed experiment modes and verify retained training dependencies."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT/'training_original/oref'
sys.path.insert(0, str(PROJECT))
import native24_object_graph_prepare as prepare
import native24_object_graph_worker as worker
from native24_original82_data import Bank


class Metadata(ast.NodeTransformer):
    def visit_keyword(self, node):
        if node.arg in ('official_truth_used', 'official_truth_used_for_fitting'):
            node.value = ast.Constant(False)
        return self.generic_visit(node)


def canonical(node):
    if isinstance(node, ast.AST):
        return [type(node).__name__, {k:canonical(v) for k,v in ast.iter_fields(node) if k != 'type_params'}]
    if isinstance(node, list):
        return [canonical(v) for v in node]
    if node is Ellipsis or isinstance(node, bytes):
        return repr(node)
    return node


class ReleaseScope(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/original/oref.json').read_text())

    def test_only_two_training_pools(self):
        prepare.validate_config(self.config)
        self.assertEqual(worker.SOURCES, ('replay120', 'oof120'))
        self.assertEqual(worker.ARMS, ('graphQS',))
        for source in ('joint150', 'official30', 'unknown'):
            with self.assertRaises(ValueError):
                worker.training_bank(self.config, source)
            with self.assertRaises(ValueError):
                with worker.fitting_scope(self.config, source):
                    pass
        for arm in ('oldQS', 'graph00', 'no_edgesQS', 'shuffledQS'):
            with self.assertRaises(ValueError):
                worker.arm_spec(arm)

    def test_target_fitting_and_recipe_mutations_rejected(self):
        for source in worker.SOURCES:
            for role in ('fit_ids', 'calibration_ids', 'selection_ids'):
                c = copy.deepcopy(self.config)
                c['pools'][source][role] += c['official30_ids']
                with self.assertRaises(ValueError):
                    prepare.validate_config(c)
            c = copy.deepcopy(self.config)
            c['pools'][source]['official_truth_allowed'] = True
            with self.assertRaises(ValueError):
                prepare.validate_config(c)
        for key in ('batch_size', 'cq_ce_steps', 'cq_soft_steps', 'normalization_samples', 'seed'):
            c = copy.deepcopy(self.config)
            c[key] += 1
            with self.assertRaises(ValueError):
                prepare.validate_config(c)
        with self.assertRaises(ValueError):
            Bank({'train_ids':['a']}, ['a'], domain='official')

    def test_truth_read_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, store, replay, target = (root/n for n in ('source','store','replay','targets'))
            for p in (source, store, replay, target):
                p.mkdir()
            (source/'config.json').write_text(json.dumps({'store':str(store)}))
            allowed, forbidden = store/'counts.npy', target/'counts.npy'
            np.save(allowed, np.ones(1)); np.save(forbidden, np.ones(1))
            (replay/'index.json').write_text(json.dumps({'target_files':[str(allowed)]}))
            c = dict(source=str(source), replay_store=str(replay),
                     pools={s:dict(official_truth_allowed=False) for s in worker.SOURCES})
            for s in worker.SOURCES:
                with worker.fitting_scope(c, s):
                    np.load(allowed)
                    with self.assertRaises(AssertionError):
                        np.load(forbidden)
                    with self.assertRaises(AssertionError):
                        forbidden.read_bytes()
                self.assertTrue(forbidden.read_bytes())

    def test_new_experiment_seal_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, replay, store, replay_store = (root/n for n in ('source','replay','store','replay_store'))
            for p in (source, replay, store, replay_store):
                p.mkdir()
            c = self.config
            source_config = dict(train_ids=c['train120_ids'], official_ids=c['official30_ids'],
                                 controlled=str(root/'controlled'), runtime=str(root/'runtime'),
                                 resources={}, store=str(store),
                                 splits={f:dict(validation=ids) for f,ids in c['report_only_folds'].items()})
            replay_config = dict(source=str(source), replay_root=str(root/'inputs'),
                                 replay_store=str(replay_store), python=sys.executable)
            (source/'config.json').write_text(json.dumps(source_config))
            (replay/'config.json').write_text(json.dumps(replay_config))
            for p in (store, replay_store):
                (p/'index.json').write_text('{}')
            sealed = prepare.seal(root/'run', source, replay)
            prepare.verify(sealed)
            self.assertTrue((root/'run/code/native24_original82_kernels.cpp').exists())
            self.assertEqual(sealed['versions'], ['replay120','oof120'])
            with self.assertRaises(FileExistsError):
                prepare.seal(root/'run', source, replay)
            (root/'run/code/training_common.py').write_text('# changed')
            with self.assertRaises(ValueError):
                prepare.verify(sealed)

    def test_computational_core_unchanged(self):
        records = json.loads((ROOT/'docs/training_core_hashes.json').read_text())
        for key, expected in records.items():
            filename, name = key.split('::')
            tree = ast.parse((PROJECT/filename).read_text())
            node = next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name == name)
            actual = hashlib.sha256(json.dumps(canonical(Metadata().visit(node)), sort_keys=True).encode()).hexdigest()
            self.assertEqual(actual, expected, key)

    def test_native_dependencies(self):
        import native24_original82_fast as fast
        import native24_original82_reference as reference
        rng = np.random.default_rng(1337)
        hist = rng.integers(0,7,(19,24)).astype(np.float64)
        labels = rng.integers(1,25,(19,7),dtype=np.uint8)
        choice = rng.integers(0,7,19,dtype=np.uint8)
        np.testing.assert_array_equal(fast.expected_cm(hist,labels,choice),reference.expected_cm(hist,labels,choice))
        import native24_object_graph_features as features
        with tempfile.TemporaryDirectory() as directory:
            features.library({'graph_store':directory})


if __name__ == '__main__':
    unittest.main()
