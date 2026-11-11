"""Full120 plans over authenticated original caches; official30 is never read."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler

from .ms75_sar_joint_data import MS75SARJointDataset, make_joint_plan, digest
from .ms75_sar_pilot_data import boundary_reservoir
from .ms75_rebuild_data import load_manifest, GlobalViews, _check_paths
from .light_sar_data import _availability, _sha

SCHEMA = 'convnext_sar_full120_sampling_v1_20261002'


class Full120RankSampler(Sampler):
    def __init__(self, *, updates, rank, world_size, start_update=0):
        if (world_size not in (2, 4) or type(rank) is not int or not 0 <= rank < world_size
                or type(updates) is not int or updates < 1 or not 0 <= start_update <= updates):
            raise ValueError('Invalid full120 rank partition')
        self.updates, self.rank, self.start = updates, rank, start_update
        self.local_batch = 64 // world_size

    def __len__(self):
        return (self.updates-self.start)*self.local_batch

    def __iter__(self):
        for update in range(self.start, self.updates):
            first = update*64+self.rank*self.local_batch
            yield from range(first, first+self.local_batch)


def prepare_full_support(raw_manifest, prior_support, output, workers=4):
    m = load_manifest(raw_manifest)
    prior = json.loads(Path(prior_support).read_text())
    if prior['raw_manifest_digest'] != digest(m):
        raise ValueError('Original boundary provenance differs')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    def one(scene):
        if scene in prior['scenes']:
            row = prior['scenes'][scene]
            if _sha(row['path']) != row['sha256'] or row['label_sha256'] != m['scenes'][scene]['label_sha256']:
                raise ValueError('Original boundary points changed')
            return scene, row
        row = m['scenes'][scene]; _check_paths(m, row)
        labels = np.load(row['label'], mmap_mode='r', allow_pickle=False)
        if hashlib.sha256(memoryview(labels)).hexdigest() != row['label_sha256']:
            raise ValueError('Full120 labels changed')
        points, population = boundary_reservoir(labels, capacity=4096,
            seed=int(hashlib.sha256(scene.encode()).hexdigest()[:16], 16))
        p = out/(scene+'.boundary.npy'); np.save(p, points, allow_pickle=False)
        return scene, dict(path=str(p), sha256=_sha(p), population=population,
            count=len(points), label_sha256=row['label_sha256'])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = dict(pool.map(one, m['train_ids']))
    result = dict(schema=SCHEMA, raw_manifest_digest=digest(m), scene_ids=m['train_ids'], scenes=rows,
        full120_includes_former_held24=True, official30_labels_read=False,
        original_support=str(prior_support), original_support_sha256=_sha(prior_support))
    result['payload_sha256'] = digest(result)
    p = out/'manifest.json'; p.write_text(json.dumps(result, indent=2, sort_keys=True))
    return p


class Full120Dataset(MS75SARJointDataset):
    def __init__(self, config, epoch=0, start_sample=0):
        data = config['data']
        self.metadata = load_manifest(data['raw_manifest'])
        self.scene_ids = tuple(self.metadata['train_ids'])
        self.data_root, self.epoch = data['data_root'], int(epoch)
        self.seed = 1337
        self.n, self.sn, self.short_size, self.global_size = 512, 166, 512, 512
        self.updates, self.samples_per_epoch = 600, 38400
        if (data.get('scope') != 'full120' or data['updates_per_epoch'] != 600 or data['seed'] != 1337
                or type(start_sample) is not int or start_sample % 64 or not 0 <= start_sample <= 38400):
            raise ValueError('Full120 data budget/scope differs')
        self.start_sample, self.augmentation, self.max_cache = start_sample, True, 4
        self.availability, self.availability_path = _availability(self.data_root)
        if (len(self.scene_ids), sum(self.availability[s] for s in self.scene_ids)) != (120, 104):
            raise ValueError('Expected official120 with104 SAR and16 missing')
        self.support = json.loads(Path(data['boundary_support']).read_text())
        body = {k:v for k,v in self.support.items() if k != 'payload_sha256'}
        if (self.support.get('schema') != SCHEMA or self.support['payload_sha256'] != digest(body)
                or self.support['raw_manifest_digest'] != digest(self.metadata)
                or self.support['scene_ids'] != list(self.scene_ids)):
            raise ValueError('Full120 boundary source mismatch')
        # Preserve the original raw manifest and views identity. Only sampling expands.
        plan_metadata = dict(self.metadata, selection_train_ids=list(self.scene_ids), held_ids=[])
        boundary = [s for s in self.scene_ids if self.support['scenes'][s]['count'] > 0]
        self.blocks = make_joint_plan(plan_metadata, epoch=self.epoch, updates=600, seed=1337, boundary_scenes=boundary)
        self.plan_sha256 = digest(dict(epoch=self.epoch, seed=1337, blocks=self.blocks, schema=SCHEMA))
        self.global_views = GlobalViews(self.metadata, data['views_cache'], size=512)
        self._cache, self._points, self._boundaries, self._aux = (OrderedDict() for _ in range(4))

    def audit(self):
        return dict(super().audit(), schema=SCHEMA, full120_includes_former_held24=True,
                    official30_used=False)
