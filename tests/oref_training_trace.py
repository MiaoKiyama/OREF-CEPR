"""Trace the retained graphQS math and continuous sampling on synthetic inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock
import numpy as np
import torch
from oref_release_fixture import array_fixture


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def trace(project, device='cpu'):
    sys.path.insert(0, str(project))
    import native24_object_graph_worker as worker
    from native24_original82_data import canonical
    torch.set_num_threads(2)
    results = {}

    class Bank:
        def __init__(self, source):
            self.ids = [f'train_{i:03}' for i in range(120)]
            self.n = 240
            numeric, labels, self.pack, self.graph_values = array_fixture(self.n)
            self.x = canonical(numeric, labels)
            self.rows = [dict(scene=sid, local_start=2*i, local_end=2*i+2,
                              domain=source, original_fold=i//24) for i, sid in enumerate(self.ids)]

        def raw(self, indices):
            return self.x[indices].copy()

        def target_pack(self):
            return self.pack

    class Inputs:
        def __init__(self, c, bank, source, arm):
            self.bank = bank

        def raw(self, indices):
            return self.bank.raw(indices)

        def graph(self, indices, structure=None):
            return dict(zip(('geometry', 'relations', 'relation_mask'),
                            (x[indices].copy() for x in self.bank.graph_values)))

    with tempfile.TemporaryDirectory() as directory:
        for source in ('oof120', 'replay120'):
            bank = Bank(source)
            c = dict(batch_size=17, normalization_samples=200000)
            with mock.patch.object(worker, 'Inputs', Inputs):
                run = worker.Training(c, source, 'graphQS', Path(directory)/source,
                                      torch, bank=bank, device=device)
            torch.manual_seed(worker.SEED)
            model = worker.make_model(torch, run.mean, run.scale, 'graphQS', worker.GRAPH_META).to(device)
            generator = torch.Generator(device='cpu').manual_seed(worker.SEED)
            prefetch = worker.Prefetch(run, generator, 4)
            losses, draws = [], []
            try:
                # Short diagnostic only. Both original phases share this queue.
                for phase, lr in [('CE', .001), ('CE_softIoU', .00008)]:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
                    for _ in range(2):
                        indices, batch = prefetch.next()
                        loss, *_ = worker.loss_value(model, batch, phase, run.fixed, len(run.valid), torch, True)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                        draws.append(digest(indices))
            finally:
                prefetch.close()
            results[source] = dict(losses=losses, draws=draws, mean=digest(run.mean),
                                   scale=digest(run.scale), graph_norm={k:digest(v) for k,v in run.graph_norm.items()},
                                   state={k:digest(v.detach().cpu().numpy()) for k,v in model.state_dict().items()},
                                   next_rng=digest(torch.randint(1000, (8,), generator=generator).numpy()))
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1]/'training_original/oref')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    args.output.write_text(json.dumps(trace(args.project.resolve(), args.device), indent=2)+'\n')
    print('Passed: two training pools, continuous sampling, CE/soft-IoU backward and optimizer steps')
