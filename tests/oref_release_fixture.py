"""Small deterministic bank for release parity tests; no real scene targets."""
import numpy as np

def array_fixture(n=19):
    rng = np.random.default_rng(261016)
    labels = rng.integers(1, 25, (n, 7), dtype=np.uint8)
    labels[::3, 1] = labels[::3, 2]
    numeric = rng.uniform(.05, .95, (n, 144)).astype(np.float32)
    numeric[:, 70:119] = rng.integers(0, 2, (n, 49))
    numeric[:, 125] = 1
    numeric[:, 126:135] = rng.integers(0, 2, (n, 9))
    numeric[:, 135:144] = 1
    counts = rng.integers(0, 7, (n, 25), dtype=np.uint16)
    mass = counts.sum(1, dtype=np.uint32)
    fixed = np.diag(np.arange(25, dtype=np.int64) + 10)
    fixed[0] = 0
    fixed[:, 0] = 0
    fixed_obs = fixed.sum(0)
    baselines = np.empty((7, 25, 25), np.int64)
    for action in range(7):
        cm = fixed.copy()
        for truth in range(25):
            np.add.at(cm[truth], labels[:, action], counts[:, truth])
        baselines[action] = cm
    pack = dict(counts=counts, labels=labels, mass=mass, fixed=fixed,
                fixed_obs=fixed_obs, baselines=baselines,
                pixels=int(mass.sum() + fixed_obs.sum()))
    graph = (
        rng.uniform(0, 1, (n, 7, 12)).astype(np.float32),
        rng.uniform(0, 1, (n, 7, 3, 38)).astype(np.float32),
        rng.integers(0, 2, (n, 7, 3)).astype(np.float32),
    )
    graph[2][0] = 0
    return numeric, labels, pack, graph
