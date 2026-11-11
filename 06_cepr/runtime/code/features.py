from common import *
ORDER=np.array([2,4,6,3,5,0,1]);INVERSE=np.argsort(ORDER)
def canonical(numeric,labels):
    """Inverse of the later adapter; restore original raw-expert probability order."""
    n=len(labels);x=np.zeros((n,313),np.float32)
    x[:,:168]=np.eye(24,dtype=np.float32)[labels-1].reshape(n,168)
    cf=numeric[:,:21].reshape(n,3,7)[:,:,INVERSE].reshape(n,21).copy()
    known=numeric[:,70:119].reshape(n,7,7)[:,INVERSE,:].copy()
    pp=(numeric[:,21:70].reshape(n,7,7)*numeric[:,70:119].reshape(n,7,7))[:,INVERSE,:].copy()
    detail=np.any(labels[:,:5]!=labels[:,2:3],axis=1)
    # Original bank has only Top1 evidence outside its first-five disagreement domain.
    missing=~detail
    if missing.any():
        cf[missing,7:]=0
        raw_labels=labels[missing][:,INVERSE]
        hit=raw_labels[:,:,None]==labels[missing,None,:]
        known[missing]=hit
        pp[missing]=cf[missing,:7,None]*hit
    x[:,168:189]=cf;x[:,189:207]=numeric[:,126:144]
    x[:,207:214]=numeric[:,119:126];x[:,214:263]=pp.reshape(n,49)
    x[:,263:312]=known.reshape(n,49);x[:,312]=detail
    assert np.isfinite(x).all();return x
RELATIONS=("contact","overlap","continuation")
SCHEMA="native24_real_object_graph_exact_membership_v1"
def scene_path(c,sid,domain):return Path(c["graph_paths"][sid])
class SceneGraph:
    """Exact real-object convolution sufficient statistics, never group-as-object edges."""
    def __init__(self, c, sid, domain):
        root = scene_path(c, sid, domain)
        self.receipt = read(root / "completed.json")
        assert self.receipt["status"] == "sealed" and self.receipt["schema"] == SCHEMA
        assert self.receipt["ground_truth_read"] is False
        assert sha(root / "graph.npz") == self.receipt["graph_sha256"]
        with np.load(root / "graph.npz", allow_pickle=False) as z:
            self.data = {key: z[key] for key in z.files}
        self.n = len(self.data["primary"])
        self.sid = sid
        self._relations = {}
        extra = self.data["extra"]
        self.extra_rows = extra[:, 0]

    def _node_relations(self, shuffled):
        if shuffled in self._relations:
            return self._relations[shuffled]
        d = self.data
        nodes = len(d["geometry"])
        features = np.concatenate((d["geometry"], np.eye(24, dtype=np.float32)[d["labels"] - 1]), axis=1)
        packed = np.zeros((nodes, 129), np.float32)
        packed[:, :12] = d["geometry"]
        result = packed[:, 12:126].reshape(nodes, 3, 38)
        valid = packed[:, 126:129]
        pixels = self.receipt["audit"]["pixels"]
        for r, name in enumerate(RELATIONS):
            edge = d[name]
            # Undirected stored once; adjacency includes both directions.
            source = np.r_[edge[:, 0], edge[:, 1]]
            dest = np.r_[edge[:, 1], edge[:, 0]].copy()
            weight = np.r_[edge[:, 2], edge[:, 2]].astype(np.float64)
            if shuffled:
                seed = int.from_bytes(hashlib.sha256((self.sid + ":" + name + ":261016").encode()).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                perm = np.arange(nodes)
                for expert in range(7):
                    selected = np.flatnonzero(d["expert"] == expert)
                    perm[selected] = rng.permutation(selected)
                dest = perm[dest]
            den = np.bincount(source, weights=weight, minlength=nodes)
            degree = np.bincount(source, minlength=nodes)
            valid[:, r] = den > 0
            for col in range(36):
                result[:, r, col] = np.bincount(
                    source, weights=weight * features[dest, col], minlength=nodes) / np.maximum(den, 1)
            result[:, r, 36] = np.log1p(den) / np.log1p(pixels * 7)
            result[:, r, 37] = degree / (degree + 1.)
        self._relations[shuffled] = packed
        return packed

    def _aggregate(self, values, indices):
        d = self.data
        ix = np.asarray(indices, np.int64)
        primary = d["primary"][ix].reshape(-1)
        rows = (ix[:, None] * 7 + np.arange(7)).reshape(-1)
        mass = np.repeat(d["mass"][ix], 7).astype(np.float32)
        out = values[primary].astype(np.float32, copy=False)
        starts = np.searchsorted(self.extra_rows, rows, side="left")
        stops = np.searchsorted(self.extra_rows, rows + 1, side="left")
        lengths = stops - starts
        if lengths.sum():
            repeat = np.repeat(np.arange(len(rows)), lengths)
            offsets = np.repeat(starts - np.r_[0, np.cumsum(lengths)[:-1]], lengths) + np.arange(lengths.sum())
            extra = d["extra"][offsets]
            weight = extra[:, 2].astype(np.float32) / mass[repeat]
            shape = (len(weight),) + (1,) * (values.ndim - 1)
            adjustment = (values[extra[:, 1]] - values[primary[repeat]]) * weight.reshape(shape)
            selected = np.flatnonzero(lengths)
            boundaries = np.r_[0, np.cumsum(lengths[selected])[:-1]]
            out[selected] += np.add.reduceat(adjustment, boundaries, axis=0)
        return out.reshape((len(ix), 7) + values.shape[1:])

    def get(self, indices, structure="full"):
        assert structure in ("full", "no_edges", "shuffled")
        ix = np.asarray(indices, np.int64)
        if structure == "no_edges":
            geometry = self._aggregate(self.data["geometry"], ix)
            relations = np.zeros((len(ix), 7, 3, 38), np.float32)
            mask = np.zeros((len(ix), 7, 3), np.float32)
        else:
            packed = self._aggregate(self._node_relations(structure == "shuffled"), ix)
            geometry = packed[:, :, :12]
            relations = packed[:, :, 12:126].reshape(len(ix), 7, 3, 38)
            mask = packed[:, :, 126:129]
        return dict(geometry=geometry, relations=relations, relation_mask=mask)
def normalize_graph(values, norm):
    return (
        (values["geometry"] - norm["geometry_mean"]) / norm["geometry_scale"],
        (values["relations"] - norm["relations_mean"]) / norm["relations_scale"],
        values["relation_mask"],
    )
