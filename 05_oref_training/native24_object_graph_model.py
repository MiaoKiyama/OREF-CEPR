"""SevenNet with observable object geometry and typed relation messages."""
import numpy as np

ARMS = ("graphQS",)
SEED = 261016


def arm_spec(arm):
    if arm != "graphQS":
        raise ValueError("Only the released graphQS model is supported")
    return dict(Q=True, S=True, graph=True, edges=True, shuffled=False)


def mask_raw(raw, arm):
    value = np.asarray(raw, np.float32).copy()
    spec = arm_spec(arm)
    if not spec["Q"]:
        value[:, 189:207] = 0
    if not spec["S"]:
        value[:, 207:214] = 0
    return value


def masked_normalization(mean, scale, arm):
    mean, scale = np.array(mean, copy=True), np.array(scale, copy=True)
    for enabled, sl in ((arm_spec(arm)["Q"], slice(189, 207)),
                        (arm_spec(arm)["S"], slice(207, 214))):
        if not enabled:
            mean[sl], scale[sl] = 0, 1
    return mean, scale


def make_model(torch, mean, scale, arm, graph_meta=None):
    from native24_original82_reference import SevenNet
    spec = arm_spec(arm)
    graph_meta = graph_meta or {}
    relation_width = int(graph_meta.get("relation_width", 38))
    relation_count = int(graph_meta.get("relation_count", 3))
    geometry_width = int(graph_meta.get("geometry_width", 12))

    class ObjectGraphNet(SevenNet):
        def __init__(self):
            super().__init__("attention", mean, scale,
                             1 + 2 * int(spec["Q"]) + 4 * int(spec["S"]))
            self.arm = arm
            if spec["graph"]:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(SEED + 2911)
                    self.geometry_projection = torch.nn.Linear(geometry_width, 64)
                    self.relation_projection = torch.nn.Linear(relation_width, 64)
                    self.relation_identity = torch.nn.Parameter(
                        torch.zeros(relation_count, 64))
                    self.relation_query = torch.nn.Linear(64, 64, bias=False)
                    self.relation_output = torch.nn.Linear(64, 64, bias=False)
                    self.q_attribute_identity = torch.nn.Embedding(9, 64)
                    self.q_attribute_value = torch.nn.Linear(2, 64)
                    self.q_attribute_query = torch.nn.Linear(64, 64, bias=False)
                    self.q_attribute_key = torch.nn.Linear(64, 64, bias=False)
                    self.q_attribute_output = torch.nn.Linear(64, 64, bias=False)
                    torch.nn.init.zeros_(self.geometry_projection.weight)
                    torch.nn.init.zeros_(self.geometry_projection.bias)
                    torch.nn.init.zeros_(self.relation_output.weight)
                    torch.nn.init.zeros_(self.q_attribute_output.weight)

        def forward(self, x, graph=None, aux=False):
            # Enforce ablations inside forward too: no path can consume Q/S.
            if not spec["Q"] or not spec["S"]:
                x = x.clone()
                if not spec["Q"]:
                    x[:, 189:207] = 0
                if not spec["S"]:
                    x[:, 207:214] = 0
            if not spec["graph"]:
                return super().forward(x, aux=aux)
            assert graph is not None
            # Unknown attributes are absent evidence in every graph path,
            # including the inherited MLP and broadcast context.
            raw_availability = x * self.scale + self.mean
            x = x.clone()
            if spec["Q"]:
                q_observed = raw_availability[:, 198:207] >= .5
                zero_q = -self.mean[189:198] / self.scale[189:198]
                x[:, 189:198] = torch.where(q_observed, x[:, 189:198], zero_q)
            if spec["S"]:
                s_observed = raw_availability[:, 213:214] >= .5
                zero_s = -self.mean[207:213] / self.scale[207:213]
                x[:, 207:213] = torch.where(s_observed, x[:, 207:213], zero_s)
            geometry, relations, relation_mask = graph
            assert geometry.shape == (len(x), 7, geometry_width)
            assert relations.shape == (len(x), 7, relation_count, relation_width)
            assert relation_mask.shape == (len(x), 7, relation_count)
            hidden = self.base[:-1](x)
            logits = self.base[-1](hidden)
            raw = x * self.scale + self.mean
            # Availability flags are binary before normalization. Reconstructing
            # raw floats can leave tiny positive roundoff at an original zero.
            q = x[:, 189:207] * (raw[:, 198:207] >= .5).any(1)[:, None]
            s = x[:, 207:214] * (raw[:, 213] >= .5)[:, None]
            pp = x[:, 214:263].reshape(-1, 7, 7).transpose(1, 2)
            known = x[:, 263:312].reshape(-1, 7, 7).transpose(1, 2)
            rawp = raw[:, 214:263].reshape(-1, 7, 7)
            paired = torch.stack((rawp[:, 5] - rawp[:, 3],
                                  rawp[:, 6] - rawp[:, 4]), -1)
            if not spec["S"]:
                paired = torch.zeros_like(paired)
            context = torch.cat((x[:, 168:189], q, s, x[:, 312:313]), 1)
            context = context[:, None, :].expand(-1, 7, -1)
            identity = torch.eye(7, device=x.device)[None].expand(len(x), -1, -1)
            token = self.token(torch.cat((
                x[:, :168].reshape(-1, 7, 24), pp, known, context, paired, identity,
            ), 2))
            token = token + self.geometry_projection(geometry)
            if spec["Q"]:
                values, known_q = raw[:, 189:198], raw[:, 198:207]
                q_valid = known_q >= .5
                content = torch.stack((values * q_valid, q_valid.to(values.dtype)), -1)
                attribute = (self.q_attribute_value(content)
                             + self.q_attribute_identity(
                                 torch.arange(9, device=x.device))[None])
                score = torch.matmul(
                    self.q_attribute_query(token),
                    self.q_attribute_key(attribute).transpose(1, 2),
                ) / 8.
                q_weights = score.masked_fill(~q_valid[:, None], -1e9).softmax(-1)
                q_weights = q_weights * q_valid[:, None]
                q_weights = q_weights / q_weights.sum(-1, keepdim=True).clamp_min(1e-8)
                token = token + self.q_attribute_output(torch.matmul(q_weights, attribute))
            if spec["edges"]:
                messages = self.relation_projection(relations) + self.relation_identity
                score = (messages * self.relation_query(token)[:, :, None]).sum(-1) / 8.
                valid = relation_mask > 0
                weights = score.masked_fill(~valid, -1e9).softmax(-1)
                weights = weights * valid
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
                token = token + self.relation_output((messages * weights[..., None]).sum(2))
            attended, _ = self.attention(token, token, token, need_weights=False)
            token = self.norm(token + attended)
            logits = logits + self.residual(torch.cat((token.mean(1), token.max(1).values), 1))
            if aux:
                assert spec["S"], "S-off must not execute the SAR auxiliary head"
                return logits, self.sar_effect(token[:, 2] - token[:, 1]).squeeze(1)
            return logits

    return ObjectGraphNet()


def shared_parameter_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if name.startswith(("geometry_", "relation_", "q_attribute_")):
            continue
        a = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + b"\0")
        digest.update(str(a.dtype).encode() + b"\0")
        digest.update(np.asarray(a.shape, dtype="<i8").tobytes())
        digest.update(a.tobytes())
    return digest.hexdigest()


def active_parameter_count(model, arm):
    spec = arm_spec(arm)
    return sum(p.numel() for name, p in model.named_parameters()
               if not ((name.startswith("sar_effect.") and not spec["S"])
                       or (name.startswith("q_attribute_") and not spec["Q"])
                       or (name.startswith("relation_") and not spec["edges"])))


def loss_value(model, batch, phase, fixed, number_valid, torch, s_enabled):
    """Original CE + 0.1 SAR effect MSE + 0.5 candidate soft IoU."""
    x, target, counts, weight, labels, graph = batch
    if s_enabled:
        logits, effect = model(x, graph=graph, aux=True)
        desired = (target.gather(1, labels[:, 2:3]).squeeze(1)
                   - target.gather(1, labels[:, 1:2]).squeeze(1))
        auxiliary = ((effect - desired).square() * weight).mean()
    else:
        logits = model(x, graph=graph, aux=False)
        auxiliary = logits.new_zeros(())
    ce = (-(target * logits.log_softmax(1)).sum(1) * weight).mean()
    soft = logits.new_zeros(())
    if phase == "CE_softIoU":
        duplicates = (labels[:, :, None] == labels[:, None, :]).sum(2)
        actions = (logits.gather(1, labels) - duplicates.float().log()).softmax(1)
        probability = torch.zeros_like(logits).scatter_add_(1, labels, actions)
        scale = logits.new_tensor(float(number_valid) / len(x))
        matrix = fixed + (counts.T @ probability) * scale
        diagonal = matrix.diag()
        soft = 1 - (diagonal / (matrix.sum(0) + matrix.sum(1) - diagonal).clamp_min(1)).mean()
    return ce + .1 * auxiliary + .5 * soft, ce.detach(), auxiliary.detach(), soft.detach()
