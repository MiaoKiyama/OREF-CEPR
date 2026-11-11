"""Original C and dynamic I, with separately fitted raw/C starting routes."""
import numpy as np

import native24_original82_policy as algorithm
import native24_original82_reference as reference
from training_common import raw_posterior_route

ROUTES = ("raw", "C", "I", "CI")


def choose_iterations(hist, labels, fixed_expected, initial, counts, fixed_truth):
    """Select 0..12 on this fitting pool; posterior/algorithm never change."""
    cm = reference.score_route(fixed_truth, counts, labels, initial)
    best = [0, float(100 * reference.miou(cm))]
    trace = []

    def observer(step, indices, old, new, row):
        cm[:] += (reference.cm_from_groups(counts[indices], labels[indices], new)
                  - reference.cm_from_groups(counts[indices], labels[indices], old))
        value = float(100 * reference.miou(cm))
        trace.append(dict(row, fitting_miou_percent=value))
        if value > best[1] + 1e-10:
            best[:] = [int(step), value]

    algorithm.iterate_I(hist, labels, fixed_expected, initial, True, 12, observer)
    return dict(dynamic=True, iterations=best[0],
                fitting_miou_percent=best[1], trace=trace)


def fit_four_routes(posterior, pack, prior, include_I=True):
    """C and both I stop counts use exactly the same permitted fitting cohort."""
    c_policy, c_route = algorithm.fit_calibration(posterior, pack, prior)
    raw = raw_posterior_route(posterior, pack["labels"])
    policy = dict(C=c_policy, prior=prior,
                  I_definition="original_dynamic", I_max_iterations=12,
                  fitting_pixels=int(pack["pixels"]))
    routes = dict(raw=raw, C=c_route)
    if include_I:
        hist, fixed = algorithm.expected_inputs(
            posterior, pack["labels"], pack["mass"], pack["fixed_obs"],
            pack["pixels"], prior,
        )
        policy["I"] = {}
        for key, initial in (("I", raw), ("CI", c_route)):
            selected = choose_iterations(
                hist, pack["labels"], fixed, initial, pack["counts"], pack["fixed"],
            )
            policy["I"][key] = selected
            routes[key], _ = algorithm.iterate_I(
                hist, pack["labels"], fixed, initial, True, selected["iterations"],
            )
            actual = 100 * reference.miou(reference.score_route(
                pack["fixed"], pack["counts"], pack["labels"], routes[key],
            ))
            assert abs(actual - selected["fitting_miou_percent"]) < 1e-8
    policy["C_fitting_miou_percent"] = float(c_policy["training_class_miou"])
    return policy, routes


def frozen_four_routes(posterior, observed, policy):
    """No truth/count fields are accepted or used by frozen route generation."""
    forbidden = {"counts", "fixed", "baselines", "targets"}
    assert not forbidden.intersection(observed)
    labels = observed["labels"]
    raw = raw_posterior_route(posterior, labels)
    hist, fixed = algorithm.expected_inputs(
        posterior, labels, observed["mass"], observed["fixed_obs"],
        observed["pixels"], policy["prior"],
    )
    baseline = np.full(len(labels), 2, np.uint8)
    gains = reference.expected_gains(fixed + reference.expected_cm(hist, labels, baseline),
                                     hist, labels, baseline)
    calibrated = algorithm.apply_thresholds(gains, labels, policy["C"])
    routes, trace = dict(raw=raw, C=calibrated), {}
    for key, initial in (("I", raw), ("CI", calibrated)):
        chosen = policy["I"][key]
        assert chosen["dynamic"] is True and 0 <= int(chosen["iterations"]) <= 12
        routes[key], trace[key] = algorithm.iterate_I(
            hist, labels, fixed, initial, True, int(chosen["iterations"]),
        )
    return routes, trace
