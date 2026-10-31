"""Numerical controls of one physical setting, their saved trajectories and convergence gates."""

import json

import numpy as np

from kpz import DELTA, J_HZ, ROOT

TRAJECTORIES = ROOT / "results" / "trajectories"
LIMITS = {"dt_sz": .002, "chi_sz": .002, "dt_p": .015, "chi_p": .015,
          "magnetization_drift": 1.e-7, "edge_change": .001, "discarded": .001}


def setting(eta, length, dt, chi):
    """One physical preparation with one numerical resolution."""
    return {"eta": float(eta), "length": length, "dt": dt, "chi": chi, "delta": DELTA}


def controls(eta=.22, length=50, dt=.1, chi=256):
    """Baseline plus one halved time step and one doubled bond dimension."""
    return {"baseline": setting(eta, length, dt, chi),
            "dt": setting(eta, length, dt / 2, chi),
            "chi": setting(eta, length, dt, 2 * chi)}


def label(config):
    """File stem of a setting."""
    return f"eta{config['eta']:g}_L{config['length']}_dt{config['dt']:g}_chi{config['chi']}"


def transfer(sz):
    """Polarization transferred across the wall, P(t) = sum_right dSz - sum_left dSz."""
    change = sz - sz[0]
    middle = sz.shape[-1] // 2
    return change[:, middle:].sum(axis=1) - change[:, :middle].sum(axis=1)


def to_ms(u):
    """Convert Jt/hbar to milliseconds at J/h = 10.2 Hz."""
    return np.asarray(u) * 1000 / (2 * np.pi * J_HZ)


def load(config, directory=TRAJECTORIES):
    """Saved trajectory of a setting; refuses files written for different parameters."""
    with np.load(directory / (label(config) + ".npz")) as saved:
        run = dict(saved)
    if json.loads(str(run["config"])) != config:
        raise ValueError(f"{label(config)}.npz was written for a different setting.")
    return run


def assess(runs):
    """Per-time convergence diagnostics on the interval shared by all three controls."""
    count = min(len(run["u"]) for run in runs.values())
    base, fine, large = [{key: runs[name][key][:count] for key in ("u", "sz", "discarded")}
                         for name in ("baseline", "dt", "chi")]
    base_p, fine_p, large_p = [transfer(run["sz"]) for run in (base, fine, large)]
    diagnostics = {
        "u": large["u"],
        "dt_sz": np.max(np.abs(base["sz"] - fine["sz"]), axis=1),
        "chi_sz": np.max(np.abs(base["sz"] - large["sz"]), axis=1),
        "dt_p": np.abs(base_p - fine_p),
        "chi_p": np.abs(base_p - large_p),
        "magnetization_drift": np.abs(large["sz"].sum(axis=1) - large["sz"][0].sum()),
        "edge_change": np.max(np.abs(large["sz"][:, [0, 1, -2, -1]] - large["sz"][0, [0, 1, -2, -1]]), axis=1),
        "discarded": large["discarded"],
    }
    diagnostics["passed"] = np.all([diagnostics[key] < limit for key, limit in LIMITS.items()], axis=0)
    return diagnostics
