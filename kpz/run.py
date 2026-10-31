"""Evolve one numerical control of the domain wall with resumable checkpoints."""

import argparse
import json
import pickle
import time

import numpy as np

from kpz.controls import TRAJECTORIES, controls, label, transfer
from kpz.xxz import initial_wall, make_engine, make_model, measure

SAMPLE = .5


def save(payload, directory):
    """Write the resumable checkpoint and the portable npz trajectory."""
    stem = label(payload["config"])
    temporary = directory / (stem + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(directory / (stem + ".pkl"))
    arrays = {key: np.asarray([row[key] for row in payload["rows"]]) for key in payload["rows"][0]}
    arrays["transfer"] = transfer(arrays["sz"])
    arrays["config"] = np.array(json.dumps(payload["config"], sort_keys=True))
    arrays["runtime_seconds"] = np.array(payload["runtime_seconds"])
    np.savez_compressed(directory / (stem + ".npz"), **arrays)


def evolve(config, target_u, directory=TRAJECTORIES, minutes=None, max_discarded=.01):
    """Advance in steps of u = 0.5 until the target, the time budget or the truncation ceiling is reached."""
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / (label(config) + ".pkl")
    model = make_model(config["length"], config["delta"])
    steps = round(SAMPLE / config["dt"])
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            payload = pickle.load(handle)
        if payload["config"] != config:
            raise RuntimeError(f"{checkpoint} belongs to a different setting.")
        state = payload["resume"]["engine"]["psi"]
        engine = make_engine(state, model, config["eta"], config["dt"], config["chi"], steps, payload["resume"])
    else:
        state = initial_wall(model, config["eta"])
        engine = make_engine(state, model, config["eta"], config["dt"], config["chi"], steps)
        payload = {"config": config, "rows": [measure(state, model, engine)], "runtime_seconds": 0.}
    deadline = None if minutes is None else time.perf_counter() + 60 * minutes
    started = time.perf_counter()
    while payload["rows"][-1]["u"] < target_u - 1.e-8:
        if deadline is not None and time.perf_counter() > deadline:
            print("Time budget reached.")
            break
        if payload["rows"][-1]["discarded"] > max_discarded:
            print("Discarded weight exceeds the ceiling.")
            break
        engine.run()
        payload["rows"].append(measure(state, model, engine))
        payload["runtime_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        payload["resume"] = {"engine": engine.get_resume_data(), "truncation": engine.trunc_err}
        save(payload, directory)
        print(f"{label(config)}: u = {payload['rows'][-1]['u']:.1f}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", choices=("baseline", "dt", "chi"))
    parser.add_argument("--eta", type=float, default=.22)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--dt", type=float, default=.1, help="baseline time step in units of hbar/J")
    parser.add_argument("--chi", type=int, default=256, help="baseline bond dimension")
    parser.add_argument("--target-u", type=float, default=15.)
    parser.add_argument("--minutes", type=float)
    args = parser.parse_args()
    if args.length % 2 or not 0 < args.eta <= 1:
        parser.error("Use an even chain and 0 < eta <= 1.")
    if not np.isclose(args.target_u / SAMPLE, round(args.target_u / SAMPLE)):
        parser.error("target-u must be a multiple of 0.5.")
    config = controls(args.eta, args.length, args.dt, args.chi)[args.control]
    evolve(config, args.target_u, minutes=args.minutes)


if __name__ == "__main__":
    main()
