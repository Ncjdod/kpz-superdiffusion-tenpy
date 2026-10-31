"""Compare the 61/39 domain wall with the measured polarization transfer and spin profiles."""

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kpz import DELTA, J_HZ, ROOT
from kpz.controls import LIMITS, assess, controls, load, to_ms

ETA = .22
DATA = ROOT / "data"
OUTPUT = ROOT / "results" / "comparison"


def load_measurements():
    """Transfer P(t) and changes of the 46-site spin-imbalance profile from the published data."""
    transfer = json.loads((DATA / "8AC_pol-transfer.json").read_text(encoding="utf-8"))
    profiles = json.loads((DATA / "4A_experiment-unmagnetized.json").read_text(encoding="utf-8"))
    index = list(transfer["purity"]).index(ETA)
    seconds = np.asarray(transfer["time"], dtype=float)
    p = np.asarray(transfer["pol_transfer"][index], dtype=float)
    profile = np.asarray(profiles["profile"], dtype=float)
    profile_error = np.asarray(profiles["profile_err"], dtype=float)
    if profile.shape != (len(seconds), 46) or not np.allclose(seconds, profiles["time"]):
        raise ValueError("Unexpected shape or times of the measured profiles.")
    delta_m = profile - profile[0]
    delta_m_error = profile_error + profile_error[0]
    delta_m_error[0] = 0.
    sign = np.r_[-np.ones(23), np.ones(23)]
    if np.max(np.abs(.5 * delta_m @ sign - p)) > 1e-10:
        raise RuntimeError("Measured profiles do not integrate to the measured transfer.")
    return {"ms": 1000 * seconds, "u": 2 * np.pi * J_HZ * seconds, "p": p,
            "error": np.asarray(transfer["pol_transfer_err"][index], dtype=float),
            "delta_m": delta_m, "delta_m_error": delta_m_error, "sites": np.arange(46) - 22.5}


def crop_roi(sz):
    """Central 46 sites of the 50-site chain, the experimental field of view."""
    if sz.shape[-1] != 50:
        raise ValueError("The comparison needs the 50-site chain.")
    return sz[..., 2:48]


def roi_transfer(sz):
    """Polarization transfer inside the field of view."""
    change = crop_roi(sz - sz[0])
    return change[:, 23:].sum(axis=1) - change[:, :23].sum(axis=1)


def build(dt=.1, chi=256):
    """Residuals at measurement times inside the interval where every convergence gate passes."""
    measured = load_measurements()
    runs = {name: load(config) for name, config in controls(ETA, dt=dt, chi=chi).items()}
    checks = assess(runs)
    count = len(checks["u"])
    times = checks["u"]
    region = {name: roi_transfer(run["sz"][:count]) for name, run in runs.items()}
    checks["dt_p_roi"] = np.abs(region["baseline"] - region["dt"])
    checks["chi_p_roi"] = np.abs(region["baseline"] - region["chi"])
    passing = checks["passed"] & (checks["dt_p_roi"] < LIMITS["dt_p"]) & (checks["chi_p_roi"] < LIMITS["chi_p"])
    failed = np.flatnonzero(~passing)
    last = int(failed[0] - 1) if len(failed) else count - 1
    if last < 1:
        raise RuntimeError("No nonzero time interval passes the convergence gates.")
    trusted_u = float(times[last])
    finest = runs["chi"]
    rows, spatial = [], []
    for k in np.flatnonzero((measured["u"] > 0) & (measured["u"] <= trusted_u + 1e-8)):
        u = measured["u"][k]
        prediction = float(np.interp(u, times, region["chi"]))
        rows.append({"time_ms": float(measured["ms"][k]), "experiment_P": float(measured["p"][k]),
                     "experiment_error": float(measured["error"][k]), "simulation_P": prediction,
                     "residual": float(measured["p"][k] - prediction),
                     "dt_change": float(np.interp(u, times, checks["dt_p_roi"])),
                     "chi_change": float(np.interp(u, times, checks["chi_p_roi"]))})
        profile = np.array([np.interp(u, finest["u"], finest["sz"][:, i]) for i in range(50)])
        simulated = 2 * crop_roi(profile - finest["sz"][0])
        spatial.append({"time_ms": rows[-1]["time_ms"], "simulation": simulated,
                        "experiment": measured["delta_m"][k], "error": measured["delta_m_error"][k],
                        "residual": measured["delta_m"][k] - simulated})
    residual = np.array([row["residual"] for row in rows])
    spatial_residual = np.array([row["residual"] for row in spatial])
    summary = {"eta": ETA, "length": 50, "measured_sites": 46, "delta": DELTA, "J_over_h_Hz": J_HZ,
               "baseline_dt": dt, "baseline_chi": chi, "trusted_u": trusted_u,
               "trusted_time_ms": float(to_ms(trusted_u)),
               "measurements_compared": len(rows),
               "within_one_error": int(sum(abs(row["residual"]) <= row["experiment_error"] for row in rows)),
               "rmse_P": float(np.sqrt(np.mean(residual ** 2))),
               "mean_residual_P": float(np.mean(residual)),
               "max_abs_residual_P": float(np.max(np.abs(residual))),
               "spatial_rmse": float(np.sqrt(np.mean(spatial_residual ** 2))),
               "max_dt_change_P": float(checks["dt_p_roi"][:last + 1].max()),
               "max_chi_change_P": float(checks["chi_p_roi"][:last + 1].max()),
               "comparisons": rows}
    return measured, finest, checks, times[:last + 1], region["chi"][:last + 1], summary, spatial


def transfer_figure(measured, times, prediction, summary):
    """Measured and simulated transfer with residuals."""
    ms = to_ms(times)
    shown = measured["ms"] <= ms[-1] + 1e-7
    rows = summary["comparisons"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]},
                             layout="constrained")
    axes[0].plot(ms, prediction, color="#087e79", lw=2, label="TeNPy, spin-only XXZ")
    axes[0].errorbar(measured["ms"][shown], measured["p"][shown], yerr=measured["error"][shown],
                     fmt="o", capsize=3, color="#c56127", label="Experiment")
    axes[0].set(ylabel="Transferred polarization P")
    axes[0].legend()
    axes[1].axhline(0, color="#454b50", lw=1)
    axes[1].errorbar([r["time_ms"] for r in rows], [r["residual"] for r in rows],
                     yerr=[r["experiment_error"] for r in rows], fmt="o", capsize=3, color="#c56127")
    axes[1].set(xlabel="Time (ms)", ylabel="Experiment - simulation", xlim=(0, ms[-1] * 1.03))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("61/39 domain wall, L = 50, Delta = 0.986")
    fig.savefig(OUTPUT / "transfer.png", dpi=160)
    plt.close(fig)


def evolution_figure(run, trusted_u):
    """Simulated up-spin probability across the chain."""
    shown = run["u"] <= trusted_u + 1e-8
    ms = to_ms(run["u"][shown])
    up = .5 + run["sz"][shown]
    sites = np.arange(50) - 24.5
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    mesh = axes[0].pcolormesh(sites, ms, up, shading="nearest", cmap="RdBu_r", vmin=.39, vmax=.61)
    axes[0].set(xlabel="Site", ylabel="Time (ms)")
    fig.colorbar(mesh, ax=axes[0], label="Up-spin probability")
    for k in np.unique(np.linspace(0, len(ms) - 1, 4).round().astype(int)):
        axes[1].plot(sites, up[k], ".-", ms=3, label=f"{ms[k]:.0f} ms")
    axes[1].set(xlabel="Site", ylabel="Up-spin probability", ylim=(.37, .63))
    axes[1].legend()
    axes[1].grid(alpha=.2)
    fig.savefig(OUTPUT / "evolution.png", dpi=160)
    plt.close(fig)


def spatial_figure(measured, spatial):
    """Measured and simulated changes of the spin-imbalance profile with residuals."""
    chosen = np.unique(np.linspace(0, len(spatial) - 1, min(3, len(spatial))).round().astype(int))
    fig, axes = plt.subplots(2, len(chosen), figsize=(4.4 * len(chosen), 6.5), squeeze=False, layout="constrained")
    for column, index in enumerate(chosen):
        row = spatial[index]
        axes[0, column].plot(measured["sites"], row["simulation"], color="#087e79", lw=2, label="TeNPy")
        axes[0, column].errorbar(measured["sites"], row["experiment"], yerr=row["error"],
                                 fmt="o", ms=3, capsize=2, color="#c56127", label="Experiment")
        axes[0, column].set(title=f"{row['time_ms']:.0f} ms", ylabel="Change in spin imbalance")
        axes[1, column].axhline(0, color="#454b50", lw=1)
        axes[1, column].errorbar(measured["sites"], row["residual"], yerr=row["error"],
                                 fmt="o", ms=3, capsize=2, color="#c56127")
        axes[1, column].set(xlabel="Site", ylabel="Experiment - simulation")
        for ax in axes[:, column]:
            ax.set_xlim(-23.5, 23.5)
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.savefig(OUTPUT / "profiles.png", dpi=160)
    plt.close(fig)


def write_csv(path, rows):
    """Write dictionaries with a shared header."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=.1, help="baseline time step in units of hbar/J")
    parser.add_argument("--chi", type=int, default=256, help="baseline bond dimension")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False})
    measured, finest, checks, times, prediction, summary, spatial = build(args.dt, args.chi)
    transfer_figure(measured, times, prediction, summary)
    evolution_figure(finest, summary["trusted_u"])
    spatial_figure(measured, spatial)
    write_csv(OUTPUT / "transfer.csv", summary["comparisons"])
    write_csv(OUTPUT / "profiles.csv", [
        {"time_ms": row["time_ms"], "site": float(site), "experiment": float(row["experiment"][i]),
         "simulation": float(row["simulation"][i]), "error": float(row["error"][i]),
         "residual": float(row["residual"][i])}
        for row in spatial for i, site in enumerate(measured["sites"])])
    keys = ["u", *LIMITS, "dt_p_roi", "chi_p_roi", "passed"]
    write_csv(OUTPUT / "convergence.csv", [dict(zip(keys, values)) for values in zip(*(checks[key].tolist() for key in keys))])
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"RMSE {summary['rmse_P']:.5f}, {summary['within_one_error']}/{summary['measurements_compared']} "
          f"within one error, checked to {summary['trusted_time_ms']:.1f} ms")


if __name__ == "__main__":
    main()
