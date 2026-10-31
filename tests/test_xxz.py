"""TEBD against full diagonalisation on a six-site chain."""

import numpy as np
import pytest

from kpz.controls import load, setting
from kpz.run import evolve as evolve_control
from kpz.xxz import evolve, exact_wall


@pytest.mark.parametrize("eta", [.22, 1.])
def test_tebd_matches_exact_evolution(eta):
    times = np.arange(0., 1.001, .25)
    exact = exact_wall(6, times, eta)
    coarse = evolve(6, eta, .05, 64, times)
    fine = evolve(6, eta, .025, 64, times)
    coarse_error = np.max(np.abs(coarse - exact))
    fine_error = np.max(np.abs(fine - exact))
    assert np.max(np.abs(fine[0] - exact[0])) < 1e-12
    assert fine_error < 5e-5
    assert fine_error < .4 * coarse_error


def test_magnetization_is_conserved():
    times = np.arange(0., 1.001, .5)
    sz = evolve(8, .22, .05, 64, times)
    np.testing.assert_allclose(sz.sum(axis=1), sz[0].sum(), atol=1e-10)


def test_resumed_run_matches_uninterrupted_run(tmp_path):
    config = setting(.22, 8, .1, 32)
    evolve_control(config, 1., tmp_path / "resumed")
    resumed = evolve_control(config, 2., tmp_path / "resumed")
    straight = evolve_control(config, 2., tmp_path / "straight")
    assert [row["u"] for row in resumed["rows"]] == [0., .5, 1., 1.5, 2.]
    np.testing.assert_allclose([row["sz"] for row in resumed["rows"]],
                               [row["sz"] for row in straight["rows"]], atol=1e-12)
    saved = load(config, tmp_path / "resumed")
    np.testing.assert_allclose(saved["sz"], [row["sz"] for row in straight["rows"]], atol=1e-12)
