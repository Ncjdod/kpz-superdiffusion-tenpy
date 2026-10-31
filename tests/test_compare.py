"""Observable conventions and regression of the published comparison."""

import numpy as np

from kpz.compare import build, crop_roi, load_measurements, roi_transfer
from kpz.controls import controls, label


def test_measured_profiles_match_transfer():
    measured = load_measurements()
    assert measured["delta_m"].shape == (14, 46)
    assert measured["ms"][-1] == 720.
    np.testing.assert_array_equal(measured["delta_m_error"][0], np.zeros(46))


def test_field_of_view():
    np.testing.assert_array_equal(crop_roi(np.arange(50)), np.arange(2, 48))
    sz = np.zeros((2, 50))
    sz[1, 24], sz[1, 25] = -.2, .2
    np.testing.assert_allclose(roi_transfer(sz), [0, .4])
    sz = np.zeros((2, 50))
    sz[1, 0], sz[1, 49] = -.2, .2
    np.testing.assert_allclose(roi_transfer(sz), [0, 0])


def test_control_labels():
    assert [label(c) for c in controls().values()] == [
        "eta0.22_L50_dt0.1_chi256", "eta0.22_L50_dt0.05_chi256", "eta0.22_L50_dt0.1_chi512"]


def test_saved_comparison():
    summary = build()[5]
    assert summary["trusted_u"] == 15.
    assert summary["measurements_compared"] == 6
    assert summary["within_one_error"] == 5
    np.testing.assert_allclose(summary["rmse_P"], 0.06863685755136664, rtol=1e-10)
    np.testing.assert_allclose(summary["spatial_rmse"], 0.035961911134881494, rtol=1e-10)
    np.testing.assert_allclose(summary["max_dt_change_P"], 0.00020176523879533725, rtol=1e-10)
    np.testing.assert_allclose(summary["max_chi_change_P"], 0.00029764964221357104, rtol=1e-10)
