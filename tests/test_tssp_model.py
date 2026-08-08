from types import SimpleNamespace

import numpy as np
import pytest
from gurobipy import GRB

from stochastic_energy_dispatch import tssp_model


def _system_data(T=2):
    return SimpleNamespace(
        A=np.eye(2),
        R_prime=np.eye(2),
        X_prime=np.eye(2),
        P_load=np.ones((2, T)),
        PD_base=np.ones((2, 1)),
    )


def _ders_data():
    return SimpleNamespace(
        G_node=[2],
        PV_node=[],
        ESS_node=[],
        R_node=[3],
    )


def _sample_info(T=2):
    return {
        "empirical_samples": np.ones((2, 2 * T)),
        "num_empirical_samples": 2,
    }


def test_validate_model_inputs_accepts_valid_inputs():
    tssp_model._validate_model_inputs(
        _system_data(),
        _ders_data(),
        _sample_info(),
        num_nodes=3,
        T=2,
    )


def test_validate_model_inputs_rejects_wrong_load_shape():
    system_data = _system_data()
    system_data.P_load = np.ones((2, 1))

    with pytest.raises(ValueError, match="P_load must have shape"):
        tssp_model._validate_model_inputs(
            system_data,
            _ders_data(),
            _sample_info(),
            num_nodes=3,
            T=2,
        )


def test_validate_model_inputs_rejects_wrong_scenario_shape():
    sample_info = _sample_info()
    sample_info["empirical_samples"] = np.ones((2, 3))

    with pytest.raises(
        ValueError,
        match="Empirical scenarios must have shape",
    ):
        tssp_model._validate_model_inputs(
            _system_data(),
            _ders_data(),
            sample_info,
            num_nodes=3,
            T=2,
        )


def test_validate_model_inputs_rejects_non_finite_scenario():
    sample_info = _sample_info()
    sample_info["empirical_samples"][0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        tssp_model._validate_model_inputs(
            _system_data(),
            _ders_data(),
            sample_info,
            num_nodes=3,
            T=2,
        )


def test_validate_model_inputs_rejects_slack_device():
    ders_data = _ders_data()
    ders_data.G_node = [1]

    with pytest.raises(
        ValueError,
        match="invalid or slack node IDs",
    ):
        tssp_model._validate_model_inputs(
            _system_data(),
            ders_data,
            _sample_info(),
            num_nodes=3,
            T=2,
        )


def test_validate_model_inputs_rejects_multiple_device_types_same_node():
    ders_data = _ders_data()
    ders_data.PV_node = [2]

    with pytest.raises(
        ValueError,
        match="contains both",
    ):
        tssp_model._validate_model_inputs(
            _system_data(),
            ders_data,
            _sample_info(),
            num_nodes=3,
            T=2,
        )


def test_raise_if_not_optimal_accepts_optimal_status():
    model = SimpleNamespace(status=GRB.OPTIMAL)

    tssp_model._raise_if_not_optimal(model)


def test_raise_if_not_optimal_reports_infeasible():
    model = SimpleNamespace(status=GRB.INFEASIBLE)

    with pytest.raises(
        RuntimeError,
        match="INFEASIBLE",
    ):
        tssp_model._raise_if_not_optimal(model)
