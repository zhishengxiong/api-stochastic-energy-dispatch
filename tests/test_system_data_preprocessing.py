import numpy as np
import pandas as pd
import pytest

from stochastic_energy_dispatch.system_data_preprocessing import (
    _validate_network_data,
    build_load_profiles,
    build_reduced_incidence_matrix,
)


def _build_nodes():
    return pd.DataFrame(
        {
            "node_id": [1, 2, 3],
            "is_slack": [True, False, False],
            "pd_base": [0.0, 10.0, 20.0],
        }
    )


def _build_lines():
    return pd.DataFrame(
        {
            "line_id": [1, 2],
            "from_node": [1, 2],
            "to_node": [2, 3],
            "resistance": [0.1, 0.2],
            "reactance": [0.05, 0.08],
        }
    )


def test_validate_network_data_accepts_radial_network():
    raw_nodes = _build_nodes()
    raw_lines = _build_lines()

    _validate_network_data(
        raw_nodes,
        raw_lines,
        num_nodes=3,
    )


def test_build_reduced_incidence_matrix():
    raw_lines = _build_lines()

    matrix = build_reduced_incidence_matrix(
        raw_lines,
        num_nodes=3,
    )

    expected = np.array(
        [
            [-1.0, 0.0],
            [1.0, -1.0],
        ]
    )

    np.testing.assert_array_equal(matrix, expected)


def test_build_load_profiles():
    raw_nodes = _build_nodes()

    raw_demand = pd.DataFrame(
        {
            0: [1.0, 0.5],
            1: [0.8, 0.6],
        },
        index=[2, 3],
    )

    p_load, pd_base = build_load_profiles(
        raw_nodes,
        raw_demand,
        T=2,
    )

    expected_pd_base = np.array(
        [
            [10.0],
            [20.0],
        ]
    )

    expected_p_load = np.array(
        [
            [-10.0, -8.0],
            [-10.0, -12.0],
        ]
    )

    np.testing.assert_allclose(pd_base, expected_pd_base)
    np.testing.assert_allclose(p_load, expected_p_load)


def test_build_load_profiles_rejects_wrong_node_ids():
    raw_nodes = _build_nodes()

    raw_demand = pd.DataFrame(
        {
            0: [1.0, 0.5],
            1: [0.8, 0.6],
        },
        index=[2, 4],
    )

    with pytest.raises(
        ValueError,
        match="Demand-profile node IDs do not match",
    ):
        build_load_profiles(
            raw_nodes,
            raw_demand,
            T=2,
        )
