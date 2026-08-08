import numpy as np
import pandas as pd
import pytest

from stochastic_energy_dispatch import ders_data_preprocessing as ders


def test_expand_to_time_horizon():
    values = pd.Series([10.0, 20.0])

    result = ders.expand_to_time_horizon(values, T=3)

    expected = np.array(
        [
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
        ]
    )

    np.testing.assert_allclose(result, expected)


def test_build_price_profile():
    raw_prices = pd.DataFrame(
        {
            "time_step": [0, 1, 2],
            "price": [20.123, 21.456, 22.789],
        }
    )

    result = ders.build_price_profile(raw_prices, T=3)

    np.testing.assert_allclose(
        result,
        np.array([20.12, 21.46, 22.79]),
    )


def test_build_price_profile_rejects_missing_time_step():
    raw_prices = pd.DataFrame(
        {
            "time_step": [0, 2],
            "price": [20.0, 22.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="must contain exactly time steps",
    ):
        ders.build_price_profile(raw_prices, T=3)


def test_build_generator_data_preserves_database_ids():
    raw_generators = pd.DataFrame(
        {
            "generator_id": [10, 20],
            "node_id": [2, 5],
            "p_max": [100.0, 200.0],
            "p_min": [10.0, 20.0],
            "q_max": [50.0, 60.0],
            "q_min": [-50.0, -60.0],
            "ramp_up": [30.0, 40.0],
            "ramp_down": [30.0, 40.0],
            "generation_cost": [10.0, 12.0],
            "startup_cost": [5.0, 6.0],
        }
    )

    result = ders.build_generator_data(
        raw_generators,
        T=2,
    )

    generator_ids = result[0]
    generator_nodes = result[1]

    assert generator_ids == [10, 20]
    assert generator_nodes == [2, 5]


def test_build_pv_data_preserves_ids_and_scales_capacity():
    raw_pv_units = pd.DataFrame(
        {
            "pv_id": [7],
            "node_id": [4],
            "capacity": [100.0],
        }
    )

    raw_pv_forecast = pd.DataFrame(
        {
            "time_step": [0, 1],
            "pv_scale": [0.2, 0.5],
        }
    )

    pv_ids, pv_nodes, pv, pv_cap = ders.build_pv_data(
        raw_pv_units,
        raw_pv_forecast,
        T=2,
    )

    assert pv_ids == [7]
    assert pv_nodes == [4]
    np.testing.assert_allclose(pv_cap, np.array([[100.0]]))
    np.testing.assert_allclose(
        pv,
        np.array([[20.0, 50.0]]),
    )
