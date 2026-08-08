import numpy as np
import pandas as pd
import pytest

from stochastic_energy_dispatch import samples_preprocessing


def _complete_sample_dataframe():
    return pd.DataFrame(
        {
            "sample_id": [1, 1, 2, 2],
            "time_step": [0, 1, 0, 1],
            "pv_scale": [0.2, 0.3, 0.4, 0.5],
            "load_scale": [0.9, 1.0, 1.1, 1.2],
        }
    )


def test_read_historical_scenarios_builds_expected_matrix(monkeypatch):
    raw_scenarios = _complete_sample_dataframe()

    monkeypatch.setattr(
        samples_preprocessing.repository,
        "fetch_historical_scenarios",
        lambda connection, T, num_samples: raw_scenarios,
    )

    samples = samples_preprocessing.read_historical_scenarios(
        connection=object(),
        T=2,
        num_samples=2,
    )

    expected = np.array(
        [
            [0.2, 0.3, 0.9, 1.0],
            [0.4, 0.5, 1.1, 1.2],
        ]
    )

    np.testing.assert_allclose(samples, expected)


def test_read_historical_scenarios_rejects_missing_time_step(
    monkeypatch,
):
    raw_scenarios = pd.DataFrame(
        {
            "sample_id": [1, 1, 2],
            "time_step": [0, 1, 0],
            "pv_scale": [0.2, 0.3, 0.4],
            "load_scale": [0.9, 1.0, 1.1],
        }
    )

    monkeypatch.setattr(
        samples_preprocessing.repository,
        "fetch_historical_scenarios",
        lambda connection, T, num_samples: raw_scenarios,
    )

    with pytest.raises(ValueError):
        samples_preprocessing.read_historical_scenarios(
            connection=object(),
            T=2,
            num_samples=2,
        )
