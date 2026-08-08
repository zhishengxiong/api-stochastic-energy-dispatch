import numpy as np

from stochastic_energy_dispatch import input_repository as repository


def _validate_sample_table(sample_table, T, num_samples, sample_name):
    if sample_table.shape != (num_samples, T):
        raise ValueError(
            f"{sample_name} samples must have shape "
            f"({num_samples}, {T}), but found {sample_table.shape}."
        )

    expected_time_steps = list(range(T))
    actual_time_steps = sample_table.columns.astype(int).tolist()

    if actual_time_steps != expected_time_steps:
        raise ValueError(
            f"{sample_name} samples must contain time steps "
            f"{expected_time_steps}, but found {actual_time_steps}."
        )

    values = sample_table.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{sample_name} samples contain missing or non-finite values."
        )

    return values


def read_historical_scenarios(connection, T, num_samples):
    raw_scenarios = repository.fetch_historical_scenarios(
        connection,
        T,
        num_samples,
    )

    if raw_scenarios.empty:
        raise ValueError("No historical scenarios were found.")

    pv_samples = (
        raw_scenarios.pivot(
            index="sample_id",
            columns="time_step",
            values="pv_scale",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    load_samples = (
        raw_scenarios.pivot(
            index="sample_id",
            columns="time_step",
            values="load_scale",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    if not pv_samples.index.equals(load_samples.index):
        raise ValueError(
            "PV and load samples do not contain the same sample IDs."
        )

    pv_values = _validate_sample_table(
        pv_samples,
        T,
        num_samples,
        "PV",
    )
    load_values = _validate_sample_table(
        load_samples,
        T,
        num_samples,
        "Load",
    )

    return np.hstack([pv_values, load_values])


def build_samples_info(connection, T, num_samples):
    """Build empirical-sample data required by the TSSP model."""
    empirical_samples = read_historical_scenarios(
        connection,
        T,
        num_samples,
    )

    return {
        "T": T,
        "empirical_samples": empirical_samples,
        "num_empirical_samples": empirical_samples.shape[0],
    }
