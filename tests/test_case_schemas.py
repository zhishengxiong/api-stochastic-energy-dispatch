import pytest

from stochastic_energy_dispatch.case_schemas import TSSPRunConfig


def test_tssp_run_config_accepts_valid_values():
    config = TSSPRunConfig(
        T=4,
        num_samples=5,
    )

    assert config.T == 4
    assert config.num_samples == 5


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("T", 0),
        ("num_samples", 0),
        ("T", -1),
        ("num_samples", -1),
        ("num_samples", 501),
    ],
)
def test_tssp_run_config_rejects_non_positive_values(
    field_name,
    invalid_value,
):
    kwargs = {
        "T": 4,
        "num_samples": 5,
    }
    kwargs[field_name] = invalid_value

    with pytest.raises(ValueError):
        TSSPRunConfig(**kwargs)
