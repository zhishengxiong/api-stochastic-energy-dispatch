import pytest
from pydantic import ValidationError

from stochastic_energy_dispatch.api_schemas import TSSPRunRequest


def test_tssp_run_request_uses_defaults():
    request = TSSPRunRequest()

    assert request.T == 4
    assert request.num_samples == 5


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("T", 0),
        ("num_samples", 0),
        ("num_samples", 501),
    ],
)
def test_tssp_run_request_rejects_non_positive_values(
    field_name,
    invalid_value,
):
    kwargs = {
        "T": 4,
        "num_samples": 5,
    }
    kwargs[field_name] = invalid_value

    with pytest.raises(ValidationError):
        TSSPRunRequest(**kwargs)
