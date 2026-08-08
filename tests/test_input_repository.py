from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from stochastic_energy_dispatch import input_repository


class FakeCursor:
    def __init__(self, rows=None, columns=None, one_row=None):
        self.rows = rows or []
        self.description = [SimpleNamespace(name=name) for name in (columns or [])]
        self.one_row = one_row
        self.executed = []

    def execute(self, query, parameters=None):
        self.executed.append((query, parameters))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.one_row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def test_fetch_dataframe_builds_dataframe():
    cursor = FakeCursor(
        rows=[(1, True, 0.0), (2, False, 10.0)],
        columns=["node_id", "is_slack", "pd_base"],
    )
    connection = FakeConnection(cursor)

    result = input_repository._fetch_dataframe(
        connection,
        "SELECT * FROM nodes;",
    )

    expected = pd.DataFrame(
        [
            (1, True, 0.0),
            (2, False, 10.0),
        ],
        columns=["node_id", "is_slack", "pd_base"],
    )

    pd.testing.assert_frame_equal(result, expected)


def test_fetch_demand_forecasts_passes_time_horizon():
    cursor = FakeCursor(
        rows=[],
        columns=["node_id", "time_step", "demand_scale"],
    )
    connection = FakeConnection(cursor)

    input_repository.fetch_demand_forecasts(connection, T=4)

    _, parameters = cursor.executed[0]
    assert parameters == (4,)


def test_fetch_historical_scenarios_passes_sample_limit_and_T():
    cursor = FakeCursor(
        rows=[],
        columns=[
            "sample_id",
            "time_step",
            "pv_scale",
            "load_scale",
        ],
    )
    connection = FakeConnection(cursor)

    input_repository.fetch_historical_scenarios(
        connection,
        T=4,
        num_samples=5,
    )

    _, parameters = cursor.executed[0]
    assert parameters == (5, 4)


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (input_repository.fetch_demand_forecasts, (object(), 0)),
        (input_repository.fetch_electricity_prices, (object(), 0)),
        (input_repository.fetch_pv_forecasts, (object(), 0)),
    ],
)
def test_time_dependent_fetches_reject_non_positive_T(
    function,
    args,
):
    with pytest.raises(ValueError):
        function(*args)


def test_read_optimization_run_returns_dict():
    created_at = datetime(2026, 8, 8, 12, 0, 0)
    cursor = FakeCursor(
        one_row=(
            37,
            4,
            5,
            21081.05,
            created_at,
        )
    )
    connection = FakeConnection(cursor)

    result = input_repository.read_optimization_run(
        connection,
        run_id=37,
    )

    assert result == {
        "run_id": 37,
        "T": 4,
        "num_samples": 5,
        "objective_value": 21081.05,
        "created_at": created_at,
    }


def test_read_optimization_run_returns_none_when_missing():
    cursor = FakeCursor(one_row=None)
    connection = FakeConnection(cursor)

    result = input_repository.read_optimization_run(
        connection,
        run_id=999,
    )

    assert result is None


def test_read_optimization_run_rejects_invalid_id():
    with pytest.raises(ValueError):
        input_repository.read_optimization_run(
            object(),
            run_id=0,
        )
