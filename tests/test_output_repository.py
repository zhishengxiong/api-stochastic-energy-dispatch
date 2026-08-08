from types import SimpleNamespace

import numpy as np
import pytest

from stochastic_energy_dispatch import output_repository


class FakeCursor:
    def __init__(self, run_id=37):
        self.run_id = run_id
        self.execute_calls = []
        self.executemany_calls = []

    def execute(self, query, parameters=None):
        self.execute_calls.append((query, parameters))

    def executemany(self, query, rows):
        self.executemany_calls.append((query, list(rows)))

    def fetchone(self):
        return (self.run_id,)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def _valid_result():
    return {
        "cost": 21081.05,
        "x_hat": {
            "G_p": np.array([[1.0, 2.0]]),
            "G_u": np.array([[1.0, 1.0]]),
            "G_s": np.array([[0.0, 0.0]]),
            "R_pmax": np.array([[3.0, 4.0]]),
            "ESS_pch": np.array([[0.1, 0.2]]),
            "ESS_pdis": np.array([[0.0, 0.0]]),
            "ESS_E": np.array([[2.0, 2.1]]),
            "ESS_u": np.array([[1.0, 1.0]]),
            "P_net": np.array(
                [
                    [10.0, 11.0],
                    [20.0, 21.0],
                ]
            ),
            "P_flow": np.array(
                [
                    [5.0, 6.0],
                    [7.0, 8.0],
                ]
            ),
            "flow_buy": np.array([9.0, 10.0]),
            "flow_sell": np.array([0.0, 0.0]),
            "flow_u": np.array([1.0, 1.0]),
        },
    }


def _ders_data():
    return SimpleNamespace(
        G_id=[10],
        G_node=[2],
        R_id=[20],
        R_node=[3],
        ESS_id=[30],
        ESS_node=[4],
    )


def test_validate_tssp_result_rejects_missing_cost():
    result = _valid_result()
    del result["cost"]

    with pytest.raises(KeyError, match="Missing TSSP result keys"):
        output_repository._validate_tssp_result(result)


def test_validate_tssp_result_rejects_missing_decision_variable():
    result = _valid_result()
    del result["x_hat"]["G_p"]

    with pytest.raises(
        KeyError,
        match="Missing first-stage result keys",
    ):
        output_repository._validate_tssp_result(result)


def test_save_tssp_results_writes_all_result_groups():
    cursor = FakeCursor(run_id=37)
    connection = FakeConnection(cursor)

    run_id = output_repository.save_tssp_results(
        connection=connection,
        tssp_result=_valid_result(),
        ders_data=_ders_data(),
        num_nodes=3,
        T=2,
        num_samples=5,
    )

    assert run_id == 37

    # One INSERT for optimization_runs.
    assert len(cursor.execute_calls) == 1

    # generator, reserve, storage, node, line, grid exchange
    assert len(cursor.executemany_calls) == 6

    row_counts = [
        len(rows)
        for _, rows in cursor.executemany_calls
    ]

    assert row_counts == [
        2,  # 1 generator x 2 periods
        2,  # 1 reserve x 2 periods
        2,  # 1 storage x 2 periods
        4,  # 2 non-slack nodes x 2 periods
        4,  # 2 lines x 2 periods
        2,  # 2 periods
    ]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("num_nodes", 0),
        ("T", 0),
        ("num_samples", 0),
    ],
)
def test_save_tssp_results_rejects_non_positive_dimensions(
    field_name,
    value,
):
    kwargs = {
        "connection": object(),
        "tssp_result": _valid_result(),
        "ders_data": _ders_data(),
        "num_nodes": 3,
        "T": 2,
        "num_samples": 5,
    }
    kwargs[field_name] = value

    with pytest.raises(ValueError):
        output_repository.save_tssp_results(**kwargs)
