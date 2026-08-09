from contextlib import contextmanager
from types import SimpleNamespace

from stochastic_energy_dispatch import optimization_workflow
from stochastic_energy_dispatch.case_schemas import TSSPRunConfig


@contextmanager
def _fake_connection():
    yield object()


def test_run_case_orchestrates_full_workflow(monkeypatch):
    system_data = SimpleNamespace(num_nodes=33)
    ders_data = SimpleNamespace()
    samples_info = {
        "empirical_samples": object(),
        "num_empirical_samples": 5,
    }

    monkeypatch.setattr(
        optimization_workflow,
        "get_connection",
        _fake_connection,
    )

    monkeypatch.setattr(
        optimization_workflow.system,
        "build_system_data",
        lambda connection, T: system_data,
    )

    monkeypatch.setattr(
        optimization_workflow.ders,
        "build_ders_data",
        lambda connection, T: ders_data,
    )

    monkeypatch.setattr(
        optimization_workflow.samples,
        "build_samples_info",
        lambda connection, T, num_samples: samples_info,
    )

    monkeypatch.setattr(
        optimization_workflow.tssp,
        "solve_tssp",
        lambda system_data_arg, ders_data_arg, samples_info_arg, num_nodes, T: {
            "cost": 21081.051,
            "x_hat": {},
        },
    )

    monkeypatch.setattr(
        optimization_workflow.output_repository,
        "save_tssp_results",
        lambda connection, tssp_result, ders_data_arg, num_nodes, T, num_samples: 37,
    )

    config = TSSPRunConfig(
        T=4,
        num_samples=5,
    )

    result = optimization_workflow.run_case(config)

    assert result.run_id == 37
    assert result.T == 4
    assert result.num_samples == 5
    assert result.objective_value == 21081.05
