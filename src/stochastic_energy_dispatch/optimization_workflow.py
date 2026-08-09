from stochastic_energy_dispatch import ders_data_preprocessing as ders
from stochastic_energy_dispatch import input_repository, output_repository
from stochastic_energy_dispatch import samples_preprocessing as samples
from stochastic_energy_dispatch import system_data_preprocessing as system
from stochastic_energy_dispatch import tssp_model as tssp
from stochastic_energy_dispatch.case_schemas import TSSPRunConfig, TSSPRunResult
from stochastic_energy_dispatch.database import get_connection


def run_case(config: TSSPRunConfig) -> TSSPRunResult:
    """Run the complete TSSP workflow and return its summary."""
    # ---------- Data preprocessing ----------
    with get_connection() as connection:
        system_data = system.build_system_data(
            connection,
            config.T,
        )

        ders_data = ders.build_ders_data(
            connection,
            config.T,
        )

        empirical_samples_info = samples.build_samples_info(
            connection,
            config.T,
            config.num_samples,
        )

    # ---------- Solve TSSP optimization ----------
    tssp_result = tssp.solve_tssp(
        system_data,
        ders_data,
        empirical_samples_info,
        system_data.num_nodes,
        config.T,
    )

    # ---------- Save optimization results ----------
    with get_connection() as connection:
        run_id = output_repository.save_tssp_results(
            connection,
            tssp_result,
            ders_data,
            system_data.num_nodes,
            config.T,
            config.num_samples,
        )

    # ---------- Build run summary ----------
    return TSSPRunResult(
        run_id=run_id,
        T=config.T,
        num_samples=config.num_samples,
        objective_value=round(float(tssp_result["cost"]), 2),
    )


def get_run(run_id: int):
    if run_id <= 0:
        raise ValueError("run_id must be greater than 0.")

    with get_connection() as connection:
        return input_repository.read_optimization_run(
            connection,
            run_id,
        )
