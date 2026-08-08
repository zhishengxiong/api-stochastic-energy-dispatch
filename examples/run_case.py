from stochastic_energy_dispatch.case_schemas import TSSPRunConfig
from stochastic_energy_dispatch.optimization_workflow import run_case


def main() -> None:
    config = TSSPRunConfig(
        T=4,
        num_samples=5,
    )

    result = run_case(config)

    print(f"Results saved with run_id: {result.run_id}")
    print(f"The TSSP cost is: {result.objective_value}")


if __name__ == "__main__":
    main()
