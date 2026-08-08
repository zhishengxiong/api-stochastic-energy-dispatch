from dataclasses import dataclass


@dataclass
class TSSPRunConfig:
    T: int = 4
    num_samples: int = 5

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be greater than 0.")

        if not 1 <= self.num_samples <= 500:
            raise ValueError("num_samples must be between 1 and 500.")


@dataclass
class TSSPRunResult:
    run_id: int
    T: int
    num_samples: int
    objective_value: float
