import logging

from fastapi import FastAPI, HTTPException, Path, status

from stochastic_energy_dispatch import input_repository
from stochastic_energy_dispatch.api_schemas import (
    StoredRunResponse,
    TSSPRunRequest,
    TSSPRunResponse,
)
from stochastic_energy_dispatch.case_schemas import TSSPRunConfig
from stochastic_energy_dispatch.database import get_connection
from stochastic_energy_dispatch.optimization_workflow import run_case


logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/")
def root():
    return {"message": "TSSP API is running"}


@app.post(
    "/runs",
    response_model=TSSPRunResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_run(request: TSSPRunRequest):
    config = TSSPRunConfig(
        T=request.T,
        num_samples=request.num_samples,
    )

    try:
        result = run_case(config)
    except Exception as error:
        logger.exception("TSSP optimization run failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TSSP optimization run failed.",
        ) from error

    return TSSPRunResponse(
        run_id=result.run_id,
        T=result.T,
        num_samples=result.num_samples,
        objective_value=result.objective_value,
    )


@app.get(
    "/runs/{run_id}",
    response_model=StoredRunResponse,
)
def read_run(run_id: int = Path(gt=0)):
    with get_connection() as connection:
        result = input_repository.read_optimization_run(
            connection,
            run_id,
        )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found.",
        )

    return result
