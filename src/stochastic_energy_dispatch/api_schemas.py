from datetime import datetime

from pydantic import BaseModel, Field


class TSSPRunRequest(BaseModel):
    T: int = Field(default=4, gt=0)
    num_samples: int = Field(default=5, ge=1, le=500)


class TSSPRunResponse(BaseModel):
    run_id: int
    T: int
    num_samples: int
    objective_value: float


class StoredRunResponse(BaseModel):
    run_id: int
    T: int
    num_samples: int
    objective_value: float
    created_at: datetime
