"""Application orchestration shared by HTTP endpoints and integration tests."""

from planner.calibration import apply_evidence
from planner.models import Assumptions, Hardware, ModelSpec, Record, Workload
from planner.sizing import estimate, rank_candidates
from pydantic import Field


class PlanningRequest(Record):
    model: ModelSpec
    workload: Workload = Field(default_factory=Workload)
    assumptions: Assumptions = Field(default_factory=Assumptions)
    hardware: list[Hardware] = Field(min_length=1, max_length=30)
    concurrencies: list[int] = Field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128], min_length=1, max_length=20
    )


def compare(request: PlanningRequest, runs):
    if any(c < 1 or c > 4096 for c in request.concurrencies):
        raise ValueError("Concurrency sweep must be between 1 and 4096")
    if len({h.id for h in request.hardware}) != len(request.hardware):
        raise ValueError("Hardware IDs must be unique")
    result = []
    for h in request.hardware:
        for tp in h.allowed_tp:
            if tp not in {1, 2, 4, 8}:
                raise ValueError("Only verified TP profiles 1, 2, 4, 8 are supported")
            for concurrency in sorted(set(request.concurrencies)):
                c = estimate(request.model, h, request.workload, request.assumptions, tp, concurrency)
                result.append(
                    apply_evidence(c, request.model, h, request.workload, request.assumptions, runs)
                )
    return rank_candidates(result)
