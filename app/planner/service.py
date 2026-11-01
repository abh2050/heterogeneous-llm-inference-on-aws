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
    # Client-held benchmark records (browser storage in vercel mode); None means use the server's store.
    benchmarks: list[dict] | None = Field(default=None, max_length=1000)


def sweep_concurrencies(request: PlanningRequest) -> list[int]:
    """The concurrency sweep, plus the peak concurrent users when the workload sets them."""
    sweep = set(request.concurrencies)
    peak = request.workload.effective_peak_users
    if peak:
        sweep.add(peak)
    return sorted(sweep)


def compare(request: PlanningRequest, runs):
    if any(c < 1 or c > 4096 for c in request.concurrencies):
        raise ValueError("Concurrency sweep must be between 1 and 4096")
    sweep = sweep_concurrencies(request)
    if sweep[-1] > 4096:
        raise ValueError("Peak concurrent users must be between 1 and 4096 to map onto concurrency")
    if len({h.id for h in request.hardware}) != len(request.hardware):
        raise ValueError("Hardware IDs must be unique")
    result = []
    for h in request.hardware:
        for tp in h.allowed_tp:
            if tp not in {1, 2, 4, 8}:
                raise ValueError("Only verified TP profiles 1, 2, 4, 8 are supported")
            for concurrency in sweep:
                c = estimate(request.model, h, request.workload, request.assumptions, tp, concurrency)
                c = apply_evidence(c, request.model, h, request.workload, request.assumptions, runs)
                if c.replicas:
                    c.concurrent_users_capacity = c.concurrency * c.replicas
                    peak = request.workload.effective_peak_users
                    c.meets_peak_users = None if peak is None else c.concurrent_users_capacity >= peak
                result.append(c)
    return rank_candidates(result)
