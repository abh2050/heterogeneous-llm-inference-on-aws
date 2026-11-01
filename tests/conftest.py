from datetime import datetime, timezone

import pytest

from planner.benchmark import build_run_context
from planner.catalog import load_catalog
from planner.demo import example_model
from planner.models import Assumptions, BenchmarkRecord, Workload
from planner.sizing import estimate


@pytest.fixture
def model():
    m = example_model()
    # Synthetic commit for unit tests only; never imported into application history.
    m.revision = "a" * 40
    return m


@pytest.fixture
def hardware():
    return load_catalog()[0]


@pytest.fixture
def workload():
    return Workload()


@pytest.fixture
def assumptions():
    return Assumptions()


@pytest.fixture
def run_factory(model, hardware, workload, assumptions):
    def make(c=None, w=None, run_id="test-run", **updates):
        w = w or workload
        c = c or estimate(model, hardware, w, assumptions, 1, 32)
        data = build_run_context(model, hardware, c, w, assumptions)
        data.update(
            run_id=run_id,
            timestamp=datetime(2026, 10, 31, tzinfo=timezone.utc).isoformat(),
            source="SYNTHETIC UNIT TEST ONLY",
            warmup_s=10,
            duration_s=60,
            sample_count=300,
            metrics={
                "ttft_s": {"mean": 0.12, "p95": 0.2},
                "tpot_s": {"mean": 0.015, "p95": 0.02},
                "itl_s": {"p95": 0.035},
                "request_throughput_rps": 4,
                "output_throughput_tps": 1024,
                "errors": 0,
                "timeouts": 0,
            },
        )
        data.update(updates)
        return BenchmarkRecord.model_validate(data)

    return make
