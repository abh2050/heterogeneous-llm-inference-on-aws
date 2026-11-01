"""Single-user SQLite persistence; raw imports are sanitized at the boundary."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from planner.models import BenchmarkRecord, ModelSpec
from planner.security import scrub


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("PLANNER_DB", ".local/planner.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS model_cache (
                model_id TEXT NOT NULL, revision TEXT NOT NULL, sha TEXT NOT NULL,
                payload TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY (model_id, revision));
            CREATE TABLE IF NOT EXISTS benchmarks (
                run_id TEXT PRIMARY KEY, payload TEXT NOT NULL, imported_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS scenarios (
                name TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL);
            """)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def cached_model(self, model_id: str, revision: str) -> ModelSpec | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM model_cache WHERE model_id=? AND revision=?", (model_id, revision)
            ).fetchone()
        return ModelSpec.model_validate_json(row[0]) if row else None

    def cache_model(self, model: ModelSpec):
        payload = json.dumps(scrub(model.model_dump(mode="json")))
        with self.connect() as db:
            for rev in {model.requested_revision, model.revision} - {None}:
                db.execute(
                    "INSERT OR REPLACE INTO model_cache VALUES (?, ?, ?, ?, ?)",
                    (model.model_id, rev, model.revision, payload, self.now()),
                )

    def add_benchmark(self, record: BenchmarkRecord):
        self.add_benchmarks([record])

    def add_benchmarks(self, records: list[BenchmarkRecord]):
        """A batch either commits fully or rolls back, including duplicate-ID conflicts."""
        with self.connect() as db:
            for record in records:
                payload = json.dumps(scrub(record.model_dump(mode="json")), sort_keys=True)
                old = db.execute("SELECT payload FROM benchmarks WHERE run_id=?", (record.run_id,)).fetchone()
                if old and old[0] != payload:
                    raise ValueError("Run ID already exists with different content; use a unique run ID")
                db.execute(
                    "INSERT OR IGNORE INTO benchmarks VALUES (?, ?, ?)", (record.run_id, payload, self.now())
                )

    def benchmarks(self) -> list[BenchmarkRecord]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM benchmarks ORDER BY imported_at DESC").fetchall()
        return [BenchmarkRecord.model_validate_json(r[0]) for r in rows]

    def save_scenario(self, name: str, payload: dict[str, Any]):
        if not name.strip():
            raise ValueError("Scenario name is required")
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO scenarios VALUES (?, ?, ?)",
                (name.strip(), json.dumps(scrub(payload)), self.now()),
            )

    def scenarios(self):
        with self.connect() as db:
            rows = db.execute("SELECT name, payload FROM scenarios ORDER BY updated_at DESC").fetchall()
        return {name: json.loads(payload) for name, payload in rows}

    def save_job(self, job_id: str, payload: dict):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?, ?, ?)",
                (job_id, json.dumps(scrub(payload)), self.now()),
            )

    def jobs(self):
        with self.connect() as db:
            return {r[0]: json.loads(r[1]) for r in db.execute("SELECT job_id, payload FROM jobs")}
