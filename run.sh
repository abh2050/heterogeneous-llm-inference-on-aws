#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
exec uv run --frozen uvicorn app:app --host 127.0.0.1 --port "${PLANNER_PORT:-8765}" --no-access-log
