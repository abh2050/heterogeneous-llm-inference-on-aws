"""Storage mode: `local` keeps SQLite exactly as before; `vercel` keeps the server stateless.

STORAGE_MODE=local|vercel selects it explicitly; otherwise Vercel's own VERCEL variable selects vercel.
In vercel mode the browser (IndexedDB) owns scenarios, benchmarks and hardware edits, the only server-side
file is a disposable model-metadata cache under /tmp, and the external benchmark adapter is disabled.
"""

import os
from pathlib import Path

VERCEL_CACHE = Path("/tmp/inference-atlas-cache.sqlite3")


def storage_mode() -> str:
    explicit = os.getenv("STORAGE_MODE", "").strip().lower()
    if explicit in {"local", "vercel"}:
        return explicit
    if explicit:
        raise ValueError("STORAGE_MODE must be 'local' or 'vercel'")
    return "vercel" if os.getenv("VERCEL") else "local"


def is_vercel() -> bool:
    return storage_mode() == "vercel"
