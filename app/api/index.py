"""Vercel entry point: the same FastAPI app as `./run.sh`, exposed as the /api function.

vercel.json rewrites /api/* here; static files in web/ are served by Vercel's CDN. No calculation code lives
in this file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402,F401
