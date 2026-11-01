"""Planner package. On Vercel only /tmp is writable, so point the Hugging Face cache there.

This must run before huggingface_hub is imported (it reads HF_HOME at import time), which is why it lives here.
"""

import os

if os.getenv("STORAGE_MODE", "").strip().lower() == "vercel" or (
    os.getenv("VERCEL") and os.getenv("STORAGE_MODE", "").strip().lower() != "local"
):
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
