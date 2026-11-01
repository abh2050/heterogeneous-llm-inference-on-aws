"""Sanitize untrusted imported metadata before persistence or export."""

import os
import re
from typing import Any

_SECRET_KEY = re.compile(
    r"password|secret|authorization|api.?key|credential|cookie|(?:^|[_-])(?:token|auth)$|access[_-]?token|refresh[_-]?token",
    re.I,
)
_TOKEN = re.compile(r"hf_[A-Za-z0-9]{12,}|Bearer\s+\S+", re.I)
_ASSIGNMENT = re.compile(
    r"((?:HF_TOKEN|access[_-]?token|refresh[_-]?token|api[_-]?key|password|secret|authorization)\s*[=:]\s*|"
    r"--(?:api-key|token|password|secret)(?:=|\s+))([^\s,;]+)",
    re.I,
)
_BASIC_AUTH = re.compile(r"(authorization\s*[:=]\s*)(?:Basic|Bearer)\s+\S+", re.I)
_URL_AUTH = re.compile(r"(https?://)[^/@\s]+:[^/@\s]+@", re.I)


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): ("[REDACTED]" if _SECRET_KEY.search(str(k)) and not _safe_key(str(k)) else scrub(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        result = []
        redact_next = False
        for item in value:
            result.append("[REDACTED]" if redact_next else scrub(item))
            redact_next = bool(
                isinstance(item, str)
                and item.startswith("--")
                and "=" not in item
                and _SECRET_KEY.search(item[2:])
            )
        return result
    if isinstance(value, str):
        value = _BASIC_AUTH.sub(r"\1[REDACTED]", value)
        value = _ASSIGNMENT.sub(r"\1[REDACTED]", value)
        value = _URL_AUTH.sub(r"\1[REDACTED]@", value)
        value = _TOKEN.sub("[REDACTED]", value)
        for name, secret in os.environ.items():
            if _SECRET_KEY.search(name) and len(secret) > 7:
                value = value.replace(secret, "[REDACTED]")
        return value
    return value


def _safe_key(key: str) -> bool:
    # Domain quantities containing 'token' are not authentication material.
    return key in {
        "input_throughput_tps",
        "output_throughput_tps",
        "max_batched_tokens",
        "max_num_batched_tokens",
        "batched_tokens",
        "block_size",
        "cost_per_million_output",
        "cost_per_million_total",
    } or key.endswith("_tokens")
