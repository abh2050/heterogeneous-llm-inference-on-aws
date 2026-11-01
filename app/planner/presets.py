"""Workload presets: typical request shapes that fill the existing Workload fields.

Every value remains editable. Context length is the per-request maximum (input + output),
so the maximum input is derived as context length minus maximum output.
"""

from __future__ import annotations

from planner.models import Workload

PRESETS = [
    {
        "id": "chat",
        "name": "Chat",
        "summary": "Back-and-forth conversation with an assistant.",
        "why": "Prompts are short, so speed per person matters more than memory.",
        "input_avg": 512,
        "output_avg": 256,
        "output_max": 1024,
        "context_length": 4096,
    },
    {
        "id": "rag",
        "name": "Document Q&A (RAG)",
        "summary": "Answers questions using passages pulled from your documents.",
        "why": "Every question carries pages of text, so reading the prompt (the wait) dominates.",
        "input_avg": 6000,
        "output_avg": 400,
        "output_max": 1024,
        "context_length": 16384,
    },
    {
        "id": "coding",
        "name": "Coding assistant",
        "summary": "Completes and explains code using the files you have open.",
        "why": "Long code context plus long answers needs both memory and sustained speed.",
        "input_avg": 8000,
        "output_avg": 600,
        "output_max": 2048,
        "context_length": 32768,
    },
    {
        "id": "agents",
        "name": "Agents",
        "summary": "Software that plans, calls tools, and loops until a task is done.",
        "why": "Agents send long prompts again and again, so reusing cached prompts matters most.",
        "input_avg": 16000,
        "output_avg": 500,
        "output_max": 2048,
        "context_length": 32768,
    },
]


def preset_fields(preset: dict) -> dict:
    return {
        "input_avg": preset["input_avg"],
        "input_max": preset["context_length"] - preset["output_max"],
        "output_avg": preset["output_avg"],
        "output_max": preset["output_max"],
    }


def presets() -> list[dict]:
    """Presets with the exact Workload field values they set; validated against Workload."""
    result = []
    for preset in PRESETS:
        fields = preset_fields(preset)
        Workload(**fields)  # fail fast if a preset ever violates workload validation
        result.append({**preset, "workload": fields})
    return result
