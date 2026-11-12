"""Match the model names an agent uses against the retirement calendar's model ids."""

from __future__ import annotations

import re
from typing import Any

# A dated snapshot or version suffix after an alias: -20250929, -2024-08-06, -12-2025, -v1:0, -001, @001
_SNAPSHOT = re.compile(r"^(-\d{8}|-\d{4}-\d{2}-\d{2}|-\d{2}-\d{4}|-\d{3}|@\d{3})?(-v\d+(:\d+)?)?$")
_BEDROCK_VENDORS = (
    "ai21|amazon|anthropic|cohere|deepseek|google|meta|minimax|mistral|moonshot|nvidia|openai|qwen"
    "|twelvelabs|writer|zai"
)
_BEDROCK_PREFIX = re.compile(rf"^(?:[a-z]{{2}}\.)?(?:{_BEDROCK_VENDORS})\.")  # optional region prefix: us., eu.


def norm(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"^(models/|publishers/[^/]+/models/)", "", s)  # Gemini / Vertex resource names
    s = re.sub(r"^(openai|anthropic|google|mistral|mistralai)/", "", s)  # OpenRouter-style prefixes
    return re.sub(r"[\s_]+", "-", s)


def variants(name: str) -> set[str]:
    """'Claude Sonnet 4.5' -> {'claude-sonnet-4.5', 'claude-sonnet-4-5'}"""
    n = norm(name)
    out = {n, n.replace(".", "-")}
    return out | {core(v) for v in out}


def core(model_id: str) -> str:
    """Drop a Bedrock-style vendor prefix ('anthropic.') and keep the rest."""
    m = norm(model_id)
    return _BEDROCK_PREFIX.sub("", m, count=1)


def match(name: str, model_id: str) -> str | None:
    """'exact', 'snapshot' (the id is a dated snapshot of the alias you named) or None."""
    mid, c = norm(model_id), core(model_id)
    for v in variants(name):
        if v in (mid, c):
            return "exact"
        for base in (mid, c):
            if base.startswith(v) and _SNAPSHOT.match(base[len(v) :]):
                return "snapshot"
    return None


def find(name: str, events: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], list[str]]]:
    """All calendar events that list this model, with the match kind and the matching ids."""
    out = []
    for e in events:
        kinds = {}
        for mid in e.get("model_ids") or []:
            k = match(name, mid)
            if k:
                kinds[mid] = k
        if kinds:
            kind = "exact" if "exact" in kinds.values() else "snapshot"
            out.append((kind, e, sorted(kinds)))
    return out
