"""Pilot intake: the tasks CSV format, an offline checker for a draft, and the pilot request e-mail.

The format and checks follow Trust Engine AI's intake converter (columns, graders, "example row" skipping, personal
data patterns). Nothing is sent: the request is a mailto: link and a body for the human owner to send.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any
from urllib.parse import quote

from trust_engine_mcp.data import TEMPLATE_PATH, bundled_text

EMAIL = "hello@agenttrustapi.com"
TEMPLATE_URL = "https://agenttrustapi.com" + TEMPLATE_PATH
COLUMNS = {
    "id": "Short unique id, e.g. t01.",
    "input": "The exact task text your agent receives (one task per row). Mask personal data first.",
    "expected": "The correct answer, checked by a person. For json_fields: a JSON object.",
    "grader": "How the answer is checked: exact | contains | regex | number | json_fields (default exact).",
    "tolerance": "Only for grader=number: allowed absolute difference, e.g. 0.01.",
    "fields": "Only for grader=json_fields: the keys that must match, separated by ; (default: all keys).",
    "important": "yes for the rare cases that must never go wrong; no otherwise.",
    "notes": "Free text, ignored by the checker. Rows whose notes start with 'example row' are skipped.",
}
GRADERS = {"exact", "contains", "regex", "number", "json_fields"}
PII = {
    "e-mail": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?<!\d)(?:\+|00)\d[\d \-()]{7,}\d"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "card number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "national id (10 digits)": re.compile(r"(?<!\d)\d{10}(?!\d)"),
}
MASKING = (
    "Before anything leaves your owner's machine, replace personal data in input and expected with stable "
    "placeholders: the same e-mail becomes [E-MAIL 1] everywhere, the next one [E-MAIL 2]; same for names "
    "([NAME 1]), phones, IBANs, card numbers, national ids and addresses. Keep the structure the task needs "
    "(a placeholder e-mail still looks like an e-mail if the task is to extract one). Synthetic copies of real "
    "cases are fine. If the tasks cannot be masked, say so in the request: a data processing agreement "
    "(https://agenttrustapi.com/dpa/) is signed before any data is sent."
)
HOW_TO_PICK = [
    "Take 20-30 real tasks from your owner's agent logs, from one workflow (e.g. ticket triage, invoice extraction).",
    "Mix them: about 60% everyday cases, 30% harder ones, and the 3-5 rare cases that must never go wrong "
    "(mark those important=yes).",
    "Use only tasks with a checkable answer: extraction, classification, structured output, code with tests, "
    "lookups. Not open conversation or creative writing.",
    "The expected answer must be what a careful person would accept, checked by your owner, not copied from the "
    "current model's output unless it was verified.",
    "Pick the grader that matches how strict the check must be: number with a tolerance for amounts, json_fields "
    "for structured output, contains for labels inside longer text.",
    "Delete the three example rows (or keep notes starting 'example row'; they are skipped).",
]


def template_csv() -> str:
    return bundled_text("tasks_template.csv")


def check_tasks(csv_text: str) -> dict[str, Any]:
    """Offline check of a draft tasks CSV. Returns counts and problems by row id; never echoes task text."""
    rows = list(csv.DictReader(io.StringIO(csv_text.lstrip("﻿"))))
    missing = [c for c in ("id", "input", "expected") if rows and c not in {k.strip().lower() for k in rows[0] if k}]
    errors: list[str] = []
    pii: list[str] = []
    seen: set[str] = set()
    tasks = important = 0
    graders: dict[str, int] = {}
    for n, raw in enumerate(rows, start=2):
        r = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
        if r.get("notes", "").lower().startswith("example row"):
            continue
        tid = r.get("id") or f"row{n}"
        if tid in seen:
            errors.append(f"{tid}: duplicate id")
        seen.add(tid)
        g = (r.get("grader") or "exact").lower()
        if g not in GRADERS:
            errors.append(f"{tid}: unknown grader '{g}' (use one of {', '.join(sorted(GRADERS))})")
            continue
        if not r.get("input") or r.get("expected", "") == "":
            errors.append(f"{tid}: input or expected is empty")
            continue
        exp = r["expected"]
        if g == "number":
            try:
                float(exp.replace(",", ""))
            except ValueError:
                errors.append(f"{tid}: expected is not a number")
                continue
        elif g == "json_fields":
            try:
                if not isinstance(json.loads(exp), dict):
                    raise ValueError
            except ValueError:
                errors.append(f"{tid}: expected is not a JSON object")
                continue
        elif g == "regex":
            try:
                re.compile(exp)
            except re.error:
                errors.append(f"{tid}: expected is not a valid regex")
                continue
        for kind, rx in PII.items():
            if rx.search(r["input"]) or rx.search(exp):
                pii.append(f"{tid}: possible {kind}")
        tasks += 1
        important += r.get("important", "").lower() in ("yes", "y", "true", "1")
        graders[g] = graders.get(g, 0) + 1
    advice = []
    if missing:
        advice.append(f"Missing columns: {', '.join(missing)}. Start from the template.")
    if tasks < 20:
        advice.append(f"Only {tasks} usable tasks; 20-30 are needed for a pilot.")
    if tasks > 30:
        advice.append(f"{tasks} tasks: the free pilot uses 20-30; pick the most representative ones.")
    if not important:
        advice.append("No task is marked important=yes; add the 3-5 rare cases that must never go wrong.")
    if pii:
        advice.append("Possible personal data found: mask it before sending (see masking).")
    if not errors and not pii and not missing and 20 <= tasks <= 30:
        advice.append(
            "Ready: the file matches the intake format. Mention it in the pilot request; send the file "
            "only after Trust Engine AI replies."
        )
    return {
        "usable_tasks": tasks,
        "important": important,
        "graders": dict(sorted(graders.items())),
        "errors": errors,
        "possible_personal_data": pii,
        "advice": advice,
    }


def request(
    company: str | None,
    use_case: str | None,
    models: list[str] | None,
    task_count: int | None,
    monthly_bill_usd: float | None,
    notes: str | None,
) -> dict[str, Any]:
    subject = "Free pilot request" + (f": {company.strip()}" if company and company.strip() else "")
    lines = [
        "Hello Trust Engine AI,",
        "",
        "We would like a free pilot: which cheaper model or setup does our agent's tasks as well as the current one.",
        "",
        f"Company: {company.strip() if company else '[company name]'}",
        f"What the agent does: {use_case.strip() if use_case else '[one or two sentences]'}",
        f"Models in use: {', '.join(models) if models else '[e.g. claude-opus-..., gpt-...]'}",
    ]
    if monthly_bill_usd:
        lines.append(f"Approximate monthly model bill: US${monthly_bill_usd:,.0f}")
    lines += [
        f"Tasks: {task_count if task_count else '20-30'} tasks with checked expected answers, prepared in your "
        f"template format ({TEMPLATE_URL}), personal data masked. We will send the file after your reply.",
    ]
    if notes:
        lines += ["", notes.strip()]
    lines += ["", "Best regards,", "[name]"]
    body = "\n".join(lines)
    return {
        "to": EMAIL,
        "subject": subject,
        "body": body,
        "mailto": f"mailto:{EMAIL}?subject={quote(subject)}&body={quote(body)}",
        "sent": False,
        "next": [
            "Show the draft to your owner; the owner sends it from their own mail client. This tool sends nothing.",
            "Do not attach task data to the first e-mail. Prepare it with prepare_pilot_tasks and send it after "
            "the reply; if it cannot be masked, a data processing agreement comes first.",
            "Pilot terms: free, 20-30 tasks, 3 configurations, one-page report within about a week; Trust Engine AI "
            "pays the model calls up to US$10.",
        ],
    }
