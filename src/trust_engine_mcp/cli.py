"""``trust-engine-mcp check [path] [--days N]``: the scan_for_models scan as a command, for CI and terminals.

Exit code 1 if a model found in the code retires within N days (default 60; a date that has already passed counts),
else 0. Under GitHub Actions (``GITHUB_ACTIONS=true``) it also prints ``::warning`` annotations and appends a table
to ``$GITHUB_STEP_SUMMARY``. Read-only and local; the only network call is the calendar GET.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, TextIO

from trust_engine_mcp import __version__
from trust_engine_mcp.data import DataSource, Settings
from trust_engine_mcp.tools import SEVEN_DAY_TEST_URL, ToolFailure, TrustEngineTools, with_ref

DEFAULT_DAYS = 60


def channel_from_env(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    return "action" if (env.get("GITHUB_ACTIONS") or "").lower() == "true" else "cli"


def _escape_data(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_prop(s: str) -> str:
    return _escape_data(s).replace(":", "%3A").replace(",", "%2C")


def _flag_event(item: dict[str, Any], days: int) -> dict[str, Any] | None:
    """The event that flags a model: the earliest upcoming date within N days, else the latest past date."""
    within = [e for e in item["events"] if e.get("days_left") is not None and e["days_left"] <= days]
    upcoming = [e for e in within if e["days_left"] >= 0]
    if upcoming:
        return min(upcoming, key=lambda e: e["days_left"])
    return max(within, key=lambda e: e["days_left"]) if within else None


def annotation(model: str, event: dict[str, Any], location: str, ref: str = "action") -> str:
    """One ``::warning`` line, message ≤ 2 lines, text as in the spec."""
    file, _, line = location.rpartition(":")
    days = event["days_left"]
    when = f"{days} days" if days >= 0 else f"{-days} days ago"
    verb = "retires" if days >= 0 else "was retired"
    repl = event.get("replacement") or "not named by the vendor"
    msg = (
        f"`{model}` {verb} on {event['platform']} on {event['date']} ({when}). Vendor replacement: {repl}. "
        f"Source: {event['source_url']}.\n"
        f"Before switching, test the replacement on your own tasks: {SEVEN_DAY_TEST_URL}?ref={ref}"
    )
    return f"::warning file={_escape_prop(file)},line={_escape_prop(line)}::{_escape_data(msg)}"


def summary_markdown(result: dict[str, Any], flagged: list[str], days: int) -> str:
    rows = ["### Model retirement check (trust-engine-mcp)", ""]
    found = result.get("found") or []
    if not found:
        rows.append("No model id from the retirement calendar was found.")
    else:
        rows += [
            "| Model | Status | Platform | Date | Days left | Replacement | Source | Found in |",
            "|---|---|---|---|---|---|---|---|",
        ]
        by_model = {r["model"]: r for r in result["check_my_models"]["results"]}
        for f in found:
            r = by_model.get(f["model"], {"events": []})
            where = ", ".join(f"`{x}`" for x in f["locations"][:5])
            if f["locations_total"] > 5:
                where += f" (+{f['locations_total'] - 5} more)"
            evs = r["events"] or [{}]
            for e in evs:
                src = f"[source]({e['source_url']})" if e.get("source_url") else ""
                dl = e.get("days_left")
                rows.append(
                    f"| `{f['model']}` | {f['status']} | {e.get('platform', '')} | {e.get('date') or ''} | "
                    f"{'' if dl is None else dl} | {e.get('replacement') or ''} | {src} | {where} |"
                )
    rows.append("")
    if flagged:
        rows.append(f"**Retiring within {days} days:** " + ", ".join(f"`{m}`" for m in flagged))
        rows.append("")
    if result.get("next_step"):
        rows += [result["next_step"], ""]
    return "\n".join(rows) + "\n"


def render(result: dict[str, Any], flagged: list[str], days: int) -> str:
    s = result["scanned"]
    data = (result.get("check_my_models") or {}).get("data") or {}
    cal = (result.get("check_my_models") or {}).get("calendar") or {}
    head = f"trust-engine-mcp {__version__} check: {s['files_scanned']} files scanned in {s['root']}"
    if data:
        head += f" (calendar: {data.get('source')}, updated {cal.get('updated')})"
    out = [head, ""]
    found = result.get("found") or []
    if not found:
        out.append("No model id from the retirement calendar was found.")
    else:
        by_model = {r["model"]: r for r in result["check_my_models"]["results"]}
        for f in found:
            out.append(f"{f['model']}  [{f['status']}]")
            r = by_model.get(f["model"], {"events": []})
            for e in r["events"]:
                dl = e.get("days_left")
                left = "" if dl is None else f" ({dl} days left)" if dl >= 0 else f" ({-dl} days ago)"
                out.append(
                    f"  {e['platform']}: {e.get('date') or 'date not set'}{left}; "
                    f"replacement: {e.get('replacement') or 'not named'}; source: {e['source_url']}"
                )
            if r.get("match_note"):
                out.append(f"  note: {r['match_note']}")
            more = (
                f" (+{f['locations_total'] - len(f['locations'])} more)"
                if f["locations_total"] > len(f["locations"])
                else ""
            )
            out.append(f"  found in: {', '.join(f['locations'])}{more}")
    if s["skipped_env_files_not_read"]:
        out.append("")
        out.append("Not read (secrets): " + ", ".join(s["skipped_env_files_not_read"]))
    for lim in s["limits_hit"]:
        out.append(f"Limit: {lim}")
    out.append("")
    if flagged:
        out.append(f"{len(flagged)} model(s) retire within {days} days: {', '.join(flagged)}.")
    else:
        out.append(f"No model found here retires within {days} days.")
    if result.get("next_step"):
        out += ["", result["next_step"]]
    return "\n".join(out) + "\n"


def run(
    path: str,
    days: int,
    tools: TrustEngineTools,
    env: dict[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    env = os.environ if env is None else env
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        result = tools.scan_for_models(path=path, allow_outside=True)
    except ToolFailure as exc:
        print(f"trust-engine-mcp check: {exc}", file=stderr)
        return 2
    result = with_ref(result, tools.channel)
    flagged: list[tuple[str, dict[str, Any]]] = []
    if result.get("found"):
        by_model = {r["model"]: r for r in result["check_my_models"]["results"]}
        for f in result["found"]:
            ev = _flag_event(by_model.get(f["model"], {"events": []}), days)
            if ev is not None:
                flagged.append((f["model"], ev))
    names = [m for m, _ in flagged]
    if tools.channel == "action":
        locs = {f["model"]: f["locations"] for f in result.get("found") or []}
        for model, ev in flagged:
            for loc in locs.get(model, []):
                print(annotation(model, ev, loc), file=stdout)
    stdout.write(render(result, names, days))
    if tools.channel == "action":
        summary_file = env.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "a", encoding="utf-8") as fh:
                fh.write(summary_markdown(result, names, days))
    stdout.flush()
    return 1 if flagged else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="trust-engine-mcp check",
        description="Find the AI model ids in a codebase and flag the ones the retirement calendar lists "
        "within N days. Read-only and local; only the public calendar is fetched.",
    )
    p.add_argument("path", nargs="?", default=".", help="file or folder to scan (default: .)")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS, help=f"exit 1 within N days (default {DEFAULT_DAYS})")
    a = p.parse_args(argv)
    channel = channel_from_env()
    tools = TrustEngineTools(DataSource(Settings.from_env(channel=channel)))
    return run(a.path, a.days, tools)
