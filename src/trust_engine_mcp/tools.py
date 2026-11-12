"""Tool logic, separate from MCP so it can be tested without a transport."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, datetime, timezone
from typing import Any

from trust_engine_mcp import models as mm
from trust_engine_mcp import pilot, usage
from trust_engine_mcp.data import BENCHMARK_PATH, RETIREMENTS_PATH, DataSource

PILOT_HINT = (
    "Free pilot: Trust Engine AI runs 20-30 of your own tasks on your current model and on cheaper setups "
    "and reports the cost per finished task and exactly which tasks each setup gets wrong. "
    "Use request_pilot to draft the request and prepare_pilot_tasks for the task file."
)
COVERAGE_NOTE = (
    "Calendar covers Anthropic (Claude API), OpenAI, Google Gemini API and Vertex AI, AWS Bedrock and "
    "Mistral. Not covered: Azure OpenAI, Claude on Google-managed Vertex."
)

PRICING = {
    "currency": "US$",
    "valid_from": "2026-11-12",
    "source": "https://agenttrustapi.com/#pricing",
    "vat": "VAT, where it applies, is shown on the invoice.",
    "offers": [
        {
            "name": "Pilot",
            "price": "$0",
            "scope": "20-30 of your tasks, 3 configurations",
            "you_get": "one-page report within a week: cost per finished task and the tasks each cheaper setup fails",
            "notes": "Trust Engine AI pays the model calls, up to $10",
        },
        {
            "name": "7-day test",
            "price": "$1,390 fixed",
            "currency": "USD",
            "scope": "your 20 tasks with expected answers, 3 model setups (your current one plus 2 cheaper or "
            "alternative ones); model calls included up to $50, above that at cost and agreed first",
            "you_get": "signed report in 7 working days: cost per finished task, quality, the named tasks each "
            "setup gets wrong, and a recommendation",
            "guarantee": "If we cannot measure your task, you get your money back.",
            "payment": "card (payment link), bank transfer or USDC",
            "order": "https://agenttrustapi.com/seven-day-test/#order",
            "page": "https://agenttrustapi.com/seven-day-test/",
            "notes": "the fast fixed-price answer; the diet audit covers larger scopes",
        },
        {
            "name": "Diet audit",
            "price": "$2,500 to $7,500 by scope",
            "scope": "up to 100-150 tasks, 4-5 configurations; critical cases fixed before measuring",
            "you_get": "signed report, recommended setup, raw records; one re-test after your changes included "
            "(otherwise $500)",
            "notes": "model calls billed at cost",
        },
        {
            "name": "Re-test",
            "price": "$199 to $499 a month, or $750 per event",
            "scope": "when a model is retired or changed, on a new agent version, or each quarter",
            "you_get": "dated report of which tasks changed and the new cost per finished task",
        },
        {
            "name": "Pay from savings",
            "price": "20-25% of verified savings for 12 months, with a cap agreed up front",
            "scope": "instead of the fixed audit fee",
            "you_get": "the same audit",
            "notes": "savings counted on your bills, per task, net of new costs: "
            "https://agenttrustapi.com/methodology/#savings",
        },
        {
            "name": "24/7 watch on your tasks",
            "price": "$169 a month (up to 20 tasks, weekly runs) or $349 a month (up to 50 tasks, daily runs)",
            "model_calls": "included up to $20 a month; above that by agreement",
            "currency": "USD",
            "scope": "your frozen task set re-run on your production model setting",
            "you_get": "e-mail alert when results or cost per task change (silent model updates, price changes, "
            "retirements); monthly written summary",
            "notes": "public model trackers watch the models; this watches your own tasks",
        },
        {
            "name": "Bulgarian review and sign-off",
            "price": "$690 per 100 answers, or about $50 an hour for smaller jobs",
            "currency": "USD",
            "scope": "your chatbot's or agent's Bulgarian outputs, checked against your expected answers and rules",
            "you_get": "each answer marked pass or fail with a reason, reviewed and signed by a native Bulgarian "
            "speaker; turnaround as agreed in the written scope",
        },
        {
            "name": "Retirement re-test with evidence annex (add-on)",
            "price": "$2,500 to $4,500, then $400-800 a month",
            "scope": "your frozen task set on the retiring model and each candidate replacement",
            "you_get": "which tasks change, cost per finished task, recommended replacement; test evidence, "
            "not legal advice",
        },
        {
            "name": "Pre-launch check (add-on)",
            "price": "$500 to $1,500; repeat $250-500",
            "scope": "sealed exam plus a short diet test on your tasks",
            "you_get": "dated public result page, valid 90 days, published at your call",
        },
    ],
    "terms": "https://agenttrustapi.com/terms/",
    "contact": pilot.EMAIL,
    "independence": "No vendor pays to appear in public results; client audits stay private.",
}


def _iso(d: str | None) -> date | None:
    if not d:
        return None
    try:
        return date.fromisoformat(str(d)[:10])
    except ValueError:
        return None


class ToolFailure(Exception):
    """A tool could not complete; the message is safe to show to the agent."""


class TrustEngineTools:
    def __init__(self, source: DataSource, today: Callable[[], date] | None = None) -> None:
        self.source = source
        self._today = today or (lambda: datetime.now(timezone.utc).date())

    # --- retirements -------------------------------------------------------------------------------------------
    def _calendar(self) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        data, prov = self.source.get(RETIREMENTS_PATH)
        events = data.get("events")
        if not isinstance(events, list):
            raise ToolFailure("retirements.json has no events list")
        meta = {k: data.get(k) for k in ("updated", "checked", "license", "homepage")}
        return events, meta, prov

    def _with_days(self, e: dict[str, Any]) -> dict[str, Any]:
        d = _iso(e.get("date"))
        out = dict(e)
        out["days_left"] = (d - self._today()).days if d else None
        out["status"] = (
            "announced, date not set" if d is None else "already passed" if d < self._today() else "upcoming"
        )
        return out

    def model_retirements(
        self,
        provider: str | None = None,
        model: str | None = None,
        before_date: str | None = None,
        include_past: bool = False,
        text_models_only: bool = False,
    ) -> dict[str, Any]:
        events, meta, prov = self._calendar()
        before = _iso(before_date)
        if before_date and before is None:
            raise ToolFailure("before_date must be YYYY-MM-DD")
        today = self._today()
        out = []
        for e in events:
            if provider:
                p = provider.strip().lower()
                hay = f"{e.get('provider', '')} {e.get('platform', '')}".lower()
                ids = " ".join(e.get("model_ids") or []).lower()
                if p not in hay and not re.search(rf"(^|[\s.]){re.escape(p)}\.", ids):
                    continue
            if model and not mm.find(model, [e]) and model.strip().lower() not in e.get("model", "").lower():
                continue
            d = _iso(e.get("date"))
            if before and (d is None or d > before):
                continue
            if d and d < today and not include_past:
                continue
            if text_models_only and e.get("category") == "other":
                continue
            out.append(self._with_days(e))
        return {
            "count": len(out),
            "today": today.isoformat(),
            "filters": {
                "provider": provider,
                "model": model,
                "before_date": before_date,
                "include_past": include_past,
                "text_models_only": text_models_only,
            },
            "events": out,
            "calendar": meta,
            "data": prov,
            "coverage": COVERAGE_NOTE,
            "how_to_read": "date is when the model stops working on that platform; the same model can retire on "
            "different dates on different platforms. Always cite source_url.",
        }

    def check_my_models(self, models: list[str]) -> dict[str, Any]:
        if not models:
            raise ToolFailure("Pass the model ids or names your agent uses, e.g. ['claude-sonnet-4-5', 'gpt-4o'].")
        events, meta, prov = self._calendar()
        results = []
        for name in models[:100]:
            hits = mm.find(name, events)
            if not hits:
                results.append(
                    {
                        "model": name,
                        "status": "not on the calendar",
                        "meaning": "No retirement listed on the covered platforms as of the calendar date. "
                        "This is not a guarantee; check your provider's deprecation page.",
                        "events": [],
                    }
                )
                continue
            evs = []
            for kind, e, ids in sorted(hits, key=lambda h: (h[1].get("date") is None, h[1].get("date") or "")):
                x = self._with_days(e)
                x["match"] = kind
                x["matched_ids"] = ids
                evs.append(x)
            upcoming = [x for x in evs if x["status"] == "upcoming"]
            first = upcoming[0] if upcoming else None
            if first:
                repl = first.get("replacement") or "the vendor's recommended replacement or a cheaper candidate"
                step = (
                    f"Re-test your agent's tasks on {repl} before {first['date']} "
                    f"({first['days_left']} days left on {first['platform']})."
                )
                status = "retiring"
            elif any(x["status"] == "announced, date not set" for x in evs):
                step = "Deprecation announced without a date: plan the re-test now, before a date is set."
                status = "deprecated, date not set"
            else:
                step = "Listed retirement date has passed: calls may fail or be redirected; switch and re-test now."
                status = "retired"
            note = None
            if any(x["match"] == "snapshot" for x in evs):
                note = (
                    "Matched a dated snapshot of the name you gave. If your code pins a different snapshot or "
                    "uses the alias, check which one the alias points to."
                )
            results.append({"model": name, "status": status, "next_step": step, "match_note": note, "events": evs})
        return {
            "today": self._today().isoformat(),
            "results": results,
            "calendar": meta,
            "data": prov,
            "coverage": COVERAGE_NOTE,
            "re_test_help": PILOT_HINT,
        }

    # --- benchmark ---------------------------------------------------------------------------------------------
    def benchmark_results(self) -> dict[str, Any]:
        data, prov = self.source.get(BENCHMARK_PATH)
        return dict(
            data,
            data=prov,
            honesty="One public benchmark (27 coding exercises), not your tasks. Comparing the models is "
            "fair; absolute rates are not a forecast.",
        )

    def estimate_switch_savings(self, monthly_bill_usd: float, from_model: str, to_model: str) -> dict[str, Any]:
        if monthly_bill_usd is None or monthly_bill_usd < 0:
            raise ToolFailure("monthly_bill_usd must be a number >= 0 (the part of the bill spent on from_model).")
        data, prov = self.source.get(BENCHMARK_PATH)
        names = [m["model"] for m in data.get("models", [])]
        a, b = self._resolve(from_model, names), self._resolve(to_model, names)
        if not a or not b or a == b:
            return {
                "measured": False,
                "from_model": from_model,
                "to_model": to_model,
                "message": "We have not measured this pair, so we will not guess a number. Our public benchmark "
                f"covers only {', '.join(names)} (Aider coding agent).",
                "pilot_offer": PILOT_HINT,
                "measured_pairs": [f"{p['from']} -> {p['to']}" for p in data.get("pairs", [])],
            }
        pair = next(p for p in data["pairs"] if p["from"] == a and p["to"] == b)
        by = {m["model"]: m for m in data["models"]}
        ratio = pair["cost_per_solved_ratio"]
        new_bill = monthly_bill_usd * ratio
        quality = (
            f"{by[b]['name']} solved {pair['solved_to']} of {by[b]['exercises']} exercises vs "
            f"{pair['solved_from']} for {by[a]['name']}"
            + (f"; it lost: {', '.join(pair['lost_tasks'])}." if pair["lost_tasks"] else "; it lost none.")
        )
        return {
            "measured": True,
            "you_asked": {"from": from_model, "to": to_model},
            "label": "Estimate from one public benchmark; a pilot measures your tasks.",
            "from_model": a,
            "to_model": b,
            "cost_per_solved_usd": {a: by[a]["cost_per_solved_usd"], b: by[b]["cost_per_solved_usd"]},
            "ratio": ratio,
            "arithmetic": f"${monthly_bill_usd:,.2f} x {ratio} = ${new_bill:,.2f} a month; "
            f"difference ${monthly_bill_usd - new_bill:,.2f} "
            f"({'saving' if new_bill <= monthly_bill_usd else 'extra cost'}).",
            "estimated_monthly_bill_usd": round(new_bill, 2),
            "estimated_monthly_difference_usd": round(monthly_bill_usd - new_bill, 2),
            "quality_in_benchmark": quality,
            "assumes": "Your tasks behave like 27 coding exercises run by the Aider agent, and the bill you gave is "
            "spent on from_model only. Real tasks differ; losses on rare cases matter more than averages.",
            "benchmark": {
                "report_url": data.get("report_url"),
                "raw_data_url": data.get("raw_data_url"),
                "method_url": data.get("method_url"),
            },
            "data": prov,
            "pilot_offer": PILOT_HINT,
        }

    @staticmethod
    def _resolve(name: str, names: list[str]) -> str | None:
        n = mm.norm(name or "").replace(".", "-").removeprefix("anthropic/")
        if n in names:
            return n
        if not n.startswith("claude-"):
            n = "claude-" + n
        if n in names:
            return n
        fam = [m for m in names if m.split("-")[1] == n.split("-")[1]] if n.count("-") == 1 else []
        return fam[0] if len(fam) == 1 else None

    # --- usage, pilot, pricing ---------------------------------------------------------------------------------
    def analyze_usage(self, csv_text: str | None = None, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        data = usage.read_rows(csv_text, rows)
        if not data:
            raise ToolFailure(
                "No rows. Pass csv_text (the exported CSV as text) or rows (a list of objects). "
                "See the tool description for accepted column names."
            )
        result = usage.analyze(data)
        if not result["by_model"]:
            raise ToolFailure(
                "No model column found. Accepted: model, model_name, model_id, snapshot_id, "
                f"line_item. Columns seen: {', '.join(list(data[0])[:20])}"
            )
        names = [r["model"] for r in result["by_model"]]
        try:
            check = self.check_my_models(names)
            on_calendar = [
                {
                    "model": r["model"],
                    "status": r["status"],
                    "next_step": r["next_step"],
                    "dates": [
                        {
                            "platform": e["platform"],
                            "date": e["date"],
                            "days_left": e["days_left"],
                            "replacement": e["replacement"],
                            "source_url": e["source_url"],
                        }
                        for e in r["events"]
                    ],
                }
                for r in check["results"]
                if r["events"]
            ]
            cal = {"calendar_updated": check["calendar"].get("updated"), "data": check["data"]["source"]}
        except ToolFailure:
            on_calendar, cal = [], {"error": "retirement calendar unavailable"}
        drivers = result["cost_drivers"]
        quick = []
        ids = {d["id"] for d in drivers}
        if ids & {"large_uncached_context", "context_heavy"}:
            quick.append(
                "Turn on prompt caching for the fixed part of each prompt (system prompt, tool "
                "definitions, shared documents) and trim context that is sent on every call."
            )
        if ids & {"errors_and_retries", "many_retries"}:
            quick.append("Cap retries (at most 2, with backoff), stop on 4xx errors, and fix the most common error.")
        if "no_batch" in ids or "batch" not in result["summary"]["columns_recognised"]:
            quick.append("Move work that can wait (evals, backfills, nightly jobs) to the provider's batch API.")
        if "price_gap" in ids:
            quick.append(
                "Route simple steps (classification, extraction, routing) to the cheaper model you "
                "already use, after a test."
            )
        quick.append("Set max output tokens per call and a monthly budget alert in the provider console.")
        steps = {
            "quick_fixes_you_can_do": quick,
            "retirements": (
                "Plan a re-test before each date listed in models_on_retirement_calendar."
                if on_calendar
                else "No model in this export is on the retirement calendar."
            ),
            "is_a_cheaper_model_good_enough": "Averages do not show which tasks a cheaper model gets wrong. "
            + PILOT_HINT,
        }
        return {
            "privacy": "Analysed locally inside this MCP server process. The usage data was not uploaded or sent "
            "anywhere; only the public retirement calendar was read.",
            **result,
            "models_on_retirement_calendar": on_calendar,
            "calendar": cal,
            "next_steps": steps,
            "limits": "Only what the columns show. Without a cost column, ranking is by tokens or requests. "
            "Findings are heuristics to check, not conclusions.",
        }

    def prepare_pilot_tasks(self, csv_text: str | None = None, use_case: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "goal": "20-30 real tasks from your owner's agent, each with an expected answer a person has checked, "
            "in one CSV. Trust Engine AI runs them on the current model and cheaper setups and reports "
            "which tasks each setup gets wrong.",
            "use_case": use_case,
            "template_url": pilot.TEMPLATE_URL,
            "template_csv": pilot.template_csv(),
            "columns": pilot.COLUMNS,
            "graders": {
                "exact": "answer equals expected (trimmed)",
                "contains": "expected text appears in the answer",
                "regex": "answer matches the expected regular expression",
                "number": "answer is a number within tolerance of expected",
                "json_fields": "answer is JSON and the listed fields equal those in expected",
            },
            "how_to_pick_tasks": pilot.HOW_TO_PICK,
            "masking": pilot.MASKING,
            "personal_data_checks": sorted(pilot.PII),
            "then": "Check the draft with prepare_pilot_tasks(csv_text=...) (offline), then draft the request with "
            "request_pilot. Your owner sends it; the task file goes only after Trust Engine AI replies.",
        }
        if csv_text:
            out["check"] = pilot.check_tasks(csv_text)
            out["privacy"] = "Checked locally; the file was not uploaded or sent anywhere."
        return out

    def request_pilot(
        self,
        company: str | None = None,
        use_case: str | None = None,
        models: list[str] | None = None,
        task_count: int | None = None,
        monthly_bill_usd: float | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        r = pilot.request(company, use_case, models, task_count, monthly_bill_usd, notes)
        r["tasks_format"] = {"template_url": pilot.TEMPLATE_URL, "how": "prepare_pilot_tasks"}
        return r

    def pricing(self) -> dict[str, Any]:
        return PRICING
