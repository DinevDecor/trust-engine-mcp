"""Offline tests: a fake agenttrustapi.com (httpx.MockTransport) and fixtures. No network."""

import asyncio
import json
from datetime import date
from pathlib import Path
from urllib.parse import unquote

import httpx
import pytest

from trust_engine_mcp import data as dm
from trust_engine_mcp import models as mm
from trust_engine_mcp import server as ms
from trust_engine_mcp.tools import ToolFailure, TrustEngineTools

FIX = Path(__file__).parent / "fixtures"
TODAY = date(2026, 11, 12)
CAL = json.loads(dm.bundled_text("retirements.json"))
BENCH = json.loads(dm.bundled_text("aider-model-diet.json"))


class FakeSite:
    def __init__(self, status=200, retirements=CAL):
        self.status = status
        self.retirements = retirements
        self.calls = []

    def __call__(self, request):
        self.calls.append((request.method, request.url.path))
        if self.status != 200:
            return httpx.Response(self.status, text="down")
        if request.url.path == dm.RETIREMENTS_PATH:
            return httpx.Response(200, json=self.retirements)
        if request.url.path == dm.BENCHMARK_PATH:
            return httpx.Response(200, json=BENCH)
        return httpx.Response(404)


def make(site=None, offline=False, clock=None):
    site = site or FakeSite()
    settings = dm.Settings(base_url="https://site.example", offline=offline)
    src = dm.DataSource(
        settings,
        client_factory=lambda: httpx.Client(base_url=settings.base_url, transport=httpx.MockTransport(site)),
        clock=clock or (lambda: 0.0),
    )
    return TrustEngineTools(src, today=lambda: TODAY), site


# --- data source --------------------------------------------------------------------------------------------------
def test_live_fetch_is_cached_and_get_only():
    t, site = make()
    a = t.model_retirements()
    b = t.model_retirements()
    assert a["data"]["source"] == "live" and b["data"]["cached"] is True
    assert site.calls == [("GET", dm.RETIREMENTS_PATH)]


def test_fallback_to_bundled_snapshot_when_site_down():
    t, _ = make(FakeSite(status=503))
    r = t.model_retirements()
    assert r["data"]["source"] == "bundled snapshot" and "HTTP 503" in r["data"]["why"]
    assert r["count"] > 0


def test_offline_mode_makes_no_request():
    t, site = make(offline=True)
    t.check_my_models(["gpt-4o"])
    t.benchmark_results()
    assert site.calls == []


def test_failed_fetch_retried_after_short_wait():
    now = [0.0]
    site = FakeSite(status=500)
    t, _ = make(site, clock=lambda: now[0])
    t.model_retirements()
    site.status = 200
    now[0] = dm.FALLBACK_RETRY_SECONDS + 1
    assert t.model_retirements()["data"]["source"] == "live"


# --- calendar data ------------------------------------------------------------------------------------------------
def test_every_event_has_https_source_and_iso_date():
    for e in CAL["events"]:
        assert e["source_url"].startswith("https://")
        assert e["date"] is None or date.fromisoformat(e["date"])
        assert e["event"] in ("retirement", "deprecation", "redirect")
    assert CAL["license"] == "CC BY 4.0"


# --- model matching -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,mid,kind",
    [
        ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929", "snapshot"),
        ("Claude Sonnet 4.5", "anthropic.claude-sonnet-4-5-20250929-v1:0", "snapshot"),
        ("us.anthropic.claude-opus-4-1-20250805-v1:0", "anthropic.claude-opus-4-1-20250805-v1:0", "exact"),
        ("gpt-4o", "gpt-4o-2024-08-06", "snapshot"),
        ("gpt-5", "gpt-5-mini", None),
        ("gpt-4o", "gpt-4o-realtime", None),
        ("o3", "o3-pro", None),
        ("models/gemini-2.5-flash", "gemini-2.5-flash", "exact"),
        ("gemini-2.5-flash", "gemini-2.5-flash-image", None),
        ("openai/gpt-5", "gpt-5", "exact"),
    ],
)
def test_match(name, mid, kind):
    assert mm.match(name, mid) == kind


# --- tools --------------------------------------------------------------------------------------------------------
def test_check_my_models_sonnet_45_both_platforms():
    t, _ = make()
    r = t.check_my_models(["claude-sonnet-4-5", "my-own-model"])
    son, unknown = r["results"]
    assert son["status"] == "retiring"
    dates = {e["platform"]: (e["date"], e["days_left"]) for e in son["events"]}
    assert dates["Anthropic"] == ("2026-11-30", 18)
    assert dates["AWS Bedrock"] == ("2027-04-08", 147)
    assert "before 2026-11-30" in son["next_step"] and "claude-sonnet-5-5" in son["next_step"]
    assert unknown["status"] == "not on the calendar"


def test_check_my_models_needs_input():
    t, _ = make()
    with pytest.raises(ToolFailure):
        t.check_my_models([])


def test_model_retirements_filters():
    t, _ = make()
    r = t.model_retirements(provider="anthropic", before_date="2026-12-31")
    plats = {e["platform"] for e in r["events"]}
    assert plats == {"Anthropic", "AWS Bedrock"}  # Claude on Bedrock counts as anthropic
    assert all(e["date"] <= "2026-12-31" for e in r["events"])
    assert all(e["days_left"] >= 0 for e in r["events"])
    openai = t.model_retirements(provider="openai", text_models_only=True)
    assert openai["count"] > 0 and all(e["category"] != "other" for e in openai["events"])
    with pytest.raises(ToolFailure):
        t.model_retirements(before_date="soon")


def test_past_events_hidden_by_default():
    t, _ = make()
    t._today = lambda: date(2027, 1, 1)
    assert all(e["status"] != "already passed" for e in t.model_retirements()["events"])
    assert any(e["status"] == "already passed" for e in t.model_retirements(include_past=True)["events"])


def test_benchmark_numbers_match_the_report():
    t, _ = make()
    r = t.benchmark_results()
    by = {m["model"]: m for m in r["models"]}
    assert (by["claude-opus-5-5"]["solved"], by["claude-sonnet-5-5"]["solved"], by["claude-haiku-5-5"]["solved"]) == (
        26,
        26,
        22,
    )
    assert by["claude-sonnet-5-5"]["cost_per_solved_usd"] == pytest.approx(0.15, abs=0.005)
    assert by["claude-opus-5-5"]["cost_per_solved_usd"] == pytest.approx(0.30, abs=0.005)
    assert r["raw_data_url"].endswith("results.csv") and r["method_url"].endswith("protocol.md")


def test_switch_savings_measured_pair():
    t, _ = make()
    r = t.estimate_switch_savings(1000, "Claude Opus 5.5", "sonnet")
    assert r["measured"] is True and r["label"].startswith("Estimate from one public benchmark")
    assert r["estimated_monthly_difference_usd"] == pytest.approx(1000 * (1 - r["ratio"]), abs=0.01)
    assert 450 < r["estimated_monthly_difference_usd"] < 520  # about half
    assert "lost none" in r["quality_in_benchmark"]


def test_switch_savings_refuses_unmeasured_pair():
    t, _ = make()
    r = t.estimate_switch_savings(1000, "gpt-4o", "gpt-5-mini")
    assert r["measured"] is False and "pilot" in r["pilot_offer"].lower()
    assert "estimated_monthly_bill_usd" not in r
    assert t.estimate_switch_savings(1000, "claude-sonnet-4-5", "claude-haiku-5-5")["measured"] is False


def test_request_pilot_sends_nothing_and_builds_mailto():
    t, site = make()
    r = t.request_pilot(company="Acme", use_case="invoice extraction", models=["gpt-4o"], task_count=25)
    assert r["sent"] is False and r["to"] == "hello@agenttrustapi.com"
    assert r["mailto"].startswith("mailto:hello@agenttrustapi.com?subject=")
    body = unquote(r["mailto"].split("&body=", 1)[1])
    assert "Acme" in body and "tasks_template.csv" in body
    assert site.calls == []


def test_pricing_is_published_list():
    t, _ = make()
    p = t.pricing()
    assert p["valid_from"] == "2026-11-12" and p["offers"][0]["price"] == "$0"


def test_pricing_has_new_products_12_nov():
    t, _ = make()
    offers = {o["name"]: o for o in t.pricing()["offers"]}
    seven = offers["7-day test"]
    assert seven["price"] == "$1,390 fixed" and seven["currency"] == "USD"
    assert "money back" in seven["guarantee"] and seven["page"].endswith("/seven-day-test/")
    assert "$50" in seven["scope"] and "7 working days" in seven["you_get"]
    assert list(offers).index("7-day test") == 1
    watch = offers["24/7 watch on your tasks"]["price"]
    assert "$169" in watch and "$349" in watch
    assert "$20" in offers["24/7 watch on your tasks"]["model_calls"]
    assert "$690 per 100 answers" in offers["Bulgarian review and sign-off"]["price"]
    assert offers["Diet audit"]["price"] == "$2,500 to $7,500 by scope"


# --- usage analysis -----------------------------------------------------------------------------------------------
def test_analyze_openai_cost_export():
    t, site = make(offline=True)
    r = t.analyze_usage(csv_text=(FIX / "openai_usage.csv").read_text())
    assert "not uploaded" in r["privacy"]
    top = r["by_model"][0]
    assert top["model"] == "gpt-4o-2024-08-06" and top["share_of_spend_pct"] > 80
    ids = {d["id"] for d in r["cost_drivers"]}
    assert {"one_model_carries_most", "large_uncached_context", "price_gap"} <= ids
    cal = {m["model"]: m for m in r["models_on_retirement_calendar"]}
    assert cal["gpt-4o-2024-08-06"]["dates"][0]["date"] == "2027-01-14"
    assert any("caching" in s for s in r["next_steps"]["quick_fixes_you_can_do"])
    assert "request_pilot" in r["next_steps"]["is_a_cheaper_model_good_enough"]
    assert site.calls == []


def test_analyze_anthropic_rows_with_errors_and_cache():
    t, _ = make(offline=True)
    rows = json.loads((FIX / "anthropic_rows.json").read_text())
    r = t.analyze_usage(rows=rows)
    by = {m["model"]: m for m in r["by_model"]}
    opus = by["claude-opus-4-1-20250805"]
    assert opus["error_rate_pct"] == 5.0
    assert opus["input_tokens"] == 30_000 * 100 + 0  # uncached input + cache reads (none)
    son = by["claude-sonnet-4-5-20250929"]
    assert son["cached_input_share_pct"] == 80.0
    assert "errors_and_retries" in {d["id"] for d in r["cost_drivers"]}
    assert any(m["model"] == "claude-sonnet-4-5-20250929" for m in r["models_on_retirement_calendar"])


def test_analyze_without_cost_ranks_by_tokens_and_is_deterministic():
    t, _ = make(offline=True)
    csv_text = "Model Name;Prompt Tokens;Completion Tokens\nm-small;100;10\nm-big;5000;50\nm-big;5000;50\n"
    a = t.analyze_usage(csv_text=csv_text)
    assert a == t.analyze_usage(csv_text=csv_text)
    assert a["summary"]["ranked_by"].startswith("input") and a["by_model"][0]["model"] == "m-big"


def test_analyze_usage_rejects_empty_and_modelless():
    t, _ = make(offline=True)
    with pytest.raises(ToolFailure):
        t.analyze_usage(csv_text="")
    with pytest.raises(ToolFailure):
        t.analyze_usage(csv_text="foo,bar\n1,2\n")


# --- pilot tasks --------------------------------------------------------------------------------------------------
def test_prepare_pilot_tasks_format_and_check():
    t, _ = make(offline=True)
    info = t.prepare_pilot_tasks()
    assert info["template_csv"].startswith("id,input,expected,grader,tolerance,fields,important,notes")
    assert set(info["columns"]) == {"id", "input", "expected", "grader", "tolerance", "fields", "important", "notes"}
    assert "[E-MAIL 1]" in info["masking"]
    r = t.prepare_pilot_tasks(csv_text=(FIX / "draft_tasks.csv").read_text())
    c = r["check"]
    assert c["usable_tasks"] == 3 and c["important"] == 1
    assert any("t02" in e and "not a number" in e for e in c["errors"])
    assert any("t03" in p and "e-mail" in p for p in c["possible_personal_data"])
    assert any("20-30" in a for a in c["advice"])
    assert "ann@example.com" not in json.dumps(c)  # never echoes task text


def test_template_example_rows_are_skipped():
    t, _ = make(offline=True)
    c = t.prepare_pilot_tasks(csv_text=t.prepare_pilot_tasks()["template_csv"])["check"]
    assert c["usable_tasks"] == 0 and not c["errors"]


# --- MCP surface --------------------------------------------------------------------------------------------------
def test_server_lists_all_tools():
    t, _ = make(offline=True)
    names = {x.name for x in asyncio.run(ms.build_server(t).list_tools())}
    assert names == {
        "check_my_models",
        "model_retirements",
        "analyze_usage",
        "benchmark_results",
        "estimate_switch_savings",
        "prepare_pilot_tasks",
        "request_pilot",
        "pricing",
    }
