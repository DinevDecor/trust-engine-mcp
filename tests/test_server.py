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
from trust_engine_mcp import tools
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


def make(site=None, offline=False, clock=None, channel="mcp"):
    site = site or FakeSite()
    settings = dm.Settings(base_url="https://site.example", offline=offline, channel=channel)
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
    assert "before 2026-11-30" in son["action"] and "claude-sonnet-5-5" in son["action"]
    assert r["next_step"] == tools.next_step_line("mcp") and "re_test_help" not in r
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
        "scan_for_models",
    }


# --- scan_for_models, CLI, links (v0.2.0) -------------------------------------------------------------------------
from trust_engine_mcp import __version__, cli  # noqa: E402
from trust_engine_mcp import scan as sc  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text('import openai\n\nMODEL = "gpt-4.1"\nOTHER = "gpt-4.1-mini"\n')
    (tmp_path / "config.yaml").write_text("llm:\n  model: claude-sonnet-4-5\n  fallback: gpt-5.1-codex-max\n")
    (tmp_path / ".env.example").write_text("MODEL=gpt-4o\n")
    (tmp_path / ".env").write_text("MODEL=gemini-2.5-flash\nOPENAI_API_KEY=sk-not-a-real-key\n")
    (tmp_path / ".env.local").write_text("MODEL=gemini-2.5-pro\n")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text('const m = "o3";\n')
    (tmp_path / "package-lock.json").write_text('{"gpt-5": 1}\n')
    (tmp_path / "logo.bin").write_bytes(b"\x89PNG\x00\x00gpt-5\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_scan_finds_ids_with_file_and_line(repo):
    t, _ = make()
    r = t.scan_for_models()
    found = {f["model"]: f for f in r["found"]}
    assert found["gpt-4.1"]["locations"] == ["src/app.py:3"]
    assert found["gpt-4.1-mini"]["locations"] == ["src/app.py:4"]
    assert found["claude-sonnet-4-5"]["locations"] == ["config.yaml:2"]
    assert found["gpt-4o"]["locations"] == [".env.example:1"]
    assert "gpt-5.1" not in found and "gpt-5" not in found  # whole tokens only; lockfile and binary skipped
    assert "o3" not in found  # node_modules skipped
    assert found["claude-sonnet-4-5"]["status"] == "retiring"
    # the same result object check_my_models returns for those models
    direct = t.check_my_models(sorted(found))
    assert r["check_my_models"]["results"] == direct["results"]


def test_scan_never_reads_env_files(repo, monkeypatch):
    opened = []
    real_open = open

    def spy(file, *a, **k):
        opened.append(Path(file).name)
        return real_open(file, *a, **k)

    monkeypatch.setattr("builtins.open", spy)
    t, _ = make()
    r = t.scan_for_models()
    assert sorted(r["scanned"]["skipped_env_files_not_read"]) == [".env", ".env.local"]
    assert ".env" not in opened and ".env.local" not in opened and ".env.example" in opened
    found = {f["model"] for f in r["found"]}
    assert not found & {"gemini-2.5-flash", "gemini-2.5-pro"}
    assert "sk-not-a-real-key" not in json.dumps(r)
    assert r["scanned"]["skipped_binary_files"] == 1


def test_scan_path_inside_cwd_only(repo, tmp_path_factory):
    t, _ = make()
    assert {f["model"] for f in t.scan_for_models(path="src")["found"]} == {"gpt-4.1", "gpt-4.1-mini"}
    outside = tmp_path_factory.mktemp("outside")
    for p in ("..", str(outside), "/etc", "src/../../"):
        with pytest.raises(ToolFailure, match="inside the current working directory"):
            t.scan_for_models(path=p)
    (repo / "link").symlink_to(outside)
    with pytest.raises(ToolFailure):
        t.scan_for_models(path="link")


def test_scan_texts_and_env_names(repo):
    t, _ = make()
    r = t.scan_for_models(
        texts=[
            {"name": "main.py", "content": "client.messages.create(model='claude-sonnet-4-5-20250929')"},
            {"name": "deploy/.env", "content": "MODEL=gpt-4o"},
        ]
    )
    assert [(f["model"], f["locations"]) for f in r["found"]] == [("claude-sonnet-4-5-20250929", ["main.py:1"])]
    assert r["scanned"]["skipped_env_files_not_read"] == ["deploy/.env"]


def test_scan_limits(repo, monkeypatch):
    monkeypatch.setattr(sc, "MAX_FILES", 2)
    t, _ = make()
    r = t.scan_for_models()
    assert r["scanned"]["files_scanned"] == 2 and r["scanned"]["complete"] is False
    monkeypatch.setattr(sc, "MAX_FILES", 2000)
    monkeypatch.setattr(sc, "MAX_BYTES", 30)
    r = t.scan_for_models()
    assert r["scanned"]["bytes_scanned"] <= 30 and "MB limit" in r["scanned"]["limits_hit"][0]
    big = [{"name": f"f{i}", "content": "x"} for i in range(5)]
    monkeypatch.setattr(sc, "MAX_FILES", 3)
    assert t.scan_for_models(texts=big)["scanned"]["files_scanned"] == 3


def test_scan_nothing_found_has_no_next_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("print('hello')\n")
    t, _ = make()
    r = t.scan_for_models()
    assert r["found"] == [] and "next_step" not in r


def test_retiring_answers_end_with_exactly_one_next_step(repo):
    t, _ = make()
    for r in (t.check_my_models(["gpt-4.1", "claude-sonnet-4-5"]), t.scan_for_models()):
        text = json.dumps(r, ensure_ascii=False)
        line = tools.next_step_line("mcp")
        assert text.count("Same code, different model, different answers.") == 1
        assert list(r)[-1] == "next_step" and r["next_step"] == line
        assert text.count('"next_step"') == 1
    assert tools.next_step_line("mcp") == (
        "Same code, different model, different answers. Test the replacement on 20–30 of your own tasks before "
        "the date: free pilot or the fixed 7-day test → https://agenttrustapi.com/seven-day-test/?ref=mcp"
    )
    # not retiring: no conversion line
    r = t.check_my_models(["my-own-model"])
    assert "next_step" not in r and "re_test_help" in r


def test_user_agent_per_channel():
    for ch in ("mcp", "cli", "action"):
        src = dm.DataSource(dm.Settings(channel=ch))
        with src._client_factory() as c:
            assert c.headers["User-Agent"] == f"trust-engine-mcp/{__version__} ({ch})"
    assert __version__ == "0.2.0"


def test_ref_added_to_every_site_link():
    out = tools.with_ref(
        {
            "a": "see https://agenttrustapi.com/seven-day-test/#order.",
            "b": ["(https://agenttrustapi.com/dpa/)", "https://agenttrustapi.com"],
            "c": "https://agenttrustapi.com/x?y=1 and https://example.com/z",
            "d": "https://agenttrustapi.com/seven-day-test/?ref=mcp",
            "e": "hello@agenttrustapi.com",
        },
        "cli",
    )
    assert out == {
        "a": "see https://agenttrustapi.com/seven-day-test/?ref=cli#order.",
        "b": ["(https://agenttrustapi.com/dpa/?ref=cli)", "https://agenttrustapi.com/?ref=cli"],
        "c": "https://agenttrustapi.com/x?y=1&ref=cli and https://example.com/z",
        "d": "https://agenttrustapi.com/seven-day-test/?ref=mcp",
        "e": "hello@agenttrustapi.com",
    }


def test_mcp_answers_carry_ref_mcp():
    t, _ = make(offline=True)
    server = ms.build_server(t)
    res = asyncio.run(server.call_tool("pricing", {}))
    text = json.dumps(res, default=str)
    assert "seven-day-test/?ref=mcp" in text and "agenttrustapi.com/terms/?ref=mcp" in text


def test_scan_tool_registered_with_spec_description():
    t, _ = make(offline=True)
    tool = next(x for x in asyncio.run(ms.build_server(t).list_tools()) if x.name == "scan_for_models")
    assert tool.description.startswith(
        "Find every AI model id used in this codebase (OpenAI, Anthropic, Gemini, Bedrock, Mistral) and tell which "
        "ones are being retired or deprecated, by when, on which platform, and the vendor's replacement, with the "
        'source link. Use it before upgrading dependencies, when a model call starts failing, or when asked "are we '
        'using any deprecated models?". Read-only and local.'
    )
    schema = getattr(tool, "input_schema", None) or tool.inputSchema
    assert set(schema["properties"]) == {"path", "texts"}


def _run_cli(repo, days, channel="cli", env=None):
    import io

    t, _ = make(channel=channel)
    out, err = io.StringIO(), io.StringIO()
    code = cli.run(".", days, t, env=env or {}, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_cli_exit_codes(repo):
    code, out, _ = _run_cli(repo, 60)  # claude-sonnet-4-5: 18 days on Anthropic
    assert code == 1 and "retire within 60 days: claude-sonnet-4-5" in out
    assert out.rstrip().endswith(tools.next_step_line("cli"))
    assert out.count("Same code, different model") == 1 and "::warning" not in out
    code, out, _ = _run_cli(repo, 10)
    assert code == 0 and "No model found here retires within 10 days" in out
    (repo / "src" / "app.py").unlink()
    (repo / "config.yaml").unlink()
    (repo / ".env.example").unlink()
    code, out, _ = _run_cli(repo, 365)
    assert code == 0 and "No model id from the retirement calendar was found" in out


def test_cli_past_retirement_counts(tmp_path, monkeypatch):
    import io

    monkeypatch.chdir(tmp_path)
    (tmp_path / "m.py").write_text('M = "gpt-4.1"\n')
    cal = json.loads(json.dumps(CAL))
    for e in cal["events"]:
        if "gpt-4.1-2025-04-14" in e["model_ids"]:
            e["date"] = "2026-11-01"
    t, _ = make(site=FakeSite(retirements=cal), channel="cli")
    out = io.StringIO()
    assert cli.run(".", 0, t, env={}, stdout=out, stderr=io.StringIO()) == 1
    assert "11 days ago" in out.getvalue()


def test_cli_github_annotations_and_summary(repo, tmp_path_factory):
    summary = tmp_path_factory.mktemp("gh") / "summary.md"
    code, out, _ = _run_cli(repo, 90, channel="action", env={"GITHUB_STEP_SUMMARY": str(summary)})
    assert code == 1
    warns = [x for x in out.splitlines() if x.startswith("::warning ")]
    assert (
        "::warning file=src/app.py,line=3::`gpt-4.1` retires on OpenAI on 2027-01-14 (63 days). Vendor replacement: "
        "gpt-6-luna. Source: https://developers.openai.com/api/docs/deprecations.%0ABefore switching, test the "
        "replacement on your own tasks: https://agenttrustapi.com/seven-day-test/?ref=action"
    ) in warns
    assert any(
        w.startswith("::warning file=config.yaml,line=2::`claude-sonnet-4-5` retires on Anthropic") for w in warns
    )
    assert all(w.count("%0A") == 1 for w in warns)  # two lines
    md = summary.read_text()
    assert "| `gpt-4.1` | retiring | OpenAI | 2027-01-14 | 63 |" in md
    assert md.rstrip().endswith(tools.next_step_line("action"))
    assert out.rstrip().endswith(tools.next_step_line("action"))


def test_cli_channel_from_env():
    assert cli.channel_from_env({"GITHUB_ACTIONS": "true"}) == "action"
    assert cli.channel_from_env({}) == "cli"


def test_entry_point_dispatch(monkeypatch, repo):
    monkeypatch.setenv("TRUST_ENGINE_OFFLINE", "1")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with pytest.raises(SystemExit) as e:
        ms.main(["check", ".", "--days", "1"])
    assert e.value.code == 0
    started = []
    monkeypatch.setattr(ms, "build_server", lambda t: type("S", (), {"run": lambda self: started.append(1)})())
    ms.main([])
    assert started == [1]
