"""Offline, deterministic analysis of an LLM usage or cost export (OpenAI, Anthropic or similar).

Nothing here touches the network. The caller passes CSV text or a list of row dicts; column names are matched
leniently (case, spaces, dashes and underscores are ignored). Recognised columns:

- model:          model, model_name, model_id, snapshot_id, model_version, line_item, description
- cost (US$):     cost, cost_usd, amount, amount_value, amount_usd, usd, spend, total_cost, cost_in_usd, price
- requests:       requests, num_requests, num_model_requests, n_requests, request_count, calls
- input tokens:   input_tokens, prompt_tokens, n_context_tokens_total, uncached_input_tokens, input_uncached_tokens
- output tokens:  output_tokens, completion_tokens, n_generated_tokens_total
- cached input:   cache_read_input_tokens (Anthropic: on top of input), or input_cached_tokens, cached_tokens,
                  cached_input_tokens, n_cached_context_tokens_total (OpenAI: already part of input)
- cache writes:   cache_creation_input_tokens, cache_write_tokens
- errors:         errors, error_count, failed_requests, failures; or a status / status_code column
                  (rows with an error status count as errors, weighted by the requests column if present)
- retries:        retries, retry_count
- batch:          batch, is_batch, service_tier, batch_id (truthy, "batch" or "flex" counts as batch)
- date:           date, day, start_time, timestamp, bucket_start, start_date

One row may be one request or one aggregated bucket; both work. A ``line_item`` such as
"gpt-4o-2024-08-06, input" is read as the model before the comma.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any

COLUMNS: dict[str, tuple[str, ...]] = {
    "model": ("model", "modelname", "modelid", "snapshotid", "modelversion", "lineitem", "description"),
    "cost": (
        "cost",
        "costusd",
        "amount",
        "amountvalue",
        "amountusd",
        "usd",
        "spend",
        "totalcost",
        "costinusd",
        "price",
    ),
    "requests": ("requests", "numrequests", "nummodelrequests", "nrequests", "requestcount", "calls"),
    "input": ("inputtokens", "prompttokens", "ncontexttokenstotal", "uncachedinputtokens", "inputuncachedtokens"),
    "output": ("outputtokens", "completiontokens", "ngeneratedtokenstotal"),
    "cached": (
        "cachereadinputtokens",
        "inputcachedtokens",
        "cachedtokens",
        "cachedinputtokens",
        "ncachedcontexttokenstotal",
    ),
    "cache_write": ("cachecreationinputtokens", "cachewritetokens"),
    "errors": ("errors", "errorcount", "failedrequests", "failures"),
    "status": ("status", "statuscode"),
    "retries": ("retries", "retrycount"),
    "batch": ("batch", "isbatch", "servicetier", "batchid"),
    "date": ("date", "day", "starttime", "timestamp", "bucketstart", "startdate"),
}
MAX_ROWS = 200_000
PREMIUM = re.compile(r"(opus|-pro\b|-pro-|^o1|^o3(?!-mini)|gpt-4(?!o-mini|\.1-mini|\.1-nano|o)|large|ultra|-sol\b)")


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("$", "").replace("US", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _truthy_batch(v: Any) -> bool:
    s = str(v or "").strip().lower()
    return s in ("1", "true", "yes", "y", "batch", "flex") or (s.startswith("batch_") and len(s) > 6)


def _is_error_status(v: Any) -> bool:
    s = str(v or "").strip().lower()
    if not s:
        return False
    if s.isdigit():
        return not s.startswith("2")
    return s in ("error", "failed", "failure", "timeout", "rate_limited", "429", "500", "overloaded")


def read_rows(csv_text: str | None, rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if rows:
        return [dict(r) for r in rows[:MAX_ROWS] if isinstance(r, dict)]
    if not csv_text or not csv_text.strip():
        return []
    sample = csv_text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("﻿")), dialect=dialect)
    out = []
    for i, r in enumerate(reader):
        if i >= MAX_ROWS:
            break
        out.append(r)
    return out


def map_columns(names: list[str]) -> dict[str, str]:
    keys = {_key(n): n for n in names if n}
    found: dict[str, str] = {}
    for field, options in COLUMNS.items():
        for opt in options:
            if opt in keys:
                found[field] = keys[opt]
                break
    return found


def _model_of(v: Any) -> str:
    s = str(v or "").strip()
    return s.split(",")[0].strip() if "," in s else s


def aggregate(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], dict[str, str], dict[str, Any]]:
    names: list[str] = []
    for r in rows[:50]:
        for k in r:
            if k not in names:
                names.append(k)
    cols = map_columns(names)
    per: dict[str, dict[str, float]] = {}
    info: dict[str, Any] = {"rows_read": len(rows), "rows_skipped": 0, "dates": []}
    for r in rows:
        model = _model_of(r.get(cols["model"])) if "model" in cols else ""
        if not model:
            info["rows_skipped"] += 1
            continue
        a = per.setdefault(
            model,
            {
                k: 0.0
                for k in (
                    "cost",
                    "requests",
                    "input",
                    "output",
                    "cached",
                    "cache_write",
                    "errors",
                    "retries",
                    "batch_requests",
                    "rows",
                )
            },
        )
        a["rows"] += 1
        req = _num(r.get(cols["requests"])) if "requests" in cols else None
        weight = req if req is not None else 1.0
        a["requests"] += weight
        for f in ("cost", "input", "output", "cached", "cache_write", "errors", "retries"):
            if f in cols:
                a[f] += _num(r.get(cols[f])) or 0.0
        if "status" in cols and "errors" not in cols and _is_error_status(r.get(cols["status"])):
            a["errors"] += weight
        if "batch" in cols and _truthy_batch(r.get(cols["batch"])):
            a["batch_requests"] += weight
        if "date" in cols and r.get(cols["date"]):
            info["dates"].append(str(r.get(cols["date"]))[:10])
    return per, cols, info


def _pct(a: float, b: float) -> float | None:
    return round(100 * a / b, 1) if b else None


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Spend by model, shares and cost drivers. Retirement matching and next steps are added by the caller."""
    per, cols, info = aggregate(rows)
    has_cost = "cost" in cols
    rank_key = "cost" if has_cost else ("input" if "input" in cols else "requests")
    total = {k: sum(a[k] for a in per.values()) for k in ("cost", "requests", "input", "output", "cached")}
    # Anthropic reports cache reads apart from input_tokens; OpenAI's cached tokens are part of input_tokens.
    cached_separate = _key(cols.get("cached", "")) in ("cachereadinputtokens",) or "cache_write" in cols
    by_model = []
    for model, a in sorted(per.items(), key=lambda kv: (-kv[1][rank_key], kv[0])):
        tokens_in = a["input"] + (a["cached"] if cached_separate else 0.0)
        row: dict[str, Any] = {"model": model, "requests": round(a["requests"])}
        if has_cost:
            row["cost_usd"] = round(a["cost"], 4)
            row["share_of_spend_pct"] = _pct(a["cost"], total["cost"])
            if a["requests"]:
                row["cost_per_request_usd"] = round(a["cost"] / a["requests"], 6)
        if "input" in cols or "cached" in cols:
            row["input_tokens"] = round(tokens_in)
            if a["requests"]:
                row["avg_input_tokens_per_request"] = round(tokens_in / a["requests"])
        if "output" in cols:
            row["output_tokens"] = round(a["output"])
        if "cached" in cols:
            row["cached_input_share_pct"] = _pct(a["cached"], tokens_in)
        if "errors" in cols or "status" in cols:
            row["errors"] = round(a["errors"])
            row["error_rate_pct"] = _pct(a["errors"], a["requests"])
        if "retries" in cols:
            row["retries"] = round(a["retries"])
        if "batch" in cols:
            row["batch_share_pct"] = _pct(a["batch_requests"], a["requests"])
        by_model.append(row)

    share_field = "share_of_spend_pct" if has_cost else None
    if not has_cost and by_model:
        denom = total[rank_key] or 1
        for row, (_, a) in zip(by_model, sorted(per.items(), key=lambda kv: (-kv[1][rank_key], kv[0])), strict=True):
            row[f"share_of_{rank_key}_pct"] = _pct(a[rank_key], denom)
        share_field = f"share_of_{rank_key}_pct"
    shares = [r.get(share_field) or 0 for r in by_model] if share_field else []
    dates = sorted(info.pop("dates"))
    summary = {
        "rows_read": info["rows_read"],
        "rows_skipped_no_model": info["rows_skipped"],
        "columns_recognised": cols,
        "period": {"from": dates[0], "to": dates[-1]} if dates else None,
        "models_in_use": len(by_model),
        "ranked_by": "cost" if has_cost else f"{rank_key} (no cost column found)",
        "total_cost_usd": round(total["cost"], 2) if has_cost else None,
        "total_requests": round(total["requests"]),
        "top_model_share_pct": shares[0] if shares else None,
        "top3_share_pct": round(sum(shares[:3]), 1) if shares else None,
    }
    return {"summary": summary, "by_model": by_model, "cost_drivers": drivers(by_model, cols, has_cost, share_field)}


def drivers(
    by_model: list[dict[str, Any]], cols: dict[str, str], has_cost: bool, share_field: str | None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not by_model:
        return out
    top = by_model[0]
    top_share = top.get(share_field or "", 0) or 0
    if top_share >= 50:
        premium = bool(PREMIUM.search(top["model"].lower()))
        out.append(
            {
                "id": "one_model_carries_most",
                "finding": f"{top['model']} carries {top_share}% of {'spend' if has_cost else 'usage'}"
                + (" and is a premium-tier model." if premium else "."),
                "why_it_matters": "Most of the bill depends on one model choice. If a cheaper model does these tasks "
                "as well, this is the biggest lever; only a test on your own tasks shows which tasks it would lose.",
                "fix": "Measure a cheaper model on 20-30 of these tasks before switching (free pilot: request_pilot).",
            }
        )
    if has_cost:
        priced = [r for r in by_model if r.get("cost_per_request_usd") and r["requests"] >= 10]
        if len(priced) >= 2:
            hi = max(priced, key=lambda r: r["cost_per_request_usd"])
            lo = min(priced, key=lambda r: r["cost_per_request_usd"])
            ratio = hi["cost_per_request_usd"] / lo["cost_per_request_usd"]
            if ratio >= 5 and (hi.get("share_of_spend_pct") or 0) >= 20:
                out.append(
                    {
                        "id": "price_gap",
                        "finding": f"A request on {hi['model']} costs {ratio:.0f}x one on {lo['model']} "
                        f"(${hi['cost_per_request_usd']} vs ${lo['cost_per_request_usd']}), and it is "
                        f"{hi.get('share_of_spend_pct')}% of spend.",
                        "why_it_matters": "Simple, high-volume steps (classification, extraction, routing) often run "
                        "on the expensive model by default.",
                        "fix": "Route simple steps to the cheaper model you already use; test the change first.",
                    }
                )
    for r in by_model:
        avg = r.get("avg_input_tokens_per_request")
        cache = r.get("cached_input_share_pct")
        big = (r.get(share_field or "", 0) or 0) >= 10
        if avg and avg >= 8000 and big and (cache is None or cache < 30):
            out.append(
                {
                    "id": "large_uncached_context",
                    "model": r["model"],
                    "finding": f"{r['model']}: about {avg:,} input tokens per request"
                    + (f", only {cache}% served from cache." if cache is not None else "; no cache column seen."),
                    "why_it_matters": "Long repeated context (system prompt, tools, documents) is paid again on every "
                    "call.",
                    "fix": "Turn on prompt caching for the fixed prefix and trim or summarise long context.",
                }
            )
        out_tok, in_tok = r.get("output_tokens"), r.get("input_tokens")
        if out_tok and in_tok and in_tok / max(out_tok, 1) >= 20 and big and not (avg and avg >= 8000):
            out.append(
                {
                    "id": "context_heavy",
                    "model": r["model"],
                    "finding": f"{r['model']}: {in_tok / out_tok:.0f} input tokens for every output token.",
                    "why_it_matters": "Cost is driven by what is sent, not by what is generated.",
                    "fix": "Cache the fixed prefix, send fewer documents or tool definitions per call.",
                }
            )
        rate = r.get("error_rate_pct")
        if rate is not None and rate >= 2 and r["requests"] >= 20:
            out.append(
                {
                    "id": "errors_and_retries",
                    "model": r["model"],
                    "finding": f"{r['model']}: {rate}% of requests failed ({r['errors']} of {r['requests']}).",
                    "why_it_matters": "Failed calls and blind retries are paid work with no result.",
                    "fix": "Cap retries (e.g. 2 with backoff), stop on 4xx errors, fix the top error cause.",
                }
            )
        retries = r.get("retries")
        if retries and r["requests"] and retries / r["requests"] >= 0.05:
            out.append(
                {
                    "id": "many_retries",
                    "model": r["model"],
                    "finding": f"{r['model']}: {retries} retries for {r['requests']} requests.",
                    "why_it_matters": "Each retry is billed like a new call.",
                    "fix": "Cap retries and log why the first call failed.",
                }
            )
    if "batch" in cols:
        share = [r.get("batch_share_pct") or 0 for r in by_model]
        if share and max(share) == 0:
            out.append(
                {
                    "id": "no_batch",
                    "finding": "No requests used the batch tier.",
                    "why_it_matters": "OpenAI's and Anthropic's batch APIs are priced at about half (their published "
                    "pricing) for work that can wait up to 24 hours.",
                    "fix": "Move non-urgent jobs (evals, backfills, nightly reports) to the batch API.",
                }
            )
    return out
