# trust-engine-mcp

Free MCP server from [Trust Engine AI](https://agenttrustapi.com). It answers questions agents otherwise search the web for:

- **Which AI models does this code use, and are any being retired?** `scan_for_models` finds the model ids in the
  working directory (read-only, on your machine) and checks them against the calendar. The same scan runs as a
  command (`trust-engine-mcp check`) and as a GitHub Action, without an agent.
- **Is a model we use being retired, and when?** Verified dates for Anthropic, OpenAI, Google Gemini API and Vertex AI,
  AWS Bedrock and Mistral. Each date has its official source URL and the date it was checked.
- **Why do our agents cost so much?** It reads an OpenAI or Anthropic usage/cost export **locally** and shows spend by
  model, cost drivers and the fixes you can make yourself.
- **Is a cheaper model good enough?** You get our public benchmark numbers, a savings estimate for the model pairs we
  measured, and a ready request for a free pilot on your own tasks.

Read-only. No key, no account, no telemetry. The server only sends GET requests to `agenttrustapi.com`, with the
User-Agent `trust-engine-mcp/<version> (mcp|cli|action)` so we can count use per channel; links to agenttrustapi.com
in answers carry `?ref=mcp`, `?ref=cli` or `?ref=action` for the same reason. Nothing else is recorded or sent.

> **Honesty notes.** The retirement data is copied from vendor pages, so follow `source_url` before you act on a date.
> The benchmark is one coding agent on 27 public exercises: it compares the models fairly, but it does not predict
> results on your tasks. Savings estimates exist only for pairs we measured; for any other pair the tool gives no
> number. Nothing is sent for you: `request_pilot` drafts an e-mail that a person sends.

## Connect with one command

Needs Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# Claude Code
claude mcp add --transport stdio trust-engine -- uvx trust-engine-mcp

# Codex CLI
codex mcp add trust-engine -- uvx trust-engine-mcp
```

Claude Desktop, Cursor and other stdio clients use this config (`claude_desktop_config.json`, `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "trust-engine": { "command": "uvx", "args": ["trust-engine-mcp"] }
  }
}
```

Claude Desktop can also install the one-file bundle `trust-engine-mcp-<version>.mcpb` from the GitHub releases.
Without uv, run `pip install trust-engine-mcp` and use the command `trust-engine-mcp`.

## Try this first

> Are we using any deprecated AI models in this repo?

> Use the trust-engine tools. We use claude-sonnet-4-5, gpt-4o and gemini-2.5-flash. Which of them are being
> retired, on which platform, and by when must we re-test? Cite the sources.

> Here is our usage export (CSV). Why are our agents so expensive? List the quick fixes we can do ourselves this week.

> Is Claude Sonnet good enough instead of Opus for coding work? Show the benchmark numbers and what we would save on
> a $4,000 monthly bill. Then tell me how to test this on our own tasks.

## Tools

| Tool | Answers |
|---|---|
| `scan_for_models(path?, texts?)` | Finds every model id from the calendar in the working directory (or in `texts` you pass) as a whole token, with `file:line` for each, plus the `check_my_models` result for those models. `path` must be inside the working directory. Limits: 2,000 files / 5 MB. Skips `.git`, `node_modules`, `venv`, `.venv`, `dist`, `build`, `__pycache__`, lockfiles and binaries. Never opens `.env` or `.env.*` (only `.env.example`, `.env.sample`, `.env.template`); skipped ones are listed by name |
| `check_my_models(models)` | For each model your agent uses: retirement date per platform, days left, the vendor's replacement, the source, and what to do before the date |
| `model_retirements(provider?, model?, before_date?, include_past?, text_models_only?)` | Upcoming retirements and deprecations, filtered, with sources |
| `analyze_usage(csv_text? \| rows?)` | Spend by model, top-model share, models on the retirement calendar, cost drivers, next steps. Runs offline; nothing is uploaded |
| `benchmark_results()` | Public Aider benchmark for Claude Opus, Sonnet and Haiku 5.5: tasks solved, cost per solved task, failed tasks, method and raw data |
| `estimate_switch_savings(monthly_bill_usd, from_model, to_model)` | Measured cost ratio and the arithmetic, for measured pairs only |
| `prepare_pilot_tasks(csv_text?, use_case?)` | The pilot task format (CSV template), how to pick 20-30 tasks, how to mask personal data, and an offline check of a draft |
| `request_pilot(company?, use_case?, models?, task_count?, monthly_bill_usd?, notes?)` | A draft e-mail and `mailto:` link to hello@agenttrustapi.com for a person to send |
| `pricing()` | Published prices (US$, valid from 12 Nov 2026), incl. the fixed-price 7-day test ($1,390), 24/7 watch and Bulgarian review |

What the scan does not see: model ids built at runtime (from environment variables or string concatenation), models
that are not on the calendar, and ids written in a form the calendar does not list. Aliases such as `gpt-4.1` match
the dated snapshot on the calendar (`gpt-4.1-2025-04-14`); the answer says so, because an alias can point to a newer
snapshot.

## Check a repository without an agent

### Command line

```bash
uvx trust-engine-mcp check            # scan the current folder
uvx trust-engine-mcp check src --days 90
```

It prints each model found, its dates, replacement and source, and where it is used. Exit code `1` if a model found
retires within `--days` days (default 60; a date that has already passed counts), otherwise `0`.
`trust-engine-mcp` with no arguments still starts the MCP server.

### GitHub Action

```yaml
# .github/workflows/model-retirements.yml
name: model retirements
on:
  pull_request:
  schedule:
    - cron: "17 6 * * 1"   # Mondays
permissions:
  contents: read
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: DinevDecor/trust-engine-mcp/check@v1
        with:
          days: "60"                  # default 60
          # path: "."                 # default: the whole repository
          # fail-on-retiring: "true"  # default false: warnings only
```

It adds a warning on each line that uses a model retiring within `days` and a table to the job summary. It needs no
token and posts no comments. By default the step never fails; set `fail-on-retiring: "true"` to fail it.

### Usage exports `analyze_usage` reads

Column names are matched loosely; case, spaces and separators are ignored. Only a model column is required.

- model: `model`, `model_name`, `model_id`, `snapshot_id`, `line_item` (e.g. "gpt-4o-2024-08-06, input")
- cost: `cost`, `cost_usd`, `amount`, `amount_value`, `spend`, `total_cost`
- requests: `requests`, `num_model_requests`, `n_requests`, `request_count`
- tokens: `input_tokens`, `prompt_tokens`, `n_context_tokens_total`, `uncached_input_tokens`, `output_tokens`,
  `completion_tokens`, `cache_read_input_tokens`, `input_cached_tokens`, `cache_creation_input_tokens`
- reliability: `errors`, `error_count`, `status`, `status_code`, `retries`
- other: `batch`, `service_tier`, `date`, `start_time`, `bucket_start`

## Data sources

| Data | URL | License |
|---|---|---|
| Model retirement calendar | <https://agenttrustapi.com/retirements.json> ([schema](https://agenttrustapi.com/retirements.schema.json), [Atom](https://agenttrustapi.com/retirements.atom), [page](https://agenttrustapi.com/retirements/)) | CC BY 4.0 |
| Aider model-diet benchmark | <https://agenttrustapi.com/benchmarks/aider-model-diet.json> ([report](https://agenttrustapi.com/reports/model-diet-aider/), [protocol](https://agenttrustapi.com/reports/model-diet-aider/protocol.md), [raw CSV](https://agenttrustapi.com/reports/model-diet-aider/results.csv)) | as on the report |
| Pilot task template | <https://agenttrustapi.com/pilot/tasks_template.csv> | — |

The server caches each file for 6 hours with a 10-second timeout. The package includes a snapshot of both files. If
the site is unreachable, the server uses that snapshot and says so in every answer.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `TRUST_ENGINE_OFFLINE` | `0` | `1`: never fetch; use the bundled snapshot only |
| `TRUST_ENGINE_BASE_URL` | `https://agenttrustapi.com` | data host (for testing against a local copy) |

## Development

```bash
pip install -e ".[dev]"
pytest && ruff check .
```

The tests use fixtures and a fake site. They make no network calls.

## License

Code MIT © 2026 Trust Engine AI. Retirement data CC BY 4.0. Contact: hello@agenttrustapi.com

<!-- Official MCP Registry ownership marker -->
mcp-name: io.github.DinevDecor/trust-engine-mcp
