# trust-engine-mcp

Free MCP server from [Trust Engine AI](https://agenttrustapi.com). It answers questions agents otherwise search the web for:

- **Is a model we use being retired, and when?** Verified dates for Anthropic, OpenAI, Google Gemini API and Vertex AI,
  AWS Bedrock and Mistral. Each date has its official source URL and the date it was checked.
- **Why do our agents cost so much?** It reads an OpenAI or Anthropic usage/cost export **locally** and shows spend by
  model, cost drivers and the fixes you can make yourself.
- **Is a cheaper model good enough?** You get our public benchmark numbers, a savings estimate for the model pairs we
  measured, and a ready request for a free pilot on your own tasks.

Read-only. No key, no account, no telemetry. The server only sends GET requests to `agenttrustapi.com`.

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

> Use the trust-engine tools. We use claude-sonnet-4-5, gpt-4o and gemini-2.5-flash. Which of them are being
> retired, on which platform, and by when must we re-test? Cite the sources.

> Here is our usage export (CSV). Why are our agents so expensive? List the quick fixes we can do ourselves this week.

> Is Claude Sonnet good enough instead of Opus for coding work? Show the benchmark numbers and what we would save on
> a $4,000 monthly bill. Then tell me how to test this on our own tasks.

## Tools

| Tool | Answers |
|---|---|
| `check_my_models(models)` | For each model your agent uses: retirement date per platform, days left, the vendor's replacement, the source, and the next step ("re-test before ...") |
| `model_retirements(provider?, model?, before_date?, include_past?, text_models_only?)` | Upcoming retirements and deprecations, filtered, with sources |
| `analyze_usage(csv_text? \| rows?)` | Spend by model, top-model share, models on the retirement calendar, cost drivers, next steps. Runs offline; nothing is uploaded |
| `benchmark_results()` | Public Aider benchmark for Claude Opus, Sonnet and Haiku 5.5: tasks solved, cost per solved task, failed tasks, method and raw data |
| `estimate_switch_savings(monthly_bill_usd, from_model, to_model)` | Measured cost ratio and the arithmetic, for measured pairs only |
| `prepare_pilot_tasks(csv_text?, use_case?)` | The pilot task format (CSV template), how to pick 20-30 tasks, how to mask personal data, and an offline check of a draft |
| `request_pilot(company?, use_case?, models?, task_count?, monthly_bill_usd?, notes?)` | A draft e-mail and `mailto:` link to hello@agenttrustapi.com for a person to send |
| `pricing()` | Published prices (US$, valid from 12 Nov 2026), incl. the fixed-price 7-day test ($1,390), 24/7 watch and Bulgarian review |

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
