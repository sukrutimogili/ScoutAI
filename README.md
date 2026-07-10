# ScoutAI

**Evidence-driven AI hiring intelligence system.**

> Status: Under active development — S17 complete.

ScoutAI takes a job description and a batch of résumés and produces evidence-backed,
confidence-scored, human-reviewable shortlisting recommendations.
It never makes a final hiring decision — every recommendation requires human approval.

---

## Architecture

Orchestrated by **LangGraph** with:
- A deterministic outer graph for compliance gates, fairness probing, and human review.
- A bounded ReAct agent loop (`candidate_agent`) for per-candidate evidence extraction,
  capability assessment, and confidence-driven clarification.
- A Streamlit UI (`ui/app.py`) with a monochrome, editorial-grid design system.

See [`spec_ScoutAI.md`](spec_ScoutAI.md) for the full specification and ADRs.

### Key Design Decisions

| ADR | Decision |
|---|---|
| ADR-1 | Confidence is a 4-value enum (`unknown`/`low`/`medium`/`high`), not a float |
| ADR-3 | Interview clarification capped at exactly 1 round, enforced as an autonomy budget |
| ADR-4 | LangGraph over CrewAI — deterministic replay and step-level audit |
| ADR-7 | Bias checked in two tiers: cheap `leakage_flag` at intake, expensive `run_fairness_probe` on shortlist only |
| ADR-8 | Audit logging is infrastructure, not a callable capability — auto-appended to trajectory |
| ADR-9 | Per-candidate evaluation is one bounded agentic tool-calling loop, not a chain of fixed edges |
| ADR-10 | `ask_candidate` is agent-only, never recruiter-facing |

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) or pip

### Install

```bash
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Fill in at least one provider API key
```

### Run (CLI)

```bash
hiring-run --jd path/to/jd.txt --resumes path/to/resumes/ --config config.yaml
```

### Run (Streamlit UI)

```bash
streamlit run ui/app.py
```

---

## Configuration Reference

All configurable parameters live in `config.yaml`. Key sections:

### `model_roles`

```yaml
model_roles:
  fast_structured:
    primary: "gemini/gemini-2.0-flash-exp"
    fallback: "groq/llama-3.3-70b-versatile"
    fallback_2: "groq/qwen-2.5-32b"
  high_context:
    primary: "gemini/gemini-2.0-flash-exp"
    fallback: "groq/llama-3.3-70b-versatile"
```

- `fast_structured`: Used by high-volume capabilities (screen_resume, extract_evidence, assess_capabilities, etc.)
- `high_context`: Used by low-volume, context-heavy capabilities (extract_role_requirements, generate_rubric, compose_decision_summary)

### `graph`

```yaml
graph:
  recursion_limit: 40
  max_iterations: 8
```

- `recursion_limit`: Maximum outer graph steps before forced termination
- `max_iterations`: Maximum tool calls per candidate before force-finalize with `hold`

### `rubric`

```yaml
rubric:
  config_version: "1.0.0"
  cache_ttl_seconds: 3600
```

### `security`

```yaml
security:
  sensitive_attributes: ["name", "age", "gender", "photo", "address", "phone", "email"]
  secret_patterns: ["api_key", "token", "password", "secret"]
```

### `rate_limiting`

```yaml
rate_limiting:
  max_requests_per_minute: 30
  max_retries: 3
  retry_backoff_ms: [250, 500, 1000]
  circuit_breaker_threshold: 5
  circuit_breaker_open_seconds: 60
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes (if using Gemini) | Google AI / Vertex AI API key |
| `GROQ_API_KEY` | Yes (if using Groq) | Groq API key for Llama/Qwen models |

---

## How to Replay a Run from the Audit Log

Every node execution and agent tool-call is automatically logged to the `trajectory` list in the
graph state (ADR-8). Each `TrajectoryEntry` contains:

- `node`: The graph node that executed
- `tool_used`: The agent tool called (None for fixed-node capabilities)
- `timestamp`: ISO 8601 timestamp
- `input_hash` / `output_hash`: SHA-256 of validated payloads
- `latency_ms`: Execution time in milliseconds
- `model`: The actual provider/model used
- `model_role`: `fast_structured` or `high_context`
- `prompt_version` / `tool_version` / `schema_version`: Version tracking
- `status`: `success`, `retried`, `failed_closed`, or `escalated`

### Replay Steps

1. **Export the run state** to JSON:
   ```python
   from scoutai.graph.export import export_to_json
   state = get_state(thread_id)  # from runtime.session
   print(export_to_json(state))
   ```

2. **Inspect the trajectory** to trace every decision:
   ```python
   for entry in state["trajectory"]:
       print(f"{entry['timestamp']} | {entry['node']}.{entry['tool_used']} "
             f"→ {entry['status']} ({entry['latency_ms']}ms)")
   ```

3. **Verify evidence integrity** by checking that every `evidence_ref` in the shortlist
   traces back to a real `TrajectoryEntry` output.

4. **Re-run with the same inputs** by passing the original JD and résumés through
   `runtime.session.start_run()` — the deterministic outer graph ensures identical
   routing for identical inputs.

---

## Development

```bash
# Run all tests
pytest

# Run specific milestone tests
pytest tests/test_s15_e2e.py -v
pytest tests/test_s16_evaluation.py -v

# Lint
ruff check scoutai tests

# Type check
mypy scoutai
```

### Test Structure

| File | Milestone | What it covers |
|---|---|---|
| `tests/test_s0_scaffold.py` | S0 | Repo structure, imports, config loading |
| `tests/test_s1_schemas.py` | S1 | Pydantic schema round-trips, enum boundaries |
| `tests/test_s2_model_router.py` | S2 | Model routing, retry/backoff, circuit breaker |
| `tests/test_s3_screen_resume.py` | S3 | Sanitization, injection/leakage flags |
| `tests/test_s4_role_requirements.py` | S4 | Role profile extraction, rubric generation, caching |
| `tests/test_s5_evidence.py` | S5 | Evidence extraction, capability assessment, verification |
| `tests/test_s6_interview.py` | S6 | Interview questions, ask_candidate, reevaluate, finalize |
| `tests/test_s7_candidate_agent.py` | S7 | Agent harness, tool allowlist, budget enforcement |
| `tests/test_s8_outer_graph.py` | S8 | Outer graph wiring, per-candidate loop, edge routing |
| `tests/test_s9_fairness.py` | S9 | Fairness probe, bias reports, decision summary |
| `tests/test_s10_scheduling.py` | S10 | Calendar availability, interview proposal |
| `tests/test_s11_human_review.py` | S11 | Human review node, 4 actions, routing |
| `tests/test_s12_audit.py` | S12 | Audit log, trajectory auto-append, secret redaction |
| `tests/test_s13_metrics.py` | S13 | Operational logging, metrics counters |
| `tests/test_export.py` | S13.5 | PDF/CSV/JSON export formatting |
| `tests/test_runtime_session.py` | S13.6 | Session management, start/get/resume |
| `tests/test_s15_e2e.py` | S15 | End-to-end integration with synthetic résumés |
| `tests/test_s16_evaluation.py` | S16 | Evaluation harness against §11 metrics |

---

## Directory Layout

```
scoutai/
├── schemas/        # Pydantic data models (§5)
├── graph/          # LangGraph outer graph + nodes + export
├── agent/          # candidate_agent bounded ReAct loop (ADR-9)
├── capabilities/   # All tool implementations (§4)
├── audit/          # Audit log infrastructure (ADR-8)
├── runtime/        # Session management (S13.6)
├── config.py       # Configuration loader
└── cli.py          # CLI entry point (S14)
ui/
├── app.py          # Streamlit entry point
├── styles.py       # Design tokens and CSS injection
├── components.py   # Reusable UI components
├── mock_data.py    # Mock data matching backend schemas
└── screens/        # 8 screens matching the UX flow
tests/
├── fixtures/       # Synthetic JDs, résumés, adversarial inputs
config.yaml         # All configurable parameters
.env.example        # Required environment variables
spec_ScoutAI.md     # Full technical specification (source of truth)
docs/
└── ux_flow.md      # UX flow documentation