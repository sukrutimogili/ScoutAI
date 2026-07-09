# ScoutAI

**Evidence-driven AI hiring intelligence system.**

> Status: Under active development — S0 scaffold complete.

ScoutAI takes a job description and a batch of résumés and produces evidence-backed,
confidence-scored, human-reviewable shortlisting recommendations.
It never makes a final hiring decision — every recommendation requires human approval.

---

## Architecture

Orchestrated by **LangGraph** with:
- A deterministic outer graph for compliance gates, fairness probing, and human review.
- A bounded ReAct agent loop (`candidate_agent`) for per-candidate evidence extraction,
  capability assessment, and confidence-driven clarification.

See [`spec_ScoutAI.md`](spec_ScoutAI.md) for the full specification and ADRs.

---

## Setup

> Full setup instructions will be added in S17. Quick-start below.

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
# Fill in GOOGLE_API_KEY and/or GROQ_API_KEY
```

### Run

```bash
hiring-run --jd path/to/jd.txt --resumes path/to/resumes/ --config config.yaml
```

---

## Development

```bash
# Run tests
pytest

# Lint
ruff check scoutai tests

# Type check
mypy scoutai
```

---

## Directory Layout

```
scoutai/
├── schemas/        # Pydantic data models (§5)
├── graph/          # LangGraph outer graph + nodes
├── agent/          # candidate_agent bounded ReAct loop (ADR-9)
├── capabilities/   # All tool implementations (§4)
├── audit/          # Audit log infrastructure (ADR-8)
├── config.py       # Configuration loader
└── cli.py          # CLI entry point (S14)
tests/
├── fixtures/       # Synthetic JDs, résumés, adversarial inputs
config.yaml         # All configurable parameters
.env.example        # Required environment variables
spec_ScoutAI.md     # Full technical specification (source of truth)
```
