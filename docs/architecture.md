# Architecture – Talk to Data

## Design Philosophy

Talk to Data separates **data computation** from **language generation**.
Gemini is deliberately used only for two narrow tasks: classifying the
user's intent (a lightweight call) and synthesising a plain-English
narrative from pre-computed aggregated results. All number-crunching
happens locally in pandas — this design means the system produces
deterministic, verifiable answers while Gemini adds readability.

## Why Raw Data Never Enters the Gemini Prompt

Every prompt sent to Gemini contains **only aggregated summaries**
(e.g. `{"South": 120000, "North": 340000}`), never individual rows.
This serves two purposes:

1. **Privacy** – even if a dataset contains sensitive information that
   the sanitiser didn't catch by column name, raw values never leave
   the server boundary for a third-party API call.
2. **Token efficiency** – sending thousands of rows would be slow and
   expensive. Aggregated results are typically under 200 tokens,
   keeping response times well under 5 seconds.

## The Semantic Layer

Business teams frequently use different names for the same concept:
"revenue", "sales", and "income" all mean the same metric. The
semantic layer maintains a curated dictionary of canonical metrics
with their aliases and default aggregation functions. When a user says
"sales", the system resolves it to the canonical `revenue` metric
with `sum` aggregation, ensuring consistent results regardless of
phrasing. This also makes it easy for organisations to extend the
system for their own terminology by editing one Python dictionary.

## Session Model

The application uses a stateless per-request design with a lightweight
in-memory session store (a Python dictionary keyed by UUID). Each
uploaded dataset is assigned a session ID, and all subsequent queries
reference that ID. This avoids database overhead, simplifies
deployment (no external state store), and means each request is
self-contained. For production use, this would be replaced with
Redis or a similar ephemeral store with TTL-based expiry.

## Data Flow

```
User question
  → Intent Parser (Gemini mini-call: classify into CHANGE / COMPARE / BREAKDOWN / SUMMARY)
  → Semantic Layer (resolve "sales" → revenue, aggregation = sum)
  → Data Engine (pandas aggregation on the session DataFrame)
  → Gemini API (language synthesis on aggregated results only)
  → Response (answer + chart_type + chart_data + source_ref + confidence_note)
```

Each step is a separate module in `backend/services/`, making the
pipeline testable and easy to extend with new intent types.
