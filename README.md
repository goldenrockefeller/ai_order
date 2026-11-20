# Raft Order Agent

A small Python agent that fetches unstructured customer orders, extracts each
order with a schema-constrained language model, validates the extracted values
against the source text, and returns normalized JSON.

## Architecture

```text
Customer API
    |
    v
Validated raw orders
    |
    v
ChatOpenRouter (openai/gpt-oss-120b:exacto)
    |  strict JSON Schema output
    v
Grounding and field validation
    |
    v
Normalization (state code and integer cents)
    |
    v
Whole-record quorum vote
    |
    v
SQLite orders
    |
    v
Bound-parameter query compiler -> clean JSON

Natural-language request -> Query Agent -> normalized whole-filter quorum vote
                         -> typed filter (never SQL) -> SQLite
```

The models only extract fields and classify query meaning. Python validates the
API envelope, rejects missing or ungrounded values, performs all normalization,
and rejects unsupported query conditions instead of silently dropping them.

LangGraph runs the fetch, extract, store, and query nodes in order. Its state
holds one immutable API response and one parsed SQLite snapshot, so multiple
requests in the same invocation do not refetch or re-extract. The `QueryAgent`
returns `logic: all|any` plus typed field/operator/value conditions. Python
parses field-specific values at execution time and combines fixed SQL fragments
with bound parameters; the model never supplies SQL.

Both agents use the same sequential quorum function. Each response must pass
the existing normalization boundary before it can vote, and agreement is exact
equality of the entire immutable `Order` or `QueryFilter`. Invalid responses
and model failures are rejected votes; the configured quorum is never reduced.
Order no-quorum skips that source record and marks every snapshot result
incomplete. Query no-quorum rejects the request before the database call.

## Requirements

- Python 3.11 or newer
- An OpenRouter API key

## Setup

From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open the newly created local `.env` file and set your key:

```dotenv
OPENROUTER_API_KEY=sk-or-your-key-here
```

## Run

Start the dummy customer API in the first terminal:

```powershell
.\.venv\Scripts\python.exe dummy_customer_api.py
```

The dummy API explicitly serves `fixtures/raw_orders.py`; production database
creation remains empty unless extracted orders are supplied. Normalized labels
used by automated tests live separately under `tests/fixtures/`.

Run the agent in a second terminal:

```powershell
.\.venv\Scripts\python.exe main.py
```

With no arguments, the agent runs the sample Ohio/over-$500 request. It fetches
and validates the raw orders once, makes one structured vote per order,
stores the valid records, and prints the matching records as JSON. If any order
cannot be extracted, the remaining records are still queried and the output
contains `"incomplete": true` and `"failedExtractions"`.

Pass quoted requests as arguments to answer several questions from the same
snapshot:

```powershell
.\.venv\Scripts\python.exe main.py `
  "Ohio orders over $500" `
  "Orders for Sarah or shipped to Washington"
```

The default remains 1-of-1. Configure the same explicit vote count and quorum
for both agents with, for example:

```powershell
.\.venv\Scripts\python.exe main.py --attempts 3 --quorum 2
```

For quorum configurations, each vote uses a different fixed, curated system
prompt with different examples. Model responses are still strictly normalized
before whole-record or whole-filter voting. Up to five prompt variants are
available for each agent, keeping repeated runs reproducible at temperature 0.

Run the small deterministic labeled comparison (no API key or server needed):

```powershell
.\.venv\Scripts\python.exe voting_demo.py
```

It runs the same order and query cases under 1-of-1, 2-of-3, and 4-of-5 and
reports accepted labels, skipped/rejected labels, disagreements, rejected vote
counts, and monotonic elapsed time for the sequential model-call boundaries.

## Test

Install the development requirements and run the correctness suite:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -v
```

Run the coverage gate:

```powershell
.\.venv\Scripts\python.exe -m coverage run -m unittest discover
.\.venv\Scripts\python.exe -m coverage report
```

The configured minimum coverage is 80%.
