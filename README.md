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
SQLite orders
    ^
    | compact typed filter intent (never model-written SQL)
Natural-language request -> Query Agent -> validation and normalization
```

The models only extract fields and classify query meaning. Python validates the
API envelope, rejects missing or ungrounded values, performs all normalization,
and rejects unsupported query conditions instead of silently dropping them.

Checkpoint 4 is implemented as a standalone `QueryAgent`. Its contract is
`logic: all|any` plus typed field/operator/value conditions. This supports
ordinary compound filters without reproducing SQL as a second language. Model
values are preserved verbatim, and strict `over` (`gt`) remains distinct from
inclusive `at least` (`gte`). Pipeline wiring is left for Checkpoint 5.

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

Run the agent in a second terminal:

```powershell
.\.venv\Scripts\python.exe main.py
```

The agent fetches and validates the raw orders once, makes one structured model
call per order, and prints the normalized records as JSON.

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
