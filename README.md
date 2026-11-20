# Raft Order Agent

A small Python agent that fetches unstructured customer orders, extracts each
order with a schema-constrained language model, validates the extracted values
against the source text, and returns normalized JSON.

## Architecture

```text
Customer API
    |
    v
Validated snapshot
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
JSON output
```

The model only extracts fields. Python validates the API envelope, rejects
missing or ungrounded values, and performs all normalization.

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

The agent fetches one snapshot, makes one structured model call per order, and
prints the normalized records as JSON.

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
