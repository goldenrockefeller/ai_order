# Manual Live-Test Guide

This guide verifies the Raft Order Agent from a clean local Python environment
through the real OpenRouter model. It assumes no prior knowledge of the
project. Follow the steps in order and copy each command exactly.

The guide is written for macOS or Linux. It uses the repository at:

```text
/Users/goldenrockefeller/Documents/workspace/raft_interview
```

If your copy is somewhere else, change only the path in the first `cd`
command.

## What this test covers

You will:

1. Create an isolated Python virtual environment.
2. configure an OpenRouter API key without displaying it or saving it in shell
   history;
3. run all 35 automated tests and the coverage report;
4. run the deterministic 1-of-1, 2-of-3, and 4-of-5 voting comparison;
5. start the local customer-order API;
6. run the real end-to-end OpenRouter pipeline;
7. temporarily expand the API from five to ten unstructured orders;
8. run several positive, boundary, multi-query, and unsupported-query cases;
9. restore the original five orders; and
10. optionally run a real 2-of-3 quorum test.

OpenRouter calls can consume account credit. With the original five orders:

- One default 1-of-1 query makes 6 model calls: 5 order extractions plus 1
  query interpretation.
- One 2-of-3 query makes 18 model calls.
- The expanded ten-order, six-query command in this guide makes 16 model calls
  because all six queries reuse the same ten-order snapshot.

Agreement is not proof of correctness. Quorum only checks whether independently
normalized results are exactly equal.

## 1. Open Terminal 1 and enter the repository

Open the Terminal application. Copy and paste:

```bash
cd "/Users/goldenrockefeller/Documents/workspace/raft_interview"
pwd
test -f main.py && echo "Repository found"
```

Expected final line:

```text
Repository found
```

If you do not see that line, stop. The `cd` path is incorrect.

## 2. Create and configure the virtual environment

Run:

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python --version
```

The install is successful when its final output contains `Successfully
installed` or reports that every requirement is already satisfied.

This project always uses `.venv/bin/python` below, so activating the virtual
environment is unnecessary.

## 3. Configure the OpenRouter API key safely

First, check whether this repository is already configured. This command does
not print the key:

```bash
.venv/bin/python -c 'from dotenv import load_dotenv; import os; load_dotenv(); print("OpenRouter key configured:", bool(os.getenv("OPENROUTER_API_KEY")))'
```

If the result is `True`, skip to Step 4. If it is `False`, get an API key from
your OpenRouter account and copy and paste this command:

```bash
.venv/bin/python -c 'from getpass import getpass; from pathlib import Path; path = Path(".env"); key = getpass("Paste OPENROUTER_API_KEY and press Enter: ").strip(); assert key, "Key cannot be empty"; old = path.read_text().splitlines() if path.exists() else []; kept = [line for line in old if not line.lstrip().startswith("OPENROUTER_API_KEY=")]; path.write_text("\n".join([*kept, "OPENROUTER_API_KEY=" + key]) + "\n")'
```

Paste the key at the prompt and press Enter. The key will not appear on screen
or in the command history.

Now repeat the safe check:

```bash
.venv/bin/python -c 'from dotenv import load_dotenv; import os; load_dotenv(); print("OpenRouter key configured:", bool(os.getenv("OPENROUTER_API_KEY")))'
```

Expected output:

```text
OpenRouter key configured: True
```

Never paste the contents of `.env` into a test report. The file is ignored by
Git.

## 4. Create a folder for recorded output

The following folder is created on the Desktop, outside the repository:

```bash
RESULTS_DIR="$HOME/Desktop/raft-manual-test-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"
printf '%s\n' "$RESULTS_DIR" | tee "$HOME/Desktop/raft-manual-test-latest.txt"
```

Leave Terminal 1 open. Later commands use its `RESULTS_DIR` variable.

## 5. Run the full automated suite and coverage report

Copy and paste this entire block into Terminal 1:

```bash
.venv/bin/python -m coverage erase
.venv/bin/python -m coverage run -m unittest discover -v 2>&1 | tee "$RESULTS_DIR/01-tests.txt"
.venv/bin/python -m coverage report 2>&1 | tee "$RESULTS_DIR/02-coverage.txt"
```

Success criteria:

- `Ran 35 tests`
- `OK`
- `TOTAL ... 87%`
- `voting.py ... 100%`

Warnings about deliberately rejected votes and forced failures are expected;
the tests create those conditions intentionally.

## 6. Run the deterministic voting comparison

This step makes no network calls and needs no running API server:

```bash
.venv/bin/python voting_demo.py 2>&1 | tee "$RESULTS_DIR/03-voting-demo.json"
```

Success criteria:

- `1-of-1` accepts all three labeled cases.
- `2-of-3` accepts `order_consensus` and `query_with_invalid_vote`, while
  skipping `order_split` and rejecting one invalid vote.
- `4-of-5` has the same accept/skip pattern and rejects one invalid vote.

The elapsed milliseconds vary by machine and are not pass/fail criteria.

## 7. Start the local customer API in Terminal 1

Run:

```bash
.venv/bin/python dummy_customer_api.py 2>&1 | tee "$RESULTS_DIR/04-customer-api.txt"
```

The server is ready when Terminal 1 shows:

```text
Running on http://127.0.0.1:5001
```

Leave this command running. Terminal 1 will not return to a prompt until the
server is stopped.

## 8. Open Terminal 2 and verify the API

Open a second Terminal window. Copy and paste:

```bash
cd "/Users/goldenrockefeller/Documents/workspace/raft_interview"
RESULTS_DIR="$(cat "$HOME/Desktop/raft-manual-test-latest.txt")"
.venv/bin/python -c 'import json, urllib.request; data = json.load(urllib.request.urlopen("http://localhost:5001/api/orders")); print("status:", data["status"]); print("order count:", len(data["raw_orders"])); print(*data["raw_orders"], sep="\n- ")' 2>&1 | tee "$RESULTS_DIR/05-original-api-check.txt"
```

Success criteria:

```text
status: ok
order count: 5
```

The order sequence may change because the dummy API deliberately shuffles the
orders.

## 9. Run the real default OpenRouter pipeline

In Terminal 2, run:

```bash
.venv/bin/python main.py 2>&1 | tee "$RESULTS_DIR/06-live-default.txt"
```

This can take a minute. Do not stop it while model calls are in progress.

Success criteria:

- The API log says `received_count=5`.
- Six OpenRouter requests return `HTTP/1.1 200 OK`.
- Five order votes and one query vote say `outcome=accepted`.
- The final JSON contains order IDs `1001`, `1003`, and `1005`.
- The final JSON contains `"incomplete": false`.

The expected records are:

| Order | Buyer | State | Total |
|---|---|---:|---:|
| 1001 | John Davis | OH | 742.10 |
| 1003 | Mike Turner | OH | 1299.99 |
| 1005 | Chris Myers | OH | 512.00 |

Durations will vary.

## 10. Temporarily add five more orders

First stop the API in Terminal 1 by clicking that window and pressing
`Control-C` once. Wait for the prompt to return.

In Terminal 2, make a safety backup and replace the fixture with the expanded
ten-order set:

```bash
test ! -e fixtures/raw_orders.py.manual-backup || { echo "Backup already exists; stop and inspect it before continuing."; false; }
cp fixtures/raw_orders.py fixtures/raw_orders.py.manual-backup
cat > fixtures/raw_orders.py <<'PY'
"""Unstructured orders served only by the local dummy customer API."""

RAW_ORDERS = [
    "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable",
    "Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55, Items: headphones",
    "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc, mouse",
    "Order 1004: Buyer=Rachel Kim, Location=Seattle, WA, Total=$89.50, Items: coffee maker",
    "Order 1005: Buyer=Chris Myers, Location=Cincinnati, OH, Total=$512.00, Items: monitor, desk lamp",
    "Ref 1006 | Buyer: Ana Gómez | Ship-to: Albany, NY | Final Total: $500.00 | Items: keyboard",
    "Customer Robert O'Brien; Order 1007; Destination Los Angeles, CA; Total $2,500.00; Items: workstation",
    "Order 1008 -- Buyer Priya Shah -- Location Toledo, OH -- Total $500.00 -- Items: office chair",
    "Order ID 1009; Buyer Lee Wong; Location Dallas, TX; Total $875.25; Items: tablet",
    "Buyer=Morgan Reed, Location=Tacoma, WA, Items=printer, Total=$1,050.75, Order=1010",
]
PY
.venv/bin/python -m py_compile fixtures/raw_orders.py
echo "Expanded fixture is valid Python"
```

Expected final line:

```text
Expanded fixture is valid Python
```

## 11. Restart the API with ten orders

Return to Terminal 1 and run:

```bash
.venv/bin/python dummy_customer_api.py 2>&1 | tee "$RESULTS_DIR/07-expanded-customer-api.txt"
```

Leave it running. In Terminal 2, confirm the expanded count:

```bash
.venv/bin/python -c 'import json, urllib.request; data = json.load(urllib.request.urlopen("http://localhost:5001/api/orders")); print("status:", data["status"]); print("order count:", len(data["raw_orders"]))' 2>&1 | tee "$RESULTS_DIR/08-expanded-api-check.txt"
```

Expected output:

```text
status: ok
order count: 10
```

## 12. Run six live queries against one expanded snapshot

In Terminal 2, copy this entire command:

```bash
.venv/bin/python main.py \
  'Ohio orders over $500' \
  'Ohio orders at least $500' \
  'Orders where the buyer name contains Ana' \
  'Orders shipped to Texas or Washington' \
  'Orders with a total under $200' \
  'Order with order ID 1007' \
  2>&1 | tee "$RESULTS_DIR/09-live-expanded-queries.txt"
```

The pipeline should fetch and extract the ten orders only once. It then runs
six query interpretations against that one SQLite snapshot, for 16 total model
calls in the default 1-of-1 configuration.

Expected logical results:

| Request | Expected order IDs |
|---|---|
| Ohio orders over $500 | 1001, 1003, 1005 |
| Ohio orders at least $500 | 1001, 1003, 1005, 1008 |
| Buyer name contains Ana | 1006 |
| Texas or Washington | 1002, 1004, 1009, 1010 |
| Total under $200 | 1002, 1004 |
| Order ID 1007 | 1007 |

Every result should contain `"incomplete": false`. The `$500.00` order 1008
is intentionally excluded by `over` (`>`) and included by `at least` (`>=`).
Order 1010 verifies comma-formatted dollar normalization, while orders 1006
and 1007 exercise non-ASCII and punctuation-bearing buyer names.

Because this is a real model test, a different result is evidence to record,
not something to edit out. Keep the complete log and note which request
differed.

## 13. Verify unsupported intent fails closed

The database contract has no `items` field, so this request must not be
silently weakened into a different query:

```bash
.venv/bin/python main.py "Orders containing a laptop item" 2>&1 | tee "$RESULTS_DIR/10-live-unsupported-query.txt"
```

Success means the command reports an unsupported query condition and exits
without printing successful order-result JSON. The exact model wording may
vary. This demonstrates that unsupported intent is rejected rather than sent
to SQLite as a guess.

## 14. Restore the original five-order fixture

Stop the server in Terminal 1 with `Control-C` and wait for its prompt.

In Terminal 2, restore the exact original file:

```bash
test -f fixtures/raw_orders.py.manual-backup
mv fixtures/raw_orders.py.manual-backup fixtures/raw_orders.py
.venv/bin/python -m py_compile fixtures/raw_orders.py
git diff --exit-code -- fixtures/raw_orders.py && echo "Original fixture restored"
```

Expected final line:

```text
Original fixture restored
```

Restart the server in Terminal 1:

```bash
.venv/bin/python dummy_customer_api.py 2>&1 | tee "$RESULTS_DIR/11-restored-customer-api.txt"
```

Confirm from Terminal 2:

```bash
.venv/bin/python -c 'import json, urllib.request; data = json.load(urllib.request.urlopen("http://localhost:5001/api/orders")); print("restored order count:", len(data["raw_orders"]))'
```

Expected output:

```text
restored order count: 5
```

## 15. Optional: run a real 2-of-3 quorum pipeline

This step makes 18 sequential model calls and can take several minutes. It may
consume noticeably more OpenRouter credit than the default test.

In Terminal 2, run:

```bash
.venv/bin/python main.py --attempts 3 --quorum 2 2>&1 | tee "$RESULTS_DIR/12-live-2-of-3.txt"
```

Success criteria:

- Each of five order logs reports 3 attempted votes and an accepted outcome.
- The query log reports an accepted outcome.
- The final IDs are 1001, 1003, and 1005.
- `"incomplete": false` is present.

A log may report `disagreement=True` even when a value reaches quorum. That is
expected behavior: disagreement is surfaced, while two exactly equal normalized
values are sufficient for 2-of-3 acceptance. If an order does not reach quorum,
it is skipped and the final result must say `"incomplete": true`. If the query
does not reach quorum, the request must fail before SQLite is queried.

## 16. Stop the API and perform final checks

Stop Terminal 1 with `Control-C`.

In Terminal 2, run:

```bash
git status --short
printf '\nRecorded outputs:\n'
find "$RESULTS_DIR" -maxdepth 1 -type f -print | sort
printf '\nResults folder:\n%s\n' "$RESULTS_DIR"
```

`git status --short` must not list `fixtures/raw_orders.py` or
`fixtures/raw_orders.py.manual-backup`. If this new guide has not been committed
yet, `?? MANUAL_LIVE_TEST_GUIDE.md` is expected; otherwise the status should be
empty. The remaining lines list the saved evidence files. Keep the entire
results folder when sharing test evidence, but do not add `.env` or an API key
to it.

## Troubleshooting

### `python3: command not found`

Install Python 3.11 or newer, close and reopen Terminal, and return to Step 1.

### `No module named ...`

Repeat Step 2. Continue using `.venv/bin/python`, not bare `python`.

### `OpenRouter key configured: False`

Repeat Step 3. Do not include quotes or spaces around the key when prompted.

### `Couldn't connect to server` or `Connection refused`

The customer API is not running. Start it in Terminal 1, wait for the
`127.0.0.1:5001` message, and retry from Terminal 2.

### Port 5001 is already in use

An older API process is still running. Find it with:

```bash
lsof -nP -iTCP:5001 -sTCP:LISTEN
```

Stop the Terminal window that owns that process with `Control-C`. Do not kill
an unfamiliar process without confirming what it is.

### OpenRouter returns 401 or 403

The key is missing, invalid, or lacks access. Reconfigure it in Step 3 and
check the OpenRouter account.

### OpenRouter returns 402 or a credit error

Add credit or use an account with available credit, then repeat only the
failed command. Previously recorded deterministic tests remain valid.

### A live model call times out or returns 5xx

Record the complete output and retry the same command once. Do not change the
expected result to hide a transient provider failure.

### The temporary fixture was not restored

If `fixtures/raw_orders.py.manual-backup` exists, stop the API and run:

```bash
mv fixtures/raw_orders.py.manual-backup fixtures/raw_orders.py
git diff --exit-code -- fixtures/raw_orders.py && echo "Original fixture restored"
```

Do not delete the backup before confirming that `git diff --exit-code` succeeds.
