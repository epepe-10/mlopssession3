# Session 3: Serving the model and the latency budget

Today's scenario. The payment team will call your service on every card payment, and they
gave you two numbers:

- **P95 latency ≤ 200 ms.** 95% of requests must answer faster than this.
- **200 requests per second** at the Black Friday peak.

You will build the service, break it with a load test, find where the time goes, fix it,
and put a price on it.

| Word | Meaning |
|---|---|
| Latency | time between sending a request and getting the answer |
| P95 | the latency that 95% of requests beat. The slowest 5% are what users complain about |
| Throughput | requests served per second (RPS) |
| Latency budget | the 200 ms, split between network, validation, model and the rest |

Project: the fraud model from sessions 1 and 2, now behind an API. Use the same repo.

---

## 0. Setup

You need your session 1 repo with MLflow running and a model registered as
`fraud-detector@staging`. Check both:

```bash
curl -s localhost:5000/health        # prints OK
py score.py                          # prints AUC and AP
```

Copy today's starter files from the course folder `03-serving/` to the root of your repo:

```bash
cp <course>/03-serving/{app.py,test_app.py,locustfile.py} .
```

Add the libraries. `fastapi` and `uvicorn` are the service. `pytest` and `httpx` run the
tests. `locust` is the load tester. The last three are only for development, so they go in
the `dev` group and stay out of a production image:

```bash
uv add fastapi "uvicorn[standard]"
uv add --dev pytest httpx locust
```

You will use three terminals: MLflow (already running), the API, and one for everything else.

---

## 1. Serve the model

Open `app.py` and read it. Four things to notice:

- `lifespan` loads the model **once**, at startup. Loading takes seconds, so doing it on
  every request would make every call slow.
- `Transaction` is a Pydantic model: the contract of the API.
- `score()` turns a request into the DataFrame the model expects. Preprocessing lives in
  the service, so callers never need to know how the model was trained.
- `/health` and `/predict` are the two routes.

Some lines are blank. Complete `TODO 1.1`, `1.2` and `1.3`, then start the service.
`--reload` restarts it when you save a file:

```bash
uv run uvicorn app:app --reload --port 8000
```

```
loaded fraud-detector@staging = v1
```

If you see `NameError: name '___' is not defined`, a TODO is still open. If `/predict`
answers `500` and the log mentions `Incompatible input types for column online`, look at
TODO 1.2 and at `predict.py` from session 2.

Open http://localhost:8000/docs. FastAPI generated an interactive page from your Pydantic
models. Try `/health`, then `/predict` with "Try it out".

```bash
curl localhost:8000/health
```

```json
{"status":"ok","model_version":"1"}
```

**TODO 1.4:** score a transaction from the terminal. Write the `curl` command yourself:
a POST, a JSON body, and a `content-type` header. The body:

```json
{"amount": 42.5, "hour": 14, "distance_km": 3.2, "merchant": "grocery", "online": false, "tx_last_24h": 1}
```

```json
{"fraud_proba":0.009,"model_version":"1","model_ms":4.2}
```

Now a suspicious one: 900 at 3 am, 80 km away, `crypto`, online, 6 transactions in the
last 24 hours. What probability do you get? Every answer says `model_version`: when a
customer disputes a decision, you know which model made it.

---

## 2. Reject bad input

Send nonsense:

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"amount": 42.5, "hour": 99, "distance_km": 3.2, "merchant": "casino", "online": false, "tx_last_24h": 1}'
```

Hour 99 and a merchant that does not exist, and you still get a score with status 200.
The model's encoder silently ignores unknown merchants. A confident wrong answer is worse
than an error, because nobody notices it.

```
HTTP 422
{"detail":[{"loc":["body","hour"],"msg":"Input should be less than or equal to 23", ...
```

`422` means "your request is well-formed but invalid": the caller's fault. A `500` would
mean ours. Telling them apart is what you read first at 3 am.

---

## 3. Test it

A service without tests breaks quietly. Open `test_app.py`. The fixture starts the app
with `TestClient`, which runs the startup too, so these are integration tests against the
real model (MLflow must be running).

**TODO 3.1 to 3.3:** write the three missing tests. Then:

```bash
uv run pytest -q
```

```
4 passed, 1 failed
```

Question: these tests need MLflow running. What does that mean for a CI pipeline?

---

## 4. Load test

Your service answers one request in a few milliseconds. The real question is what happens
when 200 shoppers hit it at once.

Locust simulates shoppers. Open `locustfile.py`. `ROWS` holds real transactions from
your `data/` folder. One `SingleUser` is one shopper who pays about once per second, so
**N users is roughly N requests per second**.


Restart the service **without** `--reload` and without the per-request log lines, which
cost time:

```bash
uv run uvicorn app:app --port 8000 --no-access-log
```

In the other terminal, run 50 users:

```bash
uv run locust -f locustfile.py SingleUser|BatchUser --headless -u 50 -r 10 -t 30s --host http://localhost:8000 --only-summary
```

At the end, read the `Response time percentiles` table. The column `50%` is the median and
`95%` is P95. `Failures` must be 0. Repeat for 150 and 250 users and fill in:

| Users (about req/s) | P50 ms | P95 ms | Within 200 ms? |
|---|---|---|---|
| 50 | | | |
| 150 | | | |
| 250 | | | |

The shape matters more than your exact numbers (a laptop is not a server). Expect P95 to
stay low, then climb fast. Past the service's capacity requests queue up, like a checkout
with one till.

**TODO 4.3:** does your service survive Black Friday (200 req/s)?

Tip: `uv run locust -f locustfile.py SingleUser` without `--headless` serves live charts at
http://localhost:8089.

