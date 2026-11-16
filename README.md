# Embedding Batch Gateway

A small FastAPI service that accepts **1–4 short texts per HTTP call** and
merges concurrent calls into **single-flight batches of at most 8 texts** for a
single-slot CPU embedding backend. It ships with a controllable local mock
backend and a Docker Compose file that starts both.

```
client ──► gateway :8000 ──(one batch at a time)──► backend :9000
             │ batching, deadlines, ordering           │ configurable delay
             │ single-flight, error fan-out            │ error injection
             │ slot quarantine + fault ids             │ accepted-then-aborted
             └ /docs, /stats, /admin/fault            └ /docs, /slot, /admin/*
```

## Quick start

```bash
docker compose up --build
```

| Service | URL |
| --- | --- |
| Gateway API docs (interactive) | http://localhost:8000/docs |
| Gateway fault state | http://localhost:8000/admin/fault |
| Gateway stats | http://localhost:8000/stats |
| Gateway health | http://localhost:8000/healthz |
| Mock backend docs | http://localhost:9000/docs |
| Mock backend slot state | http://localhost:9000/slot |
| Mock backend stats (real concurrency) | http://localhost:9000/admin/stats |

The gateway's persistent state lives in the `gateway-state` named volume and
the mock backend's injected config in `backend-state`. They survive
`docker compose restart` and `docker compose down` (use `down -v` to erase
them deliberately).

No Docker? The same two apps run directly:

```bash
pip install -r requirements-dev.txt

uvicorn backend.app.main:app --port 9000 &
EMBEDDING_BACKEND_URL=http://127.0.0.1:9000/v1/embeddings \
EMBEDDING_BACKEND_ADMIN_URL=http://127.0.0.1:9000 \
GATEWAY_STATE_FILE=/tmp/gateway-state.json \
  uvicorn gateway.app.main:app --port 8000 &
```

Run the tests and the live smoke test:

```bash
pytest -q                      # no network required
python scripts/smoke_test.py   # requires both services running
```

## API

### `POST /v1/embeddings`

```json
{
  "input": ["first text", "second text"],
  "timeout_ms": 1000,
  "model": "local-demo"
}
```

* `input` — a string or a list of 1–4 strings. `texts` is accepted as an alias.
* `timeout_ms` — end-to-end deadline in milliseconds, **measured from the
  moment the gateway accepts the request** (a middleware stamps it before the
  body is parsed). Defaults to `GATEWAY_DEFAULT_TIMEOUT_MS` (5000) when omitted.
* `model` — optional; forwarded to the backend, otherwise `EMBEDDING_MODEL`.

Response (indices are always `0..n-1` in the caller's original input order):

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
  ],
  "model": "local-demo",
  "batch_id": "batch-7",
  "batch_size": 8
}
```

`batch_id`/`batch_size` describe the backend batch the caller was merged into
(useful for demos and tests).

| Status | `error.code` | Meaning |
| --- | --- | --- |
| 422 | `validation_error` | text count, `timeout_ms` or schema violation |
| 502 | `backend_error` | backend HTTP error, timeout, or invalid payload |
| 503 | `backend_unconfirmed` | dispatching paused after a batch whose completion could not be confirmed; includes `fault_id`, `reason`, `batch_id` |
| 503 | `service_unavailable` | gateway is shutting down |
| 504 | `request_timeout` | caller deadline hit while queueing or waiting |

When a backend batch fails without confirmation, the 502 for its own callers
*and* the 503 for everyone else carry the same `fault_id`:

```json
{"error": {"code": "backend_unconfirmed",
           "message": "dispatching paused after an unconfirmed backend batch (backend_unconfirmed); fault_id=fault-1a2b3c4d5e6f",
           "fault_id": "fault-1a2b3c4d5e6f",
           "reason": "backend_unconfirmed",
           "batch_id": "batch-4"}}
```

### Ops / fault endpoints

* `GET /healthz` — liveness. It stays `ok` while a fault is active because the
  process itself is healthy; readiness for embedding traffic is exposed by
  `dispatch_paused` on `/stats` and `/admin/fault`.
* `GET /stats` — counters (`accepted`, `completed`, `timed_out`,
  `disconnected`, `failed`, `batches_dispatched`, `backend_errors`,
  `unconfirmed_errors`, `rejected_requests`, `faults_created`, `recoveries`,
  `restored_faults`, `late_results_dropped`), `pending_requests`,
  `in_flight`, `dispatch_paused`, the active `fault`, and `last_batch`.
* `GET /admin/fault` — active fault or `active: false`, plus
  `persistence_enabled` and a hint.
* `POST /admin/fault/recover` — body `{"fault_id": "...", "operator": "...",
  "reason": "..."}`. Clears the quarantine **only** if `fault_id` is the
  currently active fault and (when `EMBEDDING_BACKEND_ADMIN_URL` is set) the
  backend reports `busy: false`. Otherwise `409`.

All three are visible in `/docs`.

## Batching semantics

1. **Merge window.** When the backend is idle, the first pending request opens
   a `GATEWAY_BATCH_WINDOW_MS` (20 ms) window starting when it became pending.
   Requests arriving inside the window join the batch FIFO.
2. **Size limit.** A batch never exceeds `GATEWAY_MAX_TEXTS_PER_BATCH` (8)
   texts. Requests are never split; a request only joins if it fits whole.
3. **No re-arming.** Requests that piled up while the backend was busy already
   elapsed their window, so they are merged and dispatched immediately once the
   backend frees up — no extra 20 ms per batch.
4. **Single-flight, with confirmation.** Exactly one dispatcher task exists, so
   at most one batch is *sent*. A new batch is not sent until the previous
   batch is **confirmed** complete: a full, valid response, or a complete
   error response, or a failure that provably never reached the backend.
5. **Deadline/abandon.** A request whose deadline expires or whose client
   disconnects while **pending** is removed from the queue and never sent. A
   request abandoned while **in flight** keeps its slot occupied until the
   backend answers or the fault machinery takes over; its late vectors are
   discarded, never delivered to the expired caller.
6. **Confirmed failures.** A complete non-2xx response, an invalid-but-complete
   payload, or a connect/pool timeout fails **all** waiters of that batch with
   `502 backend_error`, releases the slot, and leaves the gateway ready for the
   next request (unchanged from before).

### Unconfirmed batches and quarantine

A read timeout, a truncated/garbled response, or any other failure where the
backend may have **accepted the batch and kept computing** does *not* prove the
slot is free. In that case:

* callers of that batch still receive `502 backend_error` (the HTTP contract
  and the caller deadline are unchanged), now with `fault_id`, `reason` and
  `batch_id`;
* dispatching pauses. Every already-queued and every new request ends
  immediately with `503 backend_unconfirmed` carrying the same `fault_id` and
  the reason. No new batch is sent;
* `GET /admin/fault` shows the fault, its reason, and the batch it belongs to;
* an operator confirms the backend is idle and calls
  `POST /admin/fault/recover` with exactly that `fault_id`. The gateway probes
  `{EMBEDDING_BACKEND_ADMIN_URL}/slot` and refuses (409
  `backend_not_confirmed`) while the backend still reports `busy: true`;
* recovery uses the fault id and scheduler generation as compare-and-swap:
  an old recovery request cannot clear a newer fault, and a late callback from
  an older batch cannot release a newer batch's slot;
* on a clean success or confirmed error the slot is released normally and the
  merge/vector ordering behavior is unchanged.

Timeouts are enforced by the scheduler itself (`EMBEDDING_BACKEND_TIMEOUT_SECONDS`)
so the failure can be classified. The underlying HTTP request is allowed to
run for `EMBEDDING_BACKEND_LATE_RESPONSE_GRACE_SECONDS` longer; if it then
completes, the result is observed and discarded (counted in
`late_results_dropped`). It never resolves callers and never clears a fault.

### Restart safety

The last dispatch record and the active fault are written atomically
(`GATEWAY_STATE_FILE`, `os.replace` + fsync) **before** the batch is sent and
again when it is confirmed. On startup:

* an active fault is restored with the same `fault_id`
  (`restored_after_restart: true`) and dispatch stays paused;
* a record left at `confirmed: false` (the process crashed mid-batch) raises a
  new fault with reason `gateway_restart_with_unconfirmed_batch`;
* an unreadable/corrupt state file also fails closed
  (`reason: state_file_unreadable`) rather than assuming the slot is idle;
* a `confirmed: true` record restores normally, and requests are served.

Disconnect detection polls `Request.is_disconnected()` every
`GATEWAY_DISCONNECT_POLL_INTERVAL_MS`. On ASGI servers that cannot expose the
receive channel, probing is disabled rather than treated as a disconnect; the
deadline still bounds the caller's wait.

## Reproducing the bug and the fix

The mock backend's `accept_then_abort` mode accepts a batch, starts computing,
but cuts the response off mid-body. It keeps the slot busy for
`compute_after_abort_ms`, so `/slot` reports `busy: true` while the gateway has
already lost the answer — exactly the "backend accepted inference, response
lost" incident.

```bash
# 1. Backend: accept, abort the response, keep computing for 8 seconds
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"mode":"accept_then_abort","delay_ms":0,"compute_after_abort_ms":8000}'

# 2. This caller gets 502 with a fault id (before the fix it released the slot)
curl -s localhost:8000/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["accepted-but-lost"],"timeout_ms":5000}' | jq

# 3. Anyone else gets 503 with the same fault id, no new batch is sent
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/v1/embeddings \
  -H 'content-type: application/json' -d '{"input":["next"]}'
curl -s localhost:9000/admin/stats | jq '.total_requests, .max_concurrency'
curl -s localhost:9000/slot | jq

# 4. Recovery is refused while the backend is still busy
FAULT=$(curl -s localhost:8000/admin/fault | jq -r .fault.fault_id)
curl -s -X POST localhost:8000/admin/fault/recover -H 'content-type: application/json' \
  -d "{\"fault_id\":\"$FAULT\"}" | jq

# 5. Once /slot says busy:false, recover and serve traffic again
curl -s -X POST localhost:8000/admin/fault/recover -H 'content-type: application/json' \
  -d "{\"fault_id\":\"$FAULT\",\"operator\":\"oncall\",\"reason\":\"idle\"}" | jq
curl -s localhost:8000/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["after"]}' | jq '.data[0].index'
```

Other useful checks:

```bash
# Merge + order
curl -s localhost:8000/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["a","b"],"timeout_ms":1000}' | jq
curl -s localhost:9000/admin/stats | jq '.max_concurrency, .max_batch_seen'

# Caller deadline expiry (the batch is drained, the next one waits)
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"delay_ms":800,"mode":"ok"}'
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/v1/embeddings \
  -H 'content-type: application/json' -d '{"input":["slow"],"timeout_ms":100}'

# A complete error response releases the slot without a fault
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"mode":"error"}'
curl -s localhost:8000/admin/fault | jq .active
```

## Configuration

Gateway (environment variables):

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_BACKEND_URL` | `http://127.0.0.1:9000/v1/embeddings` | backend endpoint |
| `EMBEDDING_BACKEND_ADMIN_URL` | *unset* | base URL probed at `/slot` before recovery |
| `EMBEDDING_MODEL` | `local-demo` | default model name |
| `EMBEDDING_API_KEY` | *unset* | sent as `Authorization: Bearer …` |
| `EMBEDDING_BACKEND_TIMEOUT_SECONDS` | `30` | read timeout; exceeding it is an *unconfirmed* failure |
| `EMBEDDING_BACKEND_CONNECT_TIMEOUT_SECONDS` | `5` | connect timeout (confirmed failure) |
| `EMBEDDING_BACKEND_LATE_RESPONSE_GRACE_SECONDS` | `30` | how long an abandoned attempt may run so a late answer can be discarded |
| `GATEWAY_STATE_FILE` | *unset (in-memory)* | durable fault/last-dispatch record; set to a volume in compose |
| `GATEWAY_RECOVERY_PROBE_TIMEOUT_SECONDS` | `3` | `/slot` probe timeout |
| `GATEWAY_BATCH_WINDOW_MS` | `20` | idle merge window |
| `GATEWAY_MAX_TEXTS_PER_REQUEST` | `4` | max texts per HTTP call |
| `GATEWAY_MAX_TEXTS_PER_BATCH` | `8` | max texts per backend batch |
| `GATEWAY_DEFAULT_TIMEOUT_MS` | `5000` | default deadline |
| `GATEWAY_MAX_TIMEOUT_MS` | `600000` | max accepted deadline |
| `GATEWAY_DISCONNECT_POLL_INTERVAL_MS` | `20` | disconnect poll cadence |

Mock backend (also editable live via `POST /admin/config`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOCK_DELAY_MS` | `300` in compose | simulated inference latency |
| `MOCK_ERROR_RATE` | `0.0` | random failure probability |
| `MOCK_ERROR_STATUS` | `503` | status used for injected errors |
| `MOCK_MODE` | `ok` | `ok`, `error`, `malformed`, `short`, `duplicate_index`, `shuffle`, `accept_then_abort` |
| `MOCK_SHUFFLE_RESPONSE` | `false` | deliberately reverse `data` order |
| `MOCK_MAX_INPUTS` | `8` | returns 400 for larger batches |
| `MOCK_DIM` | `8` | embedding dimension |
| `MOCK_COMPUTE_AFTER_ABORT_MS` | `2000` in compose | how long an aborted batch keeps the slot busy |
| `MOCK_STATE_FILE` | `/var/lib/embedding-backend/config.json` in the image | persists the injected config across restarts |

The mock logs every request and tracks `max_concurrency`,
`concurrency_violations`, `oversized_batches`, `max_batch_seen`,
`disconnected_responses`, `detached_computations` and a request log at
`GET /admin/stats`. Because it is single-slot in intent, any overlap above 1
(or any batch > `max_inputs`) is recorded as a violation; `GET /slot` exposes
`busy` including detached computations.

## Project layout

```
gateway/app/config.py    settings from env
gateway/app/batcher.py   single-flight queue, merge window, deadlines,
                         quarantine, recovery, late-callback guards
gateway/app/state.py     atomic durable state (fault + last dispatch)
gateway/app/main.py      FastAPI app, routes, acceptance-time middleware
gateway/app/schemas.py   request/response models
gateway/app/errors.py    error types incl. fault-bearing 502/503/409
backend/app/main.py      controllable mock backend (+ accept_then_abort, /slot)
tests/                   batching, window, timeout, disconnect, error, quarantine
scripts/smoke_test.py    live end-to-end checks against running services
docker-compose.yml       gateway + mock backend, healthchecks, named volumes
```

# Verification summary

`pytest -q` (39 tests) covers merge sizing (never > 8), request-order
preservation, window timing, pending-expiry removal, single-flight after every
caller left, late-vector discard, error fan-out with recovery, malformed
backend payloads, disconnect abandonment, and the new quarantine behavior:
truncated responses and read timeouts pause dispatching with a fault id,
queued and new callers get 503 with that id, confirmed failures still release
the slot, recovery requires the active fault id and an idle backend, stale
recovery requests and late callbacks cannot clear a newer fault, and the
fault/`confirmed: false` slot survives a restart (including a corrupt state
file failing closed).

## Known limitations

* Recovery is manual by design. The gateway never assumes an unconfirmed batch
  finished just because the backend later looks idle; an operator must call
  `POST /admin/fault/recover`.
* The `/slot` probe is only as trustworthy as the backend's admin endpoint.
  When `EMBEDDING_BACKEND_ADMIN_URL` is unset, recovery relies purely on the
  operator's confirmation.
* A late response that arrives after
  `EMBEDDING_BACKEND_LATE_RESPONSE_GRACE_SECONDS` cannot be observed at all;
  it is simply lost, which is functionally identical to the unconfirmed case
  (the fault remains until recovery).
* `batch_size`/`batch_id` are informational; clients should rely on `index`
  ordering only.
* Disconnect-abandonment is best effort and depends on the ASGI server
  surfacing `http.disconnect` (verified with uvicorn).
* Batching state is per process. Keep one uvicorn worker (as in the Docker
  image) unless each replica has its own backend slot; otherwise two workers
  can issue one batch each concurrently. The state file is not a lock either:
  two gateway processes sharing it would race.
* The mock backend is for local verification only — no auth, no multi-process
  coordination.
